// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <optional>
#include <map>
#include <memory>

#include <tbb/concurrent_vector.h>
#include <index/Index.h>
#include <index/ScalarIndex.h>

#include "AckResponder.h"
#include "InsertRecord.h"
#include "common/Schema.h"
#include "segcore/SegcoreConfig.h"
#include "index/VectorIndex.h"

namespace milvus::segcore {

// this should be concurrent
// All concurrent
class FieldIndexing {
 public:
    explicit FieldIndexing(const FieldMeta& field_meta, const SegcoreConfig& segcore_config)
        : field_meta_(field_meta), segcore_config_(segcore_config) {
    }
    FieldIndexing(const FieldIndexing&) = delete;
    FieldIndexing&
    operator=(const FieldIndexing&) = delete;
    virtual ~FieldIndexing() = default;

    virtual void
    AppendSegmentIndex(int64_t reserved_offset, int64_t size, const VectorBase* vec_base) = 0;

    const FieldMeta&
    get_field_meta() {
        return field_meta_;
    }

    virtual idx_t
    get_index_cursor() = 0;

    int64_t
    get_size_per_chunk() const {
        return segcore_config_.get_chunk_rows();
    }

    virtual index::IndexBase*
    get_chunk_indexing(int64_t chunk_id) const = 0;

    virtual index::IndexBase*
    get_segment_indexing() const = 0;

 protected:
    // additional info
    const FieldMeta& field_meta_;
    const SegcoreConfig& segcore_config_;
};

template <typename T>
class ScalarFieldIndexing : public FieldIndexing {
 public:
    using FieldIndexing::FieldIndexing;

    void
    AppendSegmentIndex(int64_t reserved_offset, int64_t size, const VectorBase* vec_base) override {
        PanicInfo("scalar index don't support append segment index");
    }
    idx_t
    get_index_cursor() override {
        return 0;
    }

    // concurrent
    index::ScalarIndex<T>*
    get_chunk_indexing(int64_t chunk_id) const override {
        Assert(!field_meta_.is_vector());
        return data_.at(chunk_id).get();
    }

    index::IndexBase*
    get_segment_indexing() const override {
        return nullptr;
    }

 private:
    tbb::concurrent_vector<index::ScalarIndexPtr<T>> data_;
};

class VectorFieldIndexing : public FieldIndexing {
 public:
    using FieldIndexing::FieldIndexing;

    void
    AppendSegmentIndex(int64_t reserved_offset, int64_t size, const VectorBase* vec_base) override;

    // concurrent
    index::IndexBase*
    get_chunk_indexing(int64_t chunk_id) const override {
        Assert(field_meta_.is_vector());
        return data_.at(chunk_id).get();
    }
    index::IndexBase*
    get_segment_indexing() const override {
        return index_.get();
    }

    idx_t
    get_index_cursor() override;

    knowhere::Json
    get_build_params(const knowhere::IndexType& index_type) const;

    knowhere::Config
    get_search_params(int top_k) const;

    knowhere::Json
    get_search_params(const knowhere::IndexType& index_type) const;

    knowhere::Json
    get_search_params(const knowhere::IndexType& indexType, int top_K) const;

 private:
    std::atomic<idx_t> index_cur_ = 0;
    std::unique_ptr<index::VectorIndex> index_;
    tbb::concurrent_vector<std::unique_ptr<index::VectorIndex>> data_;
};

std::unique_ptr<FieldIndexing>
CreateIndex(const FieldMeta& field_meta, const SegcoreConfig& segcore_config);

class IndexingRecord {
 public:
    explicit IndexingRecord(const Schema& schema, const SegcoreConfig& segcore_config)
        : schema_(schema), segcore_config_(segcore_config) {
        Initialize();
    }

    void
    Initialize() {
        int offset_id = 0;
        for (auto& [field_id, field_meta] : schema_.get_fields()) {
            ++offset_id;

            if (field_meta.is_vector()) {
                // TODO: skip binary small index now, reenable after config.yaml is ready
                if (field_meta.get_data_type() == DataType::VECTOR_BINARY) {
                    continue;
                }
                // flat should be skipped
                if (!field_meta.get_metric_type().has_value()) {
                    continue;
                }
            }

            field_indexings_.try_emplace(field_id, CreateIndex(field_meta, segcore_config_));
        }
        assert(offset_id == schema_.size());
    }

    template <bool is_sealed>
    void
    AppendingIndex(int64_t reserved_offset, int64_t size, const InsertRecord<is_sealed>& record) {
        for (auto& [field_offset, entry] : field_indexings_) {
            if (entry->get_field_meta().is_vector()) {
                auto vec_base = record.get_field_data_base(field_offset);
                entry->AppendSegmentIndex(reserved_offset, size, vec_base);
            }
        }
    }

    // concurrent
    int64_t
    get_finished_ack() const {
        return finished_ack_.GetAck();
    }

    const FieldIndexing&
    get_field_indexing(FieldId field_id) const {
        Assert(field_indexings_.count(field_id));
        return *field_indexings_.at(field_id);
    }

    const VectorFieldIndexing&
    get_vec_field_indexing(FieldId field_id) const {
        auto& field_indexing = get_field_indexing(field_id);
        auto ptr = dynamic_cast<const VectorFieldIndexing*>(&field_indexing);
        AssertInfo(ptr, "invalid indexing");
        return *ptr;
    }

    bool
    is_in(FieldId field_id) const {
        return field_indexings_.count(field_id);
    }

    template <typename T>
    auto
    get_scalar_field_indexing(FieldId field_id) const -> const ScalarFieldIndexing<T>& {
        auto& entry = get_field_indexing(field_id);
        auto ptr = dynamic_cast<const ScalarFieldIndexing<T>*>(&entry);
        AssertInfo(ptr, "invalid indexing");
        return *ptr;
    }

 private:
    const Schema& schema_;
    const SegcoreConfig& segcore_config_;

 private:
    // control info
    std::atomic<int64_t> resource_ack_ = 0;
    //    std::atomic<int64_t> finished_ack_ = 0;
    AckResponder finished_ack_;
    std::mutex mutex_;

 private:
    // field_offset => indexing
    std::map<FieldId, std::unique_ptr<FieldIndexing>> field_indexings_;
};

}  // namespace milvus::segcore
