// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <boost/dynamic_bitset.hpp>
#include "common/Types.h"
#include "knowhere/index/index_factory.h"
#include "index/VectorIndex.h"
#include "storage/MemFileManagerImpl.h"
#include "index/IndexInfo.h"
#include "cachinglayer/CacheSlot.h"

namespace milvus::index {

template <typename T>
class VectorMemIndex : public VectorIndex {
 public:
    explicit VectorMemIndex(
        const IndexType& index_type,
        const MetricType& metric_type,
        const IndexVersion& version,
        const IndexConstructType& index_construct_type,
        const storage::FileManagerContext& file_manager_context =
            storage::FileManagerContext());

    BinarySet
    Serialize(const Config& config) override;

    void
    Load(const BinarySet& binary_set, const Config& config = {}) override;

    void
    Load(milvus::tracer::TraceContext ctx, const Config& config = {}) override;

    void
    BuildWithDataset(const DatasetPtr& dataset,
                     const Config& config = {}) override;

    void
    Build(const Config& config = {}) override;

    void
    AddWithDataset(const DatasetPtr& dataset, const Config& config) override;

    int64_t
    Count() override {
        return GetAccessor()->Get()->Count();
    }

    void
    Query(const DatasetPtr dataset,
          const SearchInfo& search_info,
          const BitsetView& bitset,
          SearchResult& search_result) const override;

    const bool
    HasRawData() const override;

    std::vector<uint8_t>
    GetVector(const DatasetPtr dataset) const override;

    std::unique_ptr<const knowhere::sparse::SparseRow<float>[]>
    GetSparseVector(const DatasetPtr dataset) const override;

    IndexStatsPtr
    Upload(const Config& config = {}) override;

    knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>
    VectorIterators(const DatasetPtr dataset,
                    const knowhere::Json& json,
                    const BitsetView& bitset) const override;

 private:
    void
    LoadFromFile(const Config& config);

 protected:
    struct SingleCellAccessor {
        SingleCellAccessor(
            std::unique_ptr<milvus::cachinglayer::CellAccessor<
                knowhere::Index<knowhere::IndexNode>>> cell_accessor,
            std::shared_ptr<milvus::cachinglayer::CacheSlot<
                knowhere::Index<knowhere::IndexNode>>> cache_slot)
            : index_(nullptr),
              cell_accessor_(std::move(cell_accessor)),
              cache_slot_(std::move(cache_slot)) {
        }

        explicit SingleCellAccessor(knowhere::Index<knowhere::IndexNode>* index)
            : index_(index), cell_accessor_(nullptr), cache_slot_(nullptr) {
        }

        knowhere::Index<knowhere::IndexNode>*
        Get() {
            if (cell_accessor_ != nullptr) {
                return const_cast<knowhere::Index<knowhere::IndexNode>*>(
                    cell_accessor_->get_cell_of(0));
            }
            return index_;
        }

        const knowhere::Index<knowhere::IndexNode>*
        Get() const {
            if (cell_accessor_ != nullptr) {
                return cell_accessor_->get_cell_of(0);
            }
            return index_;
        }

     private:
        knowhere::Index<knowhere::IndexNode>* index_;
        std::unique_ptr<milvus::cachinglayer::CellAccessor<
            knowhere::Index<knowhere::IndexNode>>>
            cell_accessor_;
        std::shared_ptr<milvus::cachinglayer::CacheSlot<
            knowhere::Index<knowhere::IndexNode>>>
            cache_slot_;
    };

    std::unique_ptr<SingleCellAccessor>
    GetAccessor() const {
        if (IsConstructByLoad()) {
            return std::make_unique<SingleCellAccessor>(
                cache_managed_index_->PinCells(&uid, 1).get(),
                cache_managed_index_);
        }
        return std::make_unique<SingleCellAccessor>(
            const_cast<knowhere::Index<knowhere::IndexNode>*>(
                &building_index_));
    }

    std::unique_ptr<SingleCellAccessor>
    GetAccessor() {
        if (IsConstructByLoad()) {
            return std::make_unique<SingleCellAccessor>(
                cache_managed_index_->PinCells(&uid, 1).get(),
                cache_managed_index_);
        }
        return std::make_unique<SingleCellAccessor>(&building_index_);
    }

    Config config_;
    const static milvus::cachinglayer::uid_t uid = 0;
    // TODO: split vector index into buildable and non-buildable
    knowhere::Index<knowhere::IndexNode> building_index_;
    mutable std::shared_ptr<
        milvus::cachinglayer::CacheSlot<knowhere::Index<knowhere::IndexNode>>>
        cache_managed_index_;
    std::shared_ptr<storage::MemFileManagerImpl> file_manager_;

    int64_t segment_id_;
    int64_t field_id_;

    CreateIndexInfo create_index_info_;
};

template <typename T>
using VectorMemIndexPtr = std::unique_ptr<VectorMemIndex<T>>;
}  // namespace milvus::index
