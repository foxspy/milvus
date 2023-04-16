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

#include <string>
#include <thread>
#include "index/ScalarIndexSort.h"
#include "index/StringIndexSort.h"

#include "common/SystemProperty.h"
#include "segcore/FieldIndexing.h"
#include "index/VectorMemNMIndex.h"
#include "utils/TimeProfiler.h"
#include "IndexConfigGenerator.h"

namespace milvus::segcore {

VectorFieldIndexing::VectorFieldIndexing(const FieldMeta& field_meta, const FieldIndexMeta& field_index_meta, int64_t segment_max_row_count, const SegcoreConfig& segcore_config): FieldIndexing(field_meta, segcore_config), config_(std::make_unique<VecIndexConfig>(segment_max_row_count, field_index_meta, segcore_config)), sync_with_index(false) {
}

void
VectorFieldIndexing::BuildIndexRange(int64_t ack_beg,
                                     int64_t ack_end,
                                     const VectorBase* vec_base) {
    AssertInfo(field_meta_.get_data_type() == DataType::VECTOR_FLOAT,
               "Data type of vector field is not VECTOR_FLOAT");
    auto dim = field_meta_.get_dim();

    auto source = dynamic_cast<const ConcurrentVector<FloatVector>*>(vec_base);
    AssertInfo(source, "vec_base can't cast to ConcurrentVector type");
    auto num_chunk = source->num_chunk();
    AssertInfo(ack_end <= num_chunk, "ack_end is bigger than num_chunk");
    auto conf = get_build_params();
    data_.grow_to_at_least(ack_end);
    for (int chunk_id = ack_beg; chunk_id < ack_end; chunk_id++) {
        const auto& chunk = source->get_chunk(chunk_id);
        auto indexing = std::make_unique<index::VectorMemNMIndex>(
            knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, knowhere::metric::L2);
        auto dataset = knowhere::GenDataSet(
            source->get_size_per_chunk(), dim, chunk.data());
        indexing->BuildWithDataset(dataset, conf);
        data_[chunk_id] = std::move(indexing);
    }
}

void
VectorFieldIndexing::AppendSegmentIndex(int64_t reserved_offset,
                                        int64_t size,
                                        const VectorBase* vec_base,
                                        const void* data_source) {
    AssertInfo(field_meta_.get_data_type() == DataType::VECTOR_FLOAT,
               "Data type of vector field is not VECTOR_FLOAT");

    auto dim = field_meta_.get_dim();
    auto conf = get_build_params();
    auto source = dynamic_cast<const ConcurrentVector<FloatVector>*>(vec_base);

    auto per_chunk = source->get_size_per_chunk();
    //append vector [vector_id_beg, vector_id_end] into index
    //build index [vector_id_beg, build_threshold] when index not exist
    if (!index_.get()) {
        idx_t vector_id_beg = index_cur_.load();
        idx_t vector_id_end = get_build_threshold();
        auto chunk_id_beg =  vector_id_beg / per_chunk;
        auto chunk_id_end =  vector_id_end / per_chunk;

        int64_t vec_num = vector_id_end - vector_id_beg + 1;
        TimeProfiler profiler("AppendSegmentIndex-Build[" + collection_name_ + "][" + config_->getIndexType() + "]:" + rangeStr(vector_id_beg, vec_num));
        const void *data_addr;
        std::unique_ptr<float[]> vec_data;
        //all train data in one chunk
        if (chunk_id_beg == chunk_id_end) {
            data_addr = vec_base->get_chunk_data(chunk_id_beg);
        } else {
            //merge data from multiple chunks together
            vec_data = std::make_unique<float[]>(vec_num * dim);
            int64_t offset = 0;
            for (int chunk_id = chunk_id_beg; chunk_id <= chunk_id_end; chunk_id++) {
                int chunk_offset = 0;
                int chunk_copysz = chunk_id == chunk_id_end ? (vector_id_beg + vec_num) - chunk_id * per_chunk : per_chunk;
                std::memcpy(vec_data.get() + offset * dim, (const float *)vec_base->get_chunk_data(chunk_id) + chunk_offset * dim, chunk_copysz * dim * sizeof(float));
                offset += chunk_copysz;
            }
            data_addr = vec_data.get();
        }
        auto dataset = knowhere::GenDataSet(vec_num, dim, data_addr);
        auto indexing = std::make_unique<index::VectorMemIndex>(
            config_->getIndexType(),
            config_->getMetricType());
        indexing->BuildWithDataset(dataset, conf);
        index_cur_.fetch_add(vec_num);
        index_ = std::move(indexing);
        profiler.reportRate(vec_num);
    }
    //append rest data when index exist
    idx_t vector_id_beg = index_cur_.load();
    idx_t vector_id_end = reserved_offset + size - 1;
    auto chunk_id_beg =  vector_id_beg / per_chunk;
    auto chunk_id_end =  vector_id_end / per_chunk;

    int64_t vec_num = vector_id_end - vector_id_beg + 1;

    TimeProfiler profiler("AppendSegmentIndex-Append[" + collection_name_ + "][" +  config_->getIndexType() + "]:" + rangeStr(vector_id_beg, vec_num));
    if (sync_with_index.load()) {
        auto dataset = knowhere::GenDataSet(
            vec_num,
            dim, data_source);
        index_->AddWithDataset(dataset, conf);
        index_cur_.fetch_add(vec_num);
    } else {
        for (int chunk_id = chunk_id_beg; chunk_id <= chunk_id_end;
             chunk_id++) {
            int chunk_offset = chunk_id == chunk_id_beg
                                   ? index_cur_ - chunk_id * per_chunk
                                   : 0;
            int chunk_sz = chunk_id == chunk_id_end
                               ? vector_id_end % per_chunk - chunk_offset + 1
                               : chunk_id == chunk_id_beg ? per_chunk - chunk_offset
                                                          : per_chunk;
            auto dataset = knowhere::GenDataSet(
                chunk_sz,
                dim,
                (const float*)source->get_chunk_data(chunk_id) +
                    chunk_offset * dim);
            index_->AddWithDataset(dataset, conf);
            index_cur_.fetch_add(chunk_sz);
        }
        sync_with_index = true;
    }
    profiler.reportRate(vec_num);
}


knowhere::Json
VectorFieldIndexing::get_build_params() const {
    auto config = config_->getBuildBaseParams();
    config[knowhere::meta::DIM] = std::to_string(field_meta_.get_dim());
    config[knowhere::meta::NUM_BUILD_THREAD] = std::to_string(1);
    return config;
}

SearchInfo
VectorFieldIndexing::get_search_params(const SearchInfo& searchInfo) const {
    auto conf = config_->getSearchConf(searchInfo);
    return conf;
}

idx_t
VectorFieldIndexing::get_index_cursor() {
    return index_cur_.load();
}
bool
VectorFieldIndexing::sync_data_with_index() const {
    return sync_with_index.load();
}

template <typename T>
void
ScalarFieldIndexing<T>::BuildIndexRange(int64_t ack_beg,
                                        int64_t ack_end,
                                        const VectorBase* vec_base) {
    auto source = dynamic_cast<const ConcurrentVector<T>*>(vec_base);
    AssertInfo(source, "vec_base can't cast to ConcurrentVector type");
    auto num_chunk = source->num_chunk();
    AssertInfo(ack_end <= num_chunk, "Ack_end is bigger than num_chunk");
    data_.grow_to_at_least(ack_end);
    for (int chunk_id = ack_beg; chunk_id < ack_end; chunk_id++) {
        const auto& chunk = source->get_chunk(chunk_id);
        // build index for chunk
        // TODO
        if constexpr (std::is_same_v<T, std::string>) {
            auto indexing = index::CreateStringIndexSort();
            indexing->Build(vec_base->get_size_per_chunk(), chunk.data());
            data_[chunk_id] = std::move(indexing);
        } else {
            auto indexing = index::CreateScalarIndexSort<T>();
            indexing->Build(vec_base->get_size_per_chunk(), chunk.data());
            data_[chunk_id] = std::move(indexing);
        }
    }
}

std::unique_ptr<FieldIndexing>
CreateIndex(const FieldMeta& field_meta, const FieldIndexMeta& field_index_meta, int64_t segment_max_row_count, const SegcoreConfig& segcore_config) {
    if (field_meta.is_vector()) {
        if (field_meta.get_data_type() == DataType::VECTOR_FLOAT) {
            return std::make_unique<VectorFieldIndexing>(field_meta,
                                                         field_index_meta,
                                                         segment_max_row_count,
                                                         segcore_config);
        } else {
            // TODO
            PanicInfo("unsupported");
        }
    }
    switch (field_meta.get_data_type()) {
        case DataType::BOOL:
            return std::make_unique<ScalarFieldIndexing<bool>>(field_meta,
                                                               segcore_config);
        case DataType::INT8:
            return std::make_unique<ScalarFieldIndexing<int8_t>>(
                field_meta, segcore_config);
        case DataType::INT16:
            return std::make_unique<ScalarFieldIndexing<int16_t>>(
                field_meta, segcore_config);
        case DataType::INT32:
            return std::make_unique<ScalarFieldIndexing<int32_t>>(
                field_meta, segcore_config);
        case DataType::INT64:
            return std::make_unique<ScalarFieldIndexing<int64_t>>(
                field_meta, segcore_config);
        case DataType::FLOAT:
            return std::make_unique<ScalarFieldIndexing<float>>(field_meta,
                                                                segcore_config);
        case DataType::DOUBLE:
            return std::make_unique<ScalarFieldIndexing<double>>(
                field_meta, segcore_config);
        case DataType::VARCHAR:
            return std::make_unique<ScalarFieldIndexing<std::string>>(
                field_meta, segcore_config);
        default:
            PanicInfo("unsupported");
    }
}

}  // namespace milvus::segcore
