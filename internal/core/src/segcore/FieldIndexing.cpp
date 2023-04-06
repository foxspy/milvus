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

namespace milvus::segcore {


void
VectorFieldIndexing::AppendSegmentIndex(int64_t reserved_offset, int64_t size,
                                        const VectorBase* vec_base) {
    AssertInfo(field_meta_.get_data_type() == DataType::VECTOR_FLOAT,
               "Data type of vector field is not VECTOR_FLOAT");

    auto dim = field_meta_.get_dim();
    auto conf = get_build_params(segcore_config_.get_index_type());
    auto source = dynamic_cast<const ConcurrentVector<FloatVector>*>(vec_base);

    auto per_chunk = source->get_size_per_chunk();
    //append vector [vector_id_beg, vector_id_end] into index
    idx_t vector_id_beg = index_cur_.load();
    idx_t vector_id_end = reserved_offset + size - 1;
    auto chunk_id_beg =  vector_id_beg / per_chunk;
    auto chunk_id_end =  vector_id_end / per_chunk;

    int64_t vec_num = vector_id_end - vector_id_beg + 1;

    //build index when not exist
    if (!index_.get()) {
        std::string index_type = segcore_config_.get_index_type();
        TimeProfiler profiler("AppendSegmentIndex-Build[" + index_type +  "]:" + rangeStr(vector_id_beg, vec_num));
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
            segcore_config_.get_index_type(),
            segcore_config_.get_metric_type());
        indexing->BuildWithDataset(dataset, conf);
        index_cur_.fetch_add(vec_num);
        index_ = std::move(indexing);
        profiler.reportRate(vec_num);
    } else {
        //append data when index exist
        std::string index_type = segcore_config_.get_index_type();
        TimeProfiler profiler("AppendSegmentIndex-Append["+ index_type + "]:" + rangeStr(vector_id_beg, vec_num));

        for (int chunk_id = chunk_id_beg; chunk_id <= chunk_id_end; chunk_id++) {
            int chunk_offset = chunk_id == chunk_id_beg ? index_cur_ - chunk_id * per_chunk : 0;
            int chunk_sz = chunk_id == chunk_id_end ? vector_id_end % per_chunk - chunk_offset + 1:
                           chunk_id == chunk_id_beg ? per_chunk - chunk_offset: per_chunk;
            auto dataset = knowhere::GenDataSet(chunk_sz, dim, (const float *)source->get_chunk_data(chunk_id) + chunk_offset * dim);
            index_->AppendDataset(dataset, conf);
            index_cur_.fetch_add(chunk_sz);
        }
        profiler.reportRate(vec_num);
    }
}


knowhere::Json
VectorFieldIndexing::get_build_params(const knowhere::IndexType& indexType) const {
    nlohmann::json base_params;
    base_params[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    base_params[knowhere::meta::DIM] = std::to_string(field_meta_.get_dim());
    if (indexType == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
        base_params[knowhere::indexparam::NLIST] = std::to_string(128);
    } else if (indexType == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC) {
        base_params[knowhere::indexparam::NLIST] = std::to_string(128);
        base_params["ssize"] = std::to_string(segcore_config_.segment_size_);
    }
    return base_params;
}

knowhere::Json
VectorFieldIndexing::get_search_params(const knowhere::IndexType& indexType, int top_K) const {
    // TODO : how to process other metric
    nlohmann::json base_params;
    base_params[knowhere::meta::TOPK] = top_K;
    base_params[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    if (indexType == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
        base_params[knowhere::indexparam::NPROBE] = std::to_string(10);
    } else if (indexType == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC) {
        base_params[knowhere::indexparam::NPROBE] = std::to_string(10);
    }
    return base_params;
}
idx_t
VectorFieldIndexing::get_index_cursor() {
    return index_cur_.load();
}

std::unique_ptr<FieldIndexing>
CreateIndex(const FieldMeta& field_meta, const SegcoreConfig& segcore_config) {
    if (field_meta.is_vector()) {
        if (field_meta.get_data_type() == DataType::VECTOR_FLOAT) {
            return std::make_unique<VectorFieldIndexing>(field_meta,
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
