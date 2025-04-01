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

#include "index/VectorMemIndex.h"

#include <unistd.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "cachinglayer/Manager.h"
#include "common/Tracer.h"
#include "common/Types.h"
#include "common/type_c.h"
#include "fmt/format.h"

#include "index/Index.h"
#include "index/IndexInfo.h"
#include "index/Meta.h"
#include "index/Utils.h"
#include "common/EasyAssert.h"
#include "config/ConfigKnowhere.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/comp/time_recorder.h"
#include "common/BitsetView.h"
#include "common/Consts.h"
#include "common/FieldData.h"
#include "common/File.h"
#include "common/Slice.h"
#include "common/RangeSearchHelper.h"
#include "common/Utils.h"
#include "log/Log.h"
#include "storage/DataCodec.h"
#include "storage/MemFileManagerImpl.h"
#include "storage/ThreadPools.h"
#include "storage/Util.h"
#include "monitor/prometheus_client.h"
#include "cachinglayer/CacheSlot.h"
#include "segcore/storagev1translator/MemVecIndexTranslator.h"

namespace milvus::index {

template <typename T>
VectorMemIndex<T>::VectorMemIndex(
    const IndexType& index_type,
    const MetricType& metric_type,
    const IndexVersion& version,
    const storage::FileManagerContext& file_manager_context)
    : VectorIndex(index_type, metric_type, version) {
    CheckMetricTypeSupport<T>(metric_type);
    AssertInfo(!is_unsupported(index_type, metric_type),
               index_type + " doesn't support metric: " + metric_type);

    if (file_manager_context.Valid()) {
        file_manager_ =
            std::make_shared<storage::MemFileManagerImpl>(file_manager_context);
        segment_id_ = file_manager_context.fieldDataMeta.segment_id;
        field_id_ = file_manager_context.fieldDataMeta.field_id;
        AssertInfo(file_manager_ != nullptr, "create file manager failed!");
    }
    CheckCompatible(version);
    auto get_index_obj =
        knowhere::IndexFactory::Instance().Create<T>(GetIndexType(), version);
    // building_index_ is used only for writing
    if (get_index_obj.has_value()) {
        building_index_ = get_index_obj.value();
    } else {
        auto err = get_index_obj.error();
        if (err == knowhere::Status::invalid_index_error) {
            PanicInfo(ErrorCode::Unsupported, get_index_obj.what());
        }
        PanicInfo(ErrorCode::KnowhereError, get_index_obj.what());
    }
}

template <typename T>
knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>
VectorMemIndex<T>::VectorIterators(const milvus::DatasetPtr dataset,
                                   const knowhere::Json& conf,
                                   const milvus::BitsetView& bitset) const {
    return GetAccessor()->Get()->AnnIterator(dataset, conf, bitset);
}

template <typename T>
IndexStatsPtr
VectorMemIndex<T>::Upload(const Config& config) {
    auto binary_set = Serialize(config);
    file_manager_->AddFile(binary_set);

    auto remote_paths_to_size = file_manager_->GetRemotePathsToFileSize();
    return IndexStats::NewFromSizeMap(file_manager_->GetAddedTotalMemSize(),
                                      remote_paths_to_size);
}

template <typename T>
BinarySet
VectorMemIndex<T>::Serialize(const Config& config) {
    knowhere::BinarySet ret;
    auto stat = GetAccessor()->Get()->Serialize(ret);
    if (stat != knowhere::Status::success)
        PanicInfo(ErrorCode::UnexpectedError,
                  "failed to serialize index: {}",
                  KnowhereStatusString(stat));
    Disassemble(ret);

    return ret;
}

template <typename T>
void
VectorMemIndex<T>::Load(const BinarySet& binary_set, const Config& config) {
    throw std::runtime_error("Load interface is deprecated");
    // milvus::Assemble(const_cast<BinarySet&>(binary_set));
    // LoadWithoutAssemble(binary_set, config);
}

template <typename T>
void
VectorMemIndex<T>::LoadFromFile(const Config& config) {
    std::unique_ptr<
        milvus::cachinglayer::Translator<knowhere::Index<knowhere::IndexNode>>>
        translator = std::make_unique<
            segcore::storagev1translator::MemVecIndexTranslator<T>>(
            segment_id_,
            field_id_,
            GetIndexType(),
            GetIndexVersion(),
            config,
            milvus::cachinglayer::StorageType::FILE_MMAP,
            file_manager_);
    cache_managed_index_ =
        milvus::cachinglayer::Manager::GetInstance().CreateCacheSlot(
            std::move(translator));
    cache_managed_index_->PinCells(&uid, 1);
}

template <typename T>
void
VectorMemIndex<T>::Load(milvus::tracer::TraceContext ctx,
                        const Config& config) {
    if (config.contains(MMAP_FILE_PATH)) {
        return LoadFromFile(config);
    }
    std::unique_ptr<
        milvus::cachinglayer::Translator<knowhere::Index<knowhere::IndexNode>>>
        translator = std::make_unique<
            segcore::storagev1translator::MemVecIndexTranslator<T>>(
            segment_id_,
            field_id_,
            GetIndexType(),
            GetIndexVersion(),
            config,
            milvus::cachinglayer::StorageType::MEMORY,
            file_manager_);
    cache_managed_index_ =
        milvus::cachinglayer::Manager::GetInstance().CreateCacheSlot(
            std::move(translator));
    cache_managed_index_->PinCells(&uid, 1);
}

template <typename T>
void
VectorMemIndex<T>::BuildWithDataset(const DatasetPtr& dataset,
                                    const Config& config) {
    knowhere::Json index_config;
    index_config.update(config);

    SetDim(dataset->GetDim());

    knowhere::TimeRecorder rc("BuildWithoutIds", 1);
    auto stat = GetAccessor()->Get()->Build(dataset, index_config);
    if (stat != knowhere::Status::success)
        PanicInfo(ErrorCode::IndexBuildError,
                  "failed to build index, " + KnowhereStatusString(stat));
    rc.ElapseFromBegin("Done");
    SetDim(GetAccessor()->Get()->Dim());
}

template <typename T>
void
VectorMemIndex<T>::Build(const Config& config) {
    auto insert_files =
        GetValueFromConfig<std::vector<std::string>>(config, "insert_files");
    AssertInfo(insert_files.has_value(),
               "insert file paths is empty when building in memory index");
    auto field_datas =
        file_manager_->CacheRawDataToMemory(insert_files.value());

    auto opt_fields = GetValueFromConfig<OptFieldT>(config, VEC_OPT_FIELDS);
    std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>> scalar_info;
    auto is_partition_key_isolation =
        GetValueFromConfig<bool>(config, "partition_key_isolation");
    if (opt_fields.has_value() &&
        GetAccessor()->Get()->IsAdditionalScalarSupported(
            is_partition_key_isolation.value_or(false))) {
        scalar_info = file_manager_->CacheOptFieldToMemory(opt_fields.value());
    }

    Config build_config;
    build_config.update(config);
    build_config.erase("insert_files");
    build_config.erase(VEC_OPT_FIELDS);
    if (!IndexIsSparse(GetIndexType())) {
        int64_t total_size = 0;
        int64_t total_num_rows = 0;
        int64_t dim = 0;
        for (auto data : field_datas) {
            total_size += data->Size();
            total_num_rows += data->get_num_rows();
            AssertInfo(dim == 0 || dim == data->get_dim(),
                       "inconsistent dim value between field datas!");
            dim = data->get_dim();
        }

        auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[total_size]);
        int64_t offset = 0;
        // TODO: avoid copying
        for (auto data : field_datas) {
            std::memcpy(buf.get() + offset, data->Data(), data->Size());
            offset += data->Size();
            data.reset();
        }
        field_datas.clear();

        auto dataset = GenDataset(total_num_rows, dim, buf.get());
        if (!scalar_info.empty()) {
            dataset->Set(knowhere::meta::SCALAR_INFO, std::move(scalar_info));
        }
        BuildWithDataset(dataset, build_config);
    } else {
        // sparse
        int64_t total_rows = 0;
        int64_t dim = 0;
        for (auto field_data : field_datas) {
            total_rows += field_data->Length();
            dim = std::max(
                dim,
                std::dynamic_pointer_cast<FieldData<SparseFloatVector>>(
                    field_data)
                    ->Dim());
        }
        std::vector<knowhere::sparse::SparseRow<float>> vec(total_rows);
        int64_t offset = 0;
        for (auto field_data : field_datas) {
            auto ptr = static_cast<const knowhere::sparse::SparseRow<float>*>(
                field_data->Data());
            AssertInfo(ptr, "failed to cast field data to sparse rows");
            for (size_t i = 0; i < field_data->Length(); ++i) {
                // this does a deep copy of field_data's data.
                // TODO: avoid copying by enforcing field data to give up
                // ownership.
                AssertInfo(dim >= ptr[i].dim(), "bad dim");
                vec[offset + i] = ptr[i];
            }
            offset += field_data->Length();
        }
        auto dataset = GenDataset(total_rows, dim, vec.data());
        dataset->SetIsSparse(true);
        if (!scalar_info.empty()) {
            dataset->Set(knowhere::meta::SCALAR_INFO, std::move(scalar_info));
        }
        BuildWithDataset(dataset, build_config);
    }
}

template <typename T>
void
VectorMemIndex<T>::AddWithDataset(const DatasetPtr& dataset,
                                  const Config& config) {
    knowhere::Json index_config;
    index_config.update(config);

    knowhere::TimeRecorder rc("AddWithDataset", 1);
    auto stat = GetAccessor()->Get()->Add(dataset, index_config);
    if (stat != knowhere::Status::success)
        PanicInfo(ErrorCode::IndexBuildError,
                  "failed to append index, " + KnowhereStatusString(stat));
    rc.ElapseFromBegin("Done");
}

template <typename T>
void
VectorMemIndex<T>::Query(const DatasetPtr dataset,
                         const SearchInfo& search_info,
                         const BitsetView& bitset,
                         SearchResult& search_result) const {
    //    AssertInfo(GetMetricType() == search_info.metric_type_,
    //               "Metric type of field index isn't the same with search info");

    auto num_queries = dataset->GetRows();
    knowhere::Json search_conf = PrepareSearchParams(search_info);
    auto topk = search_info.topk_;

    auto accessor = GetAccessor();
    auto index = accessor->Get();

    // TODO :: check dim of search data
    auto final = [&] {
        auto index_type = GetIndexType();
        if (CheckAndUpdateKnowhereRangeSearchParam(
                search_info, topk, GetMetricType(), search_conf)) {
            milvus::tracer::AddEvent("start_knowhere_index_range_search");
            auto res = index->RangeSearch(dataset, search_conf, bitset);
            milvus::tracer::AddEvent("finish_knowhere_index_range_search");
            if (!res.has_value()) {
                PanicInfo(ErrorCode::UnexpectedError,
                          "failed to range search: {}: {}",
                          KnowhereStatusString(res.error()),
                          res.what());
            }
            auto result = ReGenRangeSearchResult(
                res.value(), topk, num_queries, GetMetricType());
            milvus::tracer::AddEvent("finish_ReGenRangeSearchResult");
            return result;
        } else {
            milvus::tracer::AddEvent("start_knowhere_index_search");
            auto res = index->Search(dataset, search_conf, bitset);
            milvus::tracer::AddEvent("finish_knowhere_index_search");
            if (!res.has_value()) {
                PanicInfo(
                    ErrorCode::UnexpectedError,
                    // escape json brace in case of using message as format
                    "failed to search: config={{{}}} {}: {}",
                    search_conf.dump(),
                    KnowhereStatusString(res.error()),
                    res.what());
            }
            return res.value();
        }
    }();

    auto ids = final->GetIds();
    float* distances = const_cast<float*>(final->GetDistance());
    final->SetIsOwner(true);
    auto round_decimal = search_info.round_decimal_;
    auto total_num = num_queries * topk;

    if (round_decimal != -1) {
        const float multiplier = pow(10.0, round_decimal);
        for (int i = 0; i < total_num; i++) {
            distances[i] = std::round(distances[i] * multiplier) / multiplier;
        }
    }
    search_result.seg_offsets_.resize(total_num);
    search_result.distances_.resize(total_num);
    search_result.total_nq_ = num_queries;
    search_result.unity_topK_ = topk;
    std::copy_n(ids, total_num, search_result.seg_offsets_.data());
    std::copy_n(distances, total_num, search_result.distances_.data());
}

template <typename T>
const bool
VectorMemIndex<T>::HasRawData() const {
    return GetAccessor()->Get()->HasRawData(GetMetricType());
}

template <typename T>
std::vector<uint8_t>
VectorMemIndex<T>::GetVector(const DatasetPtr dataset) const {
    auto index_type = GetIndexType();
    if (IndexIsSparse(index_type)) {
        PanicInfo(ErrorCode::UnexpectedError,
                  "failed to get vector, index is sparse");
    }

    auto res = GetAccessor()->Get()->GetVectorByIds(dataset);
    if (!res.has_value()) {
        PanicInfo(ErrorCode::UnexpectedError,
                  "failed to get vector, " + KnowhereStatusString(res.error()));
    }
    auto tensor = res.value()->GetTensor();
    auto row_num = res.value()->GetRows();
    auto dim = res.value()->GetDim();
    int64_t data_size = milvus::GetVecRowSize<T>(dim) * row_num;
    std::vector<uint8_t> raw_data;
    raw_data.resize(data_size);
    memcpy(raw_data.data(), tensor, data_size);
    return raw_data;
}

template <typename T>
std::unique_ptr<const knowhere::sparse::SparseRow<float>[]>
VectorMemIndex<T>::GetSparseVector(const DatasetPtr dataset) const {
    auto res = GetAccessor()->Get()->GetVectorByIds(dataset);
    if (!res.has_value()) {
        PanicInfo(ErrorCode::UnexpectedError,
                  "failed to get vector, " + KnowhereStatusString(res.error()));
    }
    // release and transfer ownership to the result unique ptr.
    res.value()->SetIsOwner(false);
    return std::unique_ptr<const knowhere::sparse::SparseRow<float>[]>(
        static_cast<const knowhere::sparse::SparseRow<float>*>(
            res.value()->GetTensor()));
}

template class VectorMemIndex<float>;
template class VectorMemIndex<bin1>;
template class VectorMemIndex<float16>;
template class VectorMemIndex<bfloat16>;
template class VectorMemIndex<int8>;

}  // namespace milvus::index
