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

#include <string.h>
#include <algorithm>

#include "common/FastMem.h"
#include <atomic>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <random>
#include <type_traits>
#include <utility>

#include "clustering/KmeansClustering.h"
#include "clustering/file_utils.h"
#include "common/Common.h"
#include "common/Consts.h"
#include "common/FieldDataInterface.h"
#include "common/Types.h"
#include "common/Utils.h"
#include "fmt/core.h"
#include "glog/logging.h"
#include "knowhere/cluster/cluster.h"
#include "knowhere/cluster/cluster_factory.h"
#include "knowhere/cluster/cluster_node.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_factory.h"
#include "log/Log.h"
#include "milvus-storage/filesystem/fs.h"
#include "nlohmann/json.hpp"
#include "pb/schema.pb.h"
#include "storage/RemoteInputStream.h"
#include "storage/RemoteOutputStream.h"

namespace milvus::clustering {

namespace {

class GlobalHeadIndexFileManager : public milvus::FileManager {
 public:
    GlobalHeadIndexFileManager(milvus_storage::ArrowFileSystemPtr fs,
                               std::string head_index_path)
        : fs_(std::move(fs)), head_index_path_(std::move(head_index_path)) {
    }

    bool
    LoadFile(const std::string&) override {
        return true;
    }

    bool
    AddFile(const std::string&) override {
        return false;
    }

    bool
    AddFileMeta(const milvus::FileMeta& file_meta) override {
        file_size_ = file_meta.file_size;
        return true;
    }

    std::optional<bool>
    IsExisted(const std::string&) override {
        return true;
    }

    bool
    RemoveFile(const std::string&) override {
        return false;
    }

    std::shared_ptr<milvus::InputStream>
    OpenInputStream(const std::string&) override {
        auto remote_file = fs_->OpenInputFile(head_index_path_);
        AssertInfo(remote_file.ok(),
                   "failed to open global head index {}, reason: {}",
                   head_index_path_,
                   remote_file.status().ToString());
        return std::static_pointer_cast<milvus::InputStream>(
            std::make_shared<milvus::storage::RemoteInputStream>(
                std::move(remote_file.ValueOrDie())));
    }

    std::shared_ptr<milvus::OutputStream>
    OpenOutputStream(const std::string&) override {
        AssertInfo(fs_, "fs is nullptr, cannot open global head index output");
        if (milvus_storage::IsLocalFileSystem(fs_)) {
            auto dir_pos = head_index_path_.find_last_of('/');
            if (dir_pos != std::string::npos && dir_pos > 0) {
                auto dir_path = head_index_path_.substr(0, dir_pos);
                auto status = fs_->CreateDir(dir_path, /*recursive=*/true);
                AssertInfo(
                    status.ok(),
                    "failed to create global head index dir {}, reason: {}",
                    dir_path,
                    status.ToString());
            }
        }
        auto remote_stream = fs_->OpenOutputStream(head_index_path_);
        AssertInfo(remote_stream.ok(),
                   "failed to open global head index output {}, reason: {}",
                   head_index_path_,
                   remote_stream.status().ToString());
        return std::make_shared<milvus::storage::RemoteOutputStream>(
            std::move(remote_stream.ValueOrDie()));
    }

    int64_t
    FileSize() const {
        return file_size_;
    }

 private:
    milvus_storage::ArrowFileSystemPtr fs_;
    std::string head_index_path_;
    int64_t file_size_ = 0;
};

class TempFileGuard {
 public:
    explicit TempFileGuard(std::string path) : path_(std::move(path)) {
    }

    ~TempFileGuard() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    const std::string&
    Path() const {
        return path_;
    }

 private:
    std::string path_;
};

knowhere::Json
BuildHeadIndexKnowhereConfig(
    int64_t dim,
    int64_t num_clusters,
    const std::string& data_path,
    const google::protobuf::Map<std::string, std::string>& cluster_params) {
    const auto graph_width =
        std::max<int64_t>(2, std::min<int64_t>(48, num_clusters - 1));
    const auto build_candidates = std::max<int64_t>(100, graph_width * 4);

    knowhere::Json config;
    config[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
    config[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    config[knowhere::meta::DIM] = dim;
    config[knowhere::meta::INDEX_PREFIX] = "global_head_index";
    config[knowhere::meta::INDEX_ENGINE_VERSION] = 9;
    config[knowhere::meta::DATA_PATH] = data_path;
    config["M"] = graph_width;
    config["efConstruction"] = build_candidates;
    config["graph_build_algo"] = "pipnn";
    config["max_degree"] = graph_width;
    config["search_list_size"] = build_candidates;
    config["index_algo"] = "GRAPH";
    config["build_quant_type"] = "NONE";
    config["search_quant_type"] = "NONE";
    config["refine_quant_type"] = "NONE";
    config["skip_refine"] = false;
    // knowhere.cluster.head_index_* overrides (strip prefix -> knowhere HNSW config key).
    const std::string head_prefix = "head_index_";
    for (const auto& kv : cluster_params) {
        if (kv.first.rfind(head_prefix, 0) == 0) {
            config[kv.first.substr(head_prefix.size())] = kv.second;
        }
    }
    return config;
}

template <typename T>
TempFileGuard
WriteHeadIndexTrainData(const T* centroids, int64_t num_clusters, int64_t dim) {
    static_assert(std::is_same_v<T, float>,
                  "global head index currently supports float centroids only");
    auto temp_path = (std::filesystem::temp_directory_path() /
                      fmt::format("global_head_index_{}_{}.bin",
                                  std::time(nullptr),
                                  reinterpret_cast<uintptr_t>(centroids)))
                         .string();
    std::ofstream writer(temp_path, std::ios::binary | std::ios::trunc);
    AssertInfo(writer.good(),
               "failed to create global head index train data: {}",
               temp_path);
    auto rows = static_cast<int32_t>(num_clusters);
    auto cols = static_cast<int32_t>(dim);
    writer.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    writer.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    writer.write(reinterpret_cast<const char*>(centroids),
                 num_clusters * dim * sizeof(T));
    writer.close();
    AssertInfo(writer.good(),
               "failed to write global head index train data: {}",
               temp_path);
    // [HEAD-CENTROID-DUMP] persist a copy of the exact centroids fed to the head
    // index build (same data/order as cluster compaction) for offline vecTool
    // analysis of why the head graph comes out empty. Guarded by env var.
    if (const char* dump_dir = std::getenv("GLOBAL_HEAD_CENTROID_DUMP_DIR")) {
        try {
            std::filesystem::create_directories(dump_dir);
            auto dump_path =
                (std::filesystem::path(dump_dir) /
                 fmt::format("head_centroids_{}_{}x{}_mem.fbin",
                             std::time(nullptr),
                             rows,
                             cols))
                    .string();
            std::filesystem::copy_file(
                temp_path,
                dump_path,
                std::filesystem::copy_options::overwrite_existing);
            LOG_INFO("[HEAD-CENTROID-DUMP] saved {} centroids (dim={}) to {}",
                     rows,
                     cols,
                     dump_path);
        } catch (const std::exception& e) {
            LOG_WARN("[HEAD-CENTROID-DUMP] failed to persist centroids: {}",
                     e.what());
        }
    }
    return TempFileGuard(temp_path);
}

template <typename T>
void
BuildAndUploadHeadIndex(
    const T* centroids,
    int64_t num_clusters,
    int64_t dim,
    const std::string& head_index_path,
    milvus_storage::ArrowFileSystemPtr fs,
    const google::protobuf::Map<std::string, std::string>& cluster_params) {
    AssertInfo(centroids != nullptr,
               "failed to build global head index, centroids is null");
    AssertInfo(!head_index_path.empty(),
               "failed to build global head index, path is empty");
    AssertInfo(fs != nullptr, "failed to build global head index, fs is null");

    auto train_data = WriteHeadIndexTrainData(centroids, num_clusters, dim);
    auto file_manager = std::make_shared<GlobalHeadIndexFileManager>(
        std::move(fs), head_index_path);
    auto file_manager_pack =
        knowhere::Pack(std::shared_ptr<milvus::FileManager>(file_manager));
    auto index_or = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        knowhere::IndexEnum::INDEX_HNSW, 9, file_manager_pack);
    AssertInfo(index_or.has_value(),
               "failed to create global head index: {}",
               index_or.what());

    auto head_index = std::move(index_or.value());
    auto dataset = knowhere::GenDataSet(num_clusters, dim, centroids);
    auto index_config =
        BuildHeadIndexKnowhereConfig(
            dim, num_clusters, train_data.Path(), cluster_params);
    auto build_status = head_index.Build(dataset, index_config);
    AssertInfo(build_status == knowhere::Status::success,
               "failed to build global head index, status: {}",
               KnowhereStatusString(build_status));

    knowhere::BinarySet binary_set;
    auto serialize_status = head_index.Serialize(binary_set);
    AssertInfo(serialize_status == knowhere::Status::success,
               "failed to serialize global head index, status: {}",
               KnowhereStatusString(serialize_status));
    LOG_INFO("global head index uploaded, path: {}, size: {}",
             head_index_path,
             file_manager->FileSize());
}

}  // namespace

KmeansClustering::KmeansClustering(
    const storage::FileManagerContext& file_manager_context) {
    file_manager_ =
        std::make_unique<storage::MemFileManagerImpl>(file_manager_context);
    fs_ = file_manager_context.fs;
    AssertInfo(file_manager_ != nullptr, "create file manager failed!");
    int64_t collection_id = file_manager_context.fieldDataMeta.collection_id;
    int64_t partition_id = file_manager_context.fieldDataMeta.partition_id;
    msg_header_ = fmt::format(
        "collection: {}, partition: {} ", collection_id, partition_id);
}

template <typename T>
void
KmeansClustering::FetchDataFiles(
    uint8_t* buf,
    const int64_t expected_train_size,
    const int64_t expected_remote_file_size,
    const std::vector<std::string>& files,
    const milvus::proto::clustering::SegmentStorageInfo* storage_info,
    const int64_t segment_id,
    const size_t storage_info_map_size,
    const int64_t first_storage_info_key,
    const int64_t dim,
    int64_t& offset) {
    // CacheRawDataToMemory mostly used as pull files from one segment
    // So we could assume memory is always enough for theses cases
    // But in clustering when we sample train data, first pre-allocate the large buffer(size controlled by config) for future knowhere usage
    // And we will have tmp memory usage at pulling stage, pull file(tmp memory) + memcpy to pre-allocated buffer, limit the batch here
    auto batch = size_t(DEFAULT_FIELD_MAX_MEMORY_LIMIT / FILE_SLICE_SIZE);
    int64_t fetched_file_size = 0;
    const bool has_storage_v2_info =
        storage_info != nullptr && (!storage_info->manifest().empty() ||
                                    storage_info->has_segment_insert_files());
    const int storage_insert_file_groups =
        storage_info != nullptr && storage_info->has_segment_insert_files()
            ? storage_info->segment_insert_files().field_insert_files_size()
            : 0;

    auto fetch = [&](const std::vector<std::string>& group_files) {
        Config config;
        config[INSERT_FILES_KEY] = group_files;
        config[DIM_KEY] = dim;
        config[DATA_TYPE_KEY] = DataType::VECTOR_FLOAT;
        config[ELEMENT_TYPE_KEY] = DataType::NONE;
        if (has_storage_v2_info) {
            config[STORAGE_VERSION_KEY] = storage_info->storage_version() > 0
                                              ? storage_info->storage_version()
                                              : STORAGE_V2;
            if (!storage_info->manifest().empty()) {
                config[SEGMENT_MANIFEST_KEY] = storage_info->manifest();
            } else if (storage_info->has_segment_insert_files()) {
                SegmentInsertFiles segment_insert_files;
                for (const auto& field_insert_files :
                     storage_info->segment_insert_files()
                         .field_insert_files()) {
                    std::vector<std::string> paths(
                        field_insert_files.file_paths().begin(),
                        field_insert_files.file_paths().end());
                    segment_insert_files.emplace_back(std::move(paths));
                }
                config[SEGMENT_INSERT_FILES_KEY] =
                    std::move(segment_insert_files);
            }
        }
        auto field_datas = file_manager_->CacheRawDataToMemory(config);

        for (auto& data : field_datas) {
            LOG_INFO(msg_header_ +
                         "fetched clustering file data rows: {}, length: {}, "
                         "bytes: {}, data type: {}",
                     data->get_num_rows(),
                     data->Length(),
                     data->Size(),
                     int(data->get_data_type()));
            size_t size = std::min(expected_train_size - offset, data->Size());
            if (size <= 0) {
                break;
            }
            fetched_file_size += size;
            milvus::fastmem::FastMemcpy(buf + offset, data->Data(), size);
            offset += size;
            data.reset();
        }
    };

    if (has_storage_v2_info) {
        fetch(files);
    } else {
        for (size_t i = 0; i < files.size(); i += batch) {
            size_t start = i;
            size_t end = std::min(files.size(), i + batch);
            std::vector<std::string> group_files(files.begin() + start,
                                                 files.begin() + end);
            fetch(group_files);
        }
    }
    // A short read is only an error if we did NOT deliberately stop because the
    // sampling buffer filled up. When training on a dataset larger than the train
    // budget (expected_train_size), the boundary segment is intentionally fetched
    // only partially (size is clamped to the remaining budget), so offset reaching
    // expected_train_size is a legitimate stop, not a truncated/corrupt file.
    AssertInfo(fetched_file_size == expected_remote_file_size ||
                   offset == expected_train_size,
               "file size inconsistent, expected: {}, actual: {}, "
               "storage_info: {}, has_storage_v2_info: {}, "
               "storage_version: {}, storage_insert_file_groups: {}, "
               "legacy_files: {}, segment_id: {}, storage_info_map_size: {}, "
               "first_storage_info_key: {}",
               expected_remote_file_size,
               fetched_file_size,
               storage_info != nullptr,
               has_storage_v2_info,
               storage_info == nullptr ? 0 : storage_info->storage_version(),
               storage_insert_file_groups,
               files.size(),
               segment_id,
               storage_info_map_size,
               first_storage_info_key);
}

template <typename T>
void
KmeansClustering::SampleTrainData(
    const std::vector<int64_t>& segment_ids,
    const std::map<int64_t, std::vector<std::string>>& segment_file_paths,
    const std::map<int64_t, milvus::proto::clustering::SegmentStorageInfo>&
        segment_storage_infos,
    const std::map<int64_t, int64_t>& segment_num_rows,
    const int64_t expected_train_size,
    const int64_t dim,
    const bool random_sample,
    uint8_t* buf) {
    int64_t offset = 0;
    std::vector<std::string> files;

    bool has_storage_v2_segments = false;
    for (const auto& [_, storage_info] : segment_storage_infos) {
        if (!storage_info.manifest().empty() ||
            storage_info.has_segment_insert_files()) {
            has_storage_v2_segments = true;
            break;
        }
    }

    if (random_sample && !has_storage_v2_segments) {
        for (auto& [segment_id, segment_files] : segment_file_paths) {
            for (auto& segment_file : segment_files) {
                files.emplace_back(segment_file);
            }
        }
        // shuffle files
        std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
        std::shuffle(files.begin(), files.end(), rng);
        FetchDataFiles<T>(buf,
                          expected_train_size,
                          expected_train_size,
                          files,
                          nullptr,
                          -1,
                          0,
                          -1,
                          dim,
                          offset);
        return;
    }

    // pick all segment_ids, no shuffle
    // and pull data once each segment to reuse the id mapping for assign stage
    for (auto i = 0; i < segment_ids.size(); i++) {
        if (offset == expected_train_size) {
            break;
        }
        int64_t cur_segment_id = segment_ids[i];
        files = segment_file_paths.at(cur_segment_id);
        std::sort(files.begin(),
                  files.end(),
                  [](const std::string& a, const std::string& b) {
                      return std::stol(a.substr(a.find_last_of("/") + 1)) <
                             std::stol(b.substr(b.find_last_of("/") + 1));
                  });
        const milvus::proto::clustering::SegmentStorageInfo* storage_info =
            nullptr;
        auto storage_it = segment_storage_infos.find(cur_segment_id);
        if (storage_it != segment_storage_infos.end()) {
            storage_info = &storage_it->second;
        }
        auto first_storage_info_key =
            segment_storage_infos.empty()
                ? -1
                : segment_storage_infos.begin()->first;
        LOG_INFO(msg_header_ +
                     "sample train data segment: {}, storage_info_hit: {}, "
                     "storage_info_map_size: {}",
                 cur_segment_id,
                 storage_info != nullptr,
                 segment_storage_infos.size());
        FetchDataFiles<T>(buf,
                          expected_train_size,
                          segment_num_rows.at(cur_segment_id) * dim * sizeof(T),
                          files,
                          storage_info,
                          cur_segment_id,
                          segment_storage_infos.size(),
                          first_storage_info_key,
                          dim,
                          offset);
    }
}

template <typename T>
milvus::proto::clustering::ClusteringCentroidsStats
KmeansClustering::CentroidsToPB(const T* centroids,
                                const int64_t num_clusters,
                                const int64_t dim) {
    milvus::proto::clustering::ClusteringCentroidsStats stats;
    for (auto i = 0; i < num_clusters; i++) {
        milvus::proto::schema::VectorField* vector_field =
            stats.add_centroids();
        vector_field->set_dim(dim);
        milvus::proto::schema::FloatArray* float_array =
            vector_field->mutable_float_vector();
        for (auto j = 0; j < dim; j++) {
            float_array->add_data(float(centroids[i * dim + j]));
        }
    }
    return stats;
}

std::vector<milvus::proto::clustering::ClusteringCentroidIdMappingStats>
KmeansClustering::CentroidIdMappingToPB(
    const uint32_t* centroid_id_mapping,
    const std::vector<int64_t>& segment_ids,
    const int64_t trained_segments_num,
    const std::map<int64_t, int64_t>& num_row_map,
    const int64_t num_clusters) {
    auto compute_num_in_centroid = [&](const uint32_t* centroid_id_mapping,
                                       uint64_t start,
                                       uint64_t end) -> std::vector<int64_t> {
        std::vector<int64_t> num_vectors(num_clusters, 0);
        for (uint64_t i = start; i < end; ++i) {
            num_vectors[centroid_id_mapping[i]]++;
        }
        return num_vectors;
    };
    std::vector<milvus::proto::clustering::ClusteringCentroidIdMappingStats>
        stats_arr;
    stats_arr.reserve(trained_segments_num);
    int64_t cur_offset = 0;
    for (auto i = 0; i < trained_segments_num; i++) {
        milvus::proto::clustering::ClusteringCentroidIdMappingStats stats;
        auto num_offset = num_row_map.at(segment_ids[i]);
        for (auto j = 0; j < num_offset; j++) {
            stats.add_centroid_id_mapping(centroid_id_mapping[cur_offset + j]);
        }
        auto num_vectors = compute_num_in_centroid(
            centroid_id_mapping, cur_offset, cur_offset + num_offset);
        for (uint64_t j = 0; j < num_clusters; j++) {
            stats.add_num_in_centroid(num_vectors[j]);
        }
        cur_offset += num_offset;
        stats_arr.emplace_back(std::move(stats));
    }
    return stats_arr;
}

milvus::proto::clustering::ClusteringCentroidIdMappingStats
KmeansClustering::CentroidIdMappingWithDistanceToPB(
    const int64_t* centroid_id_mapping,
    const float* distances,
    const int64_t num_rows,
    const int64_t num_clusters) {
    milvus::proto::clustering::ClusteringCentroidIdMappingStats stats;
    std::vector<int64_t> num_vectors(num_clusters, 0);
    for (int64_t i = 0; i < num_rows; ++i) {
        auto centroid = centroid_id_mapping[i];
        AssertInfo(centroid >= 0 && centroid < num_clusters,
                   "centroid id {} out of range [0, {})",
                   centroid,
                   num_clusters);
        stats.add_centroid_id_mapping(static_cast<uint32_t>(centroid));
        stats.add_distance_to_centroid(distances == nullptr ? 0.0f
                                                            : distances[i]);
        num_vectors[centroid]++;
    }
    for (int64_t i = 0; i < num_clusters; ++i) {
        stats.add_num_in_centroid(num_vectors[i]);
    }
    return stats;
}

template <typename T>
void
KmeansClustering::StreamingAssignandUpload(
    knowhere::Cluster<knowhere::ClusterNode>& cluster_node,
    const milvus::proto::clustering::AnalyzeInfo& config,
    const milvus::proto::clustering::ClusteringCentroidsStats& centroid_stats,
    const std::vector<
        milvus::proto::clustering::ClusteringCentroidIdMappingStats>&
        id_mapping_stats,
    const std::vector<int64_t>& segment_ids,
    const std::map<int64_t, std::vector<std::string>>& insert_files,
    const std::map<int64_t, int64_t>& num_rows,
    const int64_t dim,
    const int64_t trained_segments_num,
    const int64_t num_clusters) {
    auto byte_size = centroid_stats.ByteSizeLong();
    std::unique_ptr<uint8_t[]> data = std::make_unique<uint8_t[]>(byte_size);
    centroid_stats.SerializeToArray(data.get(), byte_size);
    std::unordered_map<std::string, int64_t> remote_paths_to_size;
    LOG_INFO(msg_header_ + "start upload cluster centroids file");
    auto centroid_remote_path =
        GetRemoteCentroidsObjectPrefix() + "/" + std::string(CENTROIDS_NAME);
    AddClusteringResultFiles(file_manager_->GetChunkManager().get(),
                             data.get(),
                             byte_size,
                             centroid_remote_path,
                             remote_paths_to_size);
    cluster_result_.centroid_path = std::move(centroid_remote_path);
    cluster_result_.centroid_file_size =
        remote_paths_to_size.at(cluster_result_.centroid_path);
    remote_paths_to_size.clear();
    LOG_INFO(msg_header_ + "upload cluster centroids file done");

    LOG_INFO(msg_header_ + "start upload cluster id mapping file");

    auto serializeIdMappingAndUpload = [&](const int64_t segment_id,
                                           const milvus::proto::clustering::
                                               ClusteringCentroidIdMappingStats&
                                                   id_mapping_pb) {
        auto byte_size = id_mapping_pb.ByteSizeLong();
        std::unique_ptr<uint8_t[]> data =
            std::make_unique<uint8_t[]>(byte_size);
        id_mapping_pb.SerializeToArray(data.get(), byte_size);
        AddClusteringResultFiles(
            file_manager_->GetChunkManager().get(),
            data.get(),
            byte_size,
            GetRemoteCentroidIdMappingObjectPrefix(segment_id) + "/" +
                std::string(OFFSET_MAPPING_NAME),
            remote_paths_to_size);
        LOG_INFO(
            msg_header_ +
                "upload segment {} cluster id mapping file with size {} B done",
            segment_id,
            byte_size);
    };

    for (size_t i = 0; i < segment_ids.size(); i++) {
        int64_t segment_id = segment_ids[i];
        // id mapping has been computed, just upload to remote
        if (i < trained_segments_num) {
            serializeIdMappingAndUpload(segment_id, id_mapping_stats[i]);
        } else {  // streaming download raw data, assign id mapping, then upload
            int64_t num_row = num_rows.at(segment_id);
            std::unique_ptr<T[]> buf = std::make_unique<T[]>(num_row * dim);
            int64_t offset = 0;
            FetchDataFiles<T>(reinterpret_cast<uint8_t*>(buf.get()),
                              INT64_MAX,
                              num_row * dim * sizeof(T),
                              insert_files.at(segment_id),
                              nullptr,
                              segment_id,
                              0,
                              -1,
                              dim,
                              offset);
            auto dataset = GenDataset(num_row, dim, buf.release());
            dataset->SetIsOwner(true);
            auto res = cluster_node.Assign(*dataset);
            if (!res.has_value()) {
                ThrowInfo(ErrorCode::UnexpectedError,
                          fmt::format("failed to kmeans assign: {}: {}",
                                      KnowhereStatusString(res.error()),
                                      res.what()));
            }
            res.value()->SetIsOwner(true);
            auto id_mapping =
                reinterpret_cast<const uint32_t*>(res.value()->GetTensor());

            auto id_mapping_pb = CentroidIdMappingToPB(
                id_mapping, {segment_id}, 1, num_rows, num_clusters)[0];
            serializeIdMappingAndUpload(segment_id, id_mapping_pb);
        }
    }
    LOG_INFO(msg_header_ + "upload cluster id mapping file done");
    cluster_result_.id_mappings = std::move(remote_paths_to_size);
    is_runned_ = true;
}

template <typename T>
void
KmeansClustering::Run(const milvus::proto::clustering::AnalyzeInfo& config) {
    std::map<int64_t, std::vector<std::string>> insert_files;
    for (const auto& pair : config.insert_files()) {
        std::vector<std::string> segment_files(
            pair.second.insert_files().begin(),
            pair.second.insert_files().end());
        insert_files[pair.first] = std::move(segment_files);
    }

    std::map<int64_t, int64_t> num_rows(config.num_rows().begin(),
                                        config.num_rows().end());
    auto num_clusters = config.num_clusters();
    AssertInfo(num_clusters > 0, "num clusters must larger than 0");
    auto train_size = config.train_size();
    AssertInfo(train_size > 0, "train size must larger than 0");
    auto dim = config.dim();
    auto min_cluster_ratio = config.min_cluster_ratio();
    AssertInfo(min_cluster_ratio > 0 && min_cluster_ratio < 1,
               "min cluster ratio must larger than 0, less than 1");
    auto max_cluster_ratio = config.max_cluster_ratio();
    AssertInfo(max_cluster_ratio > 1, "max cluster ratio must larger than 1");
    auto max_cluster_size = config.max_cluster_size();
    AssertInfo(max_cluster_size > 0, "max cluster size must larger than 0");

    auto cluster_node_obj =
        knowhere::ClusterFactory::Instance().Create<T>(KMEANS_CLUSTER);
    knowhere::Cluster<knowhere::ClusterNode> cluster_node;
    if (cluster_node_obj.has_value()) {
        cluster_node = std::move(cluster_node_obj.value());
    } else {
        auto err = cluster_node_obj.error();
        if (err == knowhere::Status::invalid_cluster_error) {
            throw SegcoreError(ErrorCode::ClusterSkip, cluster_node_obj.what());
        }
        throw SegcoreError(ErrorCode::KnowhereError, cluster_node_obj.what());
    }

    size_t data_num = 0;
    std::vector<int64_t> segment_ids;
    for (auto& [segment_id, num_row_each_segment] : num_rows) {
        data_num += num_row_each_segment;
        segment_ids.emplace_back(segment_id);
        AssertInfo(insert_files.find(segment_id) != insert_files.end(),
                   "segment id {} not exist in insert files",
                   segment_id);
    }
    size_t trained_segments_num = 0;

    size_t data_size = data_num * dim * sizeof(T);
    size_t train_num = train_size / sizeof(T) / dim;
    bool random_sample = true;
    // make train num equal to data num
    if (train_num >= data_num) {
        train_num = data_num;
        random_sample =
            false;  // all data are used for training, no need to random sampling
        trained_segments_num = segment_ids.size();
    }
    if (train_num < num_clusters) {
        LOG_WARN(msg_header_ +
                     "kmeans train num: {} less than num_clusters: {}, skip "
                     "clustering",
                 train_num,
                 num_clusters);
        throw SegcoreError(ErrorCode::ClusterSkip,
                           "sample data num less than num clusters");
    }

    size_t train_size_final = train_num * dim * sizeof(T);
    std::map<int64_t, milvus::proto::clustering::SegmentStorageInfo>
        segment_storage_infos(config.segment_storage_infos().begin(),
                              config.segment_storage_infos().end());
    knowhere::TimeRecorder rc(msg_header_ + "kmeans clustering",
                              2 /* log level: info */);
    // if data_num larger than max_train_size, we need to sample to make train data fits in memory
    // otherwise just load all the data for kmeans training
    LOG_INFO(msg_header_ + "pull and sample {}GB data out of {}GB data",
             train_size_final / 1024.0 / 1024.0 / 1024.0,
             data_size / 1024.0 / 1024.0 / 1024.0);
    auto buf = std::make_unique<uint8_t[]>(train_size_final);
    SampleTrainData<T>(segment_ids,
                       insert_files,
                       segment_storage_infos,
                       num_rows,
                       train_size_final,
                       dim,
                       random_sample,
                       buf.get());
    rc.RecordSection("sample done");

    auto dataset = GenDataset(train_num, dim, buf.release());
    dataset->SetIsOwner(true);

    LOG_INFO(msg_header_ + "train data num: {}, dim: {}, num_clusters: {}",
             train_num,
             dim,
             num_clusters);
    knowhere::Json train_conf;
    train_conf[NUM_CLUSTERS] = num_clusters;
    // inside knowhere, we will record each kmeans iteration duration
    // return id mapping
    auto res = cluster_node.Train(*dataset, train_conf);
    if (!res.has_value()) {
        ThrowInfo(ErrorCode::UnexpectedError,
                  fmt::format("failed to kmeans train: {}: {}",
                              KnowhereStatusString(res.error()),
                              res.what()));
    }
    res.value()->SetIsOwner(true);
    rc.RecordSection("clustering train done");
    dataset.reset();  // release train data

    auto centroid_id_mapping =
        reinterpret_cast<const uint32_t*>(res.value()->GetTensor());

    auto centroids_res = cluster_node.GetCentroids();
    if (!centroids_res.has_value()) {
        ThrowInfo(ErrorCode::UnexpectedError,
                  fmt::format("failed to get centroids: {}: {}",
                              KnowhereStatusString(res.error()),
                              res.what()));
    }
    // centroids owned by cluster_node
    centroids_res.value()->SetIsOwner(false);
    auto centroids =
        reinterpret_cast<const T*>(centroids_res.value()->GetTensor());

    auto centroid_stats = CentroidsToPB<T>(centroids, num_clusters, dim);
    auto id_mapping_stats = CentroidIdMappingToPB(centroid_id_mapping,
                                                  segment_ids,
                                                  trained_segments_num,
                                                  num_rows,
                                                  num_clusters);
    // upload
    StreamingAssignandUpload<T>(cluster_node,
                                config,
                                centroid_stats,
                                id_mapping_stats,
                                segment_ids,
                                insert_files,
                                num_rows,
                                dim,
                                trained_segments_num,
                                num_clusters);
    rc.RecordSection("clustering result upload done");
    rc.ElapseFromBegin("clustering done");
}

template <typename T>
void
KmeansClustering::RunV2(const milvus::proto::clustering::AnalyzeInfo& config) {
    std::map<int64_t, std::vector<std::string>> insert_files;
    for (const auto& pair : config.insert_files()) {
        std::vector<std::string> segment_files(
            pair.second.insert_files().begin(),
            pair.second.insert_files().end());
        insert_files[pair.first] = std::move(segment_files);
    }

    std::map<int64_t, int64_t> num_rows(config.num_rows().begin(),
                                        config.num_rows().end());
    auto num_clusters = config.num_clusters();
    AssertInfo(num_clusters > 0, "num clusters must larger than 0");
    auto train_size = config.train_size();
    AssertInfo(train_size > 0, "train size must larger than 0");
    auto dim = config.dim();
    AssertInfo(dim > 0, "dim must larger than 0");
    auto min_cluster_ratio = config.min_cluster_ratio();
    AssertInfo(min_cluster_ratio > 0 && min_cluster_ratio < 1,
               "min cluster ratio must larger than 0, less than 1");
    auto max_cluster_ratio = config.max_cluster_ratio();
    AssertInfo(max_cluster_ratio > 1, "max cluster ratio must larger than 1");
    auto max_cluster_size = config.max_cluster_size();
    AssertInfo(max_cluster_size > 0, "max cluster size must larger than 0");
    AssertInfo(!config.head_index_path().empty(),
               "head index path must not be empty for AnalyzeV2");
    AssertInfo(!config.compaction_plan_path().empty(),
               "compaction plan path must not be empty for AnalyzeV2");
    LOG_INFO(msg_header_ +
                 "AnalyzeV2 parsed storage infos: {}, insert files: {}, num "
                 "rows: {}",
             config.segment_storage_infos_size(),
             config.insert_files_size(),
             config.num_rows_size());
    int logged_storage_info = 0;
    for (const auto& [segment_id, storage_info] :
         config.segment_storage_infos()) {
        if (logged_storage_info++ >= 5) {
            break;
        }
        LOG_INFO(
            msg_header_ +
                "AnalyzeV2 storage info key: {}, storage_version: {}, "
                "has_insert_files: {}, insert_file_groups: {}, manifest: {}",
            segment_id,
            storage_info.storage_version(),
            storage_info.has_segment_insert_files(),
            storage_info.has_segment_insert_files()
                ? storage_info.segment_insert_files().field_insert_files_size()
                : 0,
            storage_info.manifest());
    }

    auto cluster_node_obj =
        knowhere::ClusterFactory::Instance().Create<T>(CARDINAL_KMEANS_CLUSTER);
    knowhere::Cluster<knowhere::ClusterNode> cluster_node;
    if (cluster_node_obj.has_value()) {
        cluster_node = std::move(cluster_node_obj.value());
    } else {
        auto err = cluster_node_obj.error();
        if (err == knowhere::Status::invalid_cluster_error) {
            throw SegcoreError(ErrorCode::ClusterSkip, cluster_node_obj.what());
        }
        throw SegcoreError(ErrorCode::KnowhereError, cluster_node_obj.what());
    }

    size_t data_num = 0;
    std::vector<int64_t> segment_ids;
    for (auto& [segment_id, num_row_each_segment] : num_rows) {
        data_num += num_row_each_segment;
        segment_ids.emplace_back(segment_id);
        AssertInfo(insert_files.find(segment_id) != insert_files.end(),
                   "segment id {} not exist in insert files",
                   segment_id);
    }
    AssertInfo(data_num > 0, "AnalyzeV2 input rows must larger than 0");

    size_t data_size = data_num * dim * sizeof(T);
    size_t train_num = train_size / sizeof(T) / dim;
    bool random_sample = true;
    if (train_num >= data_num) {
        train_num = data_num;
        random_sample = false;
    }
    if (train_num < num_clusters) {
        LOG_WARN(msg_header_ +
                     "AnalyzeV2 train num: {} less than num_clusters: {}, "
                     "skip clustering",
                 train_num,
                 num_clusters);
        throw SegcoreError(ErrorCode::ClusterSkip,
                           "sample data num less than num clusters");
    }

    size_t train_size_final = train_num * dim * sizeof(T);
    std::map<int64_t, milvus::proto::clustering::SegmentStorageInfo>
        segment_storage_infos(config.segment_storage_infos().begin(),
                              config.segment_storage_infos().end());
    knowhere::TimeRecorder rc(msg_header_ + "AnalyzeV2 cardinal kmeans",
                              2 /* log level: info */);
    LOG_INFO(
        msg_header_ + "AnalyzeV2 pull and sample {}GB data out of {}GB data",
        train_size_final / 1024.0 / 1024.0 / 1024.0,
        data_size / 1024.0 / 1024.0 / 1024.0);
    auto buf = std::make_unique<uint8_t[]>(train_size_final);
    SampleTrainData<T>(segment_ids,
                       insert_files,
                       segment_storage_infos,
                       num_rows,
                       train_size_final,
                       dim,
                       random_sample,
                       buf.get());
    rc.RecordSection("sample done");

    auto dataset = GenDataset(train_num, dim, buf.release());
    dataset->SetIsOwner(true);

    knowhere::Json cluster_conf;
    cluster_conf[NUM_CLUSTERS] = num_clusters;
    // Forward knowhere.cluster.* pass-through params verbatim; cardinal reads what it needs
    // and applies its own defaults for absent keys.
    for (const auto& kv : config.cluster_params()) {
        cluster_conf[kv.first] = kv.second;
    }
    cluster_conf["compaction_max_rows"] =
        config.compaction_max_rows() < 0 ? 0 : config.compaction_max_rows();
    cluster_conf["compaction_min_rows"] =
        config.compaction_min_rows() < 0 ? 0 : config.compaction_min_rows();

    LOG_INFO(
        msg_header_ + "AnalyzeV2 train data num: {}, dim: {}, num_clusters: {}",
        train_num,
        dim,
        num_clusters);
    auto train_res = cluster_node.Train(*dataset, cluster_conf);
    if (!train_res.has_value()) {
        ThrowInfo(ErrorCode::UnexpectedError,
                  fmt::format("failed to AnalyzeV2 train: {}: {}",
                              KnowhereStatusString(train_res.error()),
                              train_res.what()));
    }
    train_res.value()->SetIsOwner(true);
    dataset.reset();
    rc.RecordSection("clustering train done");

    auto centroids_res = cluster_node.GetCentroids();
    if (!centroids_res.has_value()) {
        ThrowInfo(ErrorCode::UnexpectedError,
                  fmt::format("failed to get AnalyzeV2 centroids: {}: {}",
                              KnowhereStatusString(centroids_res.error()),
                              centroids_res.what()));
    }
    centroids_res.value()->SetIsOwner(false);
    auto centroids =
        reinterpret_cast<const T*>(centroids_res.value()->GetTensor());
    auto centroid_stats = CentroidsToPB<T>(centroids, num_clusters, dim);

    std::unordered_map<std::string, int64_t> remote_paths_to_size;
    auto centroid_byte_size = centroid_stats.ByteSizeLong();
    auto centroid_data = std::make_unique<uint8_t[]>(centroid_byte_size);
    centroid_stats.SerializeToArray(centroid_data.get(), centroid_byte_size);
    auto centroid_remote_path =
        GetRemoteCentroidsObjectPrefix() + "/" + std::string(CENTROIDS_NAME);
    AddClusteringResultFiles(file_manager_->GetChunkManager().get(),
                             centroid_data.get(),
                             centroid_byte_size,
                             centroid_remote_path,
                             remote_paths_to_size);
    cluster_result_.centroid_path = centroid_remote_path;
    cluster_result_.centroid_file_size =
        remote_paths_to_size.at(centroid_remote_path);
    remote_paths_to_size.clear();

    BuildAndUploadHeadIndex<T>(
        centroids, num_clusters, dim, config.head_index_path(), fs_, config.cluster_params());
    rc.RecordSection("head index upload done");

    auto all_ids = std::make_unique<long long int[]>(data_num);
    size_t all_offset = 0;
    std::unordered_map<std::string, int64_t> id_mapping_paths_to_size;

    auto upload_mapping =
        [&](int64_t segment_id,
            const milvus::proto::clustering::ClusteringCentroidIdMappingStats&
                id_mapping_pb) {
            auto byte_size = id_mapping_pb.ByteSizeLong();
            auto data = std::make_unique<uint8_t[]>(byte_size);
            id_mapping_pb.SerializeToArray(data.get(), byte_size);
            auto offset_mapping_path =
                GetRemoteCentroidIdMappingObjectPrefix(segment_id) + "/" +
                std::string(OFFSET_MAPPING_NAME);
            AddClusteringResultFiles(file_manager_->GetChunkManager().get(),
                                     data.get(),
                                     byte_size,
                                     offset_mapping_path,
                                     id_mapping_paths_to_size);
            auto offset_distance_mapping_path =
                GetRemoteCentroidIdMappingObjectPrefix(segment_id) + "/" +
                std::string(OFFSET_DISTANCE_MAPPING_NAME);
            AddClusteringResultFiles(file_manager_->GetChunkManager().get(),
                                     data.get(),
                                     byte_size,
                                     offset_distance_mapping_path,
                                     remote_paths_to_size);
        };

    for (auto segment_id : segment_ids) {
        int64_t num_row = num_rows.at(segment_id);
        auto segment_buf = std::make_unique<T[]>(num_row * dim);
        int64_t offset = 0;
        const milvus::proto::clustering::SegmentStorageInfo* storage_info =
            nullptr;
        auto storage_it = segment_storage_infos.find(segment_id);
        if (storage_it != segment_storage_infos.end()) {
            storage_info = &storage_it->second;
        }
        auto first_storage_info_key =
            segment_storage_infos.empty()
                ? -1
                : segment_storage_infos.begin()->first;
        FetchDataFiles<T>(reinterpret_cast<uint8_t*>(segment_buf.get()),
                          INT64_MAX,
                          num_row * dim * sizeof(T),
                          insert_files.at(segment_id),
                          storage_info,
                          segment_id,
                          segment_storage_infos.size(),
                          first_storage_info_key,
                          dim,
                          offset);
        auto segment_dataset = GenDataset(num_row, dim, segment_buf.release());
        segment_dataset->SetIsOwner(true);
        auto assign_res = cluster_node.Assign(*segment_dataset, cluster_conf);
        if (!assign_res.has_value()) {
            ThrowInfo(ErrorCode::UnexpectedError,
                      fmt::format("failed to AnalyzeV2 assign: {}: {}",
                                  KnowhereStatusString(assign_res.error()),
                                  assign_res.what()));
        }
        assign_res.value()->SetIsOwner(true);
        const auto* id_mapping = assign_res.value()->GetIds();
        const auto* distances = assign_res.value()->GetDistance();
        auto id_mapping_pb = CentroidIdMappingWithDistanceToPB(
            id_mapping, distances, num_row, num_clusters);
        for (int64_t row = 0; row < num_row; ++row) {
            all_ids[all_offset++] = id_mapping[row];
        }
        upload_mapping(segment_id, id_mapping_pb);
    }

    auto assignment_dataset = std::make_shared<knowhere::DataSet>();
    assignment_dataset->SetRows(data_num);
    assignment_dataset->SetDim(1);
    assignment_dataset->SetIds(std::move(all_ids));
    assignment_dataset->SetIsOwner(true);
    auto compaction_plan_res =
        cluster_node.BuildCompactionPlan(*assignment_dataset, cluster_conf);
    if (!compaction_plan_res.has_value()) {
        ThrowInfo(
            ErrorCode::UnexpectedError,
            fmt::format("failed to build AnalyzeV2 compaction plan: {}: {}",
                        KnowhereStatusString(compaction_plan_res.error()),
                        compaction_plan_res.what()));
    }
    const auto compaction_plan = compaction_plan_res.value()->GetJsonInfo();
    AddClusteringResultFiles(
        file_manager_->GetChunkManager().get(),
        reinterpret_cast<const uint8_t*>(compaction_plan.data()),
        compaction_plan.size(),
        config.compaction_plan_path(),
        remote_paths_to_size);

    if (!config.chunk_mapping_path().empty()) {
        const std::string empty_chunk_mapping = "{}";
        AddClusteringResultFiles(
            file_manager_->GetChunkManager().get(),
            reinterpret_cast<const uint8_t*>(empty_chunk_mapping.data()),
            empty_chunk_mapping.size(),
            config.chunk_mapping_path(),
            remote_paths_to_size);
    }

    LOG_INFO(msg_header_ + "AnalyzeV2 upload cluster id mapping file done");
    cluster_result_.id_mappings = std::move(id_mapping_paths_to_size);
    is_runned_ = true;
    rc.RecordSection("AnalyzeV2 result upload done");
    rc.ElapseFromBegin("AnalyzeV2 clustering done");
}

template void
KmeansClustering::StreamingAssignandUpload<float>(
    knowhere::Cluster<knowhere::ClusterNode>& cluster_node,
    const milvus::proto::clustering::AnalyzeInfo& config,
    const milvus::proto::clustering::ClusteringCentroidsStats& centroid_stats,
    const std::vector<
        milvus::proto::clustering::ClusteringCentroidIdMappingStats>&
        id_mapping_stats,
    const std::vector<int64_t>& segment_ids,
    const std::map<int64_t, std::vector<std::string>>& insert_files,
    const std::map<int64_t, int64_t>& num_rows,
    const int64_t dim,
    const int64_t trained_segments_num,
    const int64_t num_clusters);

template void
KmeansClustering::FetchDataFiles<float>(
    uint8_t* buf,
    const int64_t expected_train_size,
    const int64_t expected_remote_file_size,
    const std::vector<std::string>& files,
    const milvus::proto::clustering::SegmentStorageInfo* storage_info,
    const int64_t segment_id,
    const size_t storage_info_map_size,
    const int64_t first_storage_info_key,
    const int64_t dim,
    int64_t& offset);
template void
KmeansClustering::SampleTrainData<float>(
    const std::vector<int64_t>& segment_ids,
    const std::map<int64_t, std::vector<std::string>>& segment_file_paths,
    const std::map<int64_t, milvus::proto::clustering::SegmentStorageInfo>&
        segment_storage_infos,
    const std::map<int64_t, int64_t>& segment_num_rows,
    const int64_t expected_train_size,
    const int64_t dim,
    const bool random_sample,
    uint8_t* buf);

template void
KmeansClustering::Run<float>(
    const milvus::proto::clustering::AnalyzeInfo& config);

template void
KmeansClustering::RunV2<float>(
    const milvus::proto::clustering::AnalyzeInfo& config);

template milvus::proto::clustering::ClusteringCentroidsStats
KmeansClustering::CentroidsToPB<float>(const float* centroids,
                                       const int64_t num_clusters,
                                       const int64_t dim);

}  // namespace milvus::clustering
