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

#include <glog/logging.h>
#include <string.h>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/Consts.h"
#include "common/EasyAssert.h"
#include "common/FieldMeta.h"
#include "common/Types.h"
#include "common/Utils.h"
#include "common/protobuf_utils.h"
#include "common/type_c.h"
#include "filemanager/InputStream.h"
#include "index/IndexStats.h"
#include "index/Meta.h"
#include "index/TextMatchIndex.h"
#include "index/Utils.h"
#include "index/json_stats/JsonKeyStats.h"
#include "indexbuilder/IndexCreatorBase.h"
#include "indexbuilder/IndexFactory.h"
#include "indexbuilder/VecIndexCreator.h"
#include "indexbuilder/index_c.h"
#include "indexbuilder/type_c.h"
#include "knowhere/binaryset.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/operands.h"
#include "knowhere/version.h"
#include "log/Log.h"
#include "monitor/scope_metric.h"
#include "nlohmann/json.hpp"
#include "pb/common.pb.h"
#include "pb/index_cgo_msg.pb.h"
#include "pb/schema.pb.h"
#include "storage/FileManager.h"
#include "storage/PluginLoader.h"
#include "storage/RemoteInputStream.h"
#include "storage/StorageV2FSCache.h"
#include "storage/Types.h"
#include "storage/Util.h"
#include "storage/loon_ffi/util.h"
#include "storage/plugin/PluginInterface.h"

using namespace milvus;

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
    AddFileMeta(const milvus::FileMeta&) override {
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
        ThrowInfo(ErrorCode::UnexpectedError,
                  "global head index file manager is read-only");
    }

 private:
    milvus_storage::ArrowFileSystemPtr fs_;
    std::string head_index_path_;
};

struct HeadIndexHandle {
    knowhere::Index<knowhere::IndexNode> index;
    std::shared_ptr<GlobalHeadIndexFileManager> file_manager;
};

knowhere::Json
BuildHeadIndexKnowhereConfig(int64_t dim, int64_t topk, int64_t ef = 0) {
    knowhere::Json config;
    config[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
    config[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    config[knowhere::meta::DIM] = dim;
    config[knowhere::meta::TOPK] = topk;
    config[knowhere::meta::INDEX_PREFIX] = "global_head_index";
    config[knowhere::meta::INDEX_ENGINE_VERSION] = 10;
    if (ef > 0) {
        config["ef"] = ef;
    }
    return config;
}

}  // namespace

CStatus
CreateIndexForUT(enum CDataType dtype,
                 const char* serialized_type_params,
                 const char* serialized_index_params,
                 CIndex* res_index) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(res_index, "failed to create index, passed index was null");

        milvus::proto::indexcgo::TypeParams type_params;
        milvus::proto::indexcgo::IndexParams index_params;
        milvus::index::ParseFromString(type_params, serialized_type_params);
        milvus::index::ParseFromString(index_params, serialized_index_params);

        milvus::Config config;
        for (auto i = 0; i < type_params.params_size(); ++i) {
            const auto& param = type_params.params(i);
            config[param.key()] = param.value();
        }

        for (auto i = 0; i < index_params.params_size(); ++i) {
            const auto& param = index_params.params(i);
            config[param.key()] = param.value();
        }

        config[milvus::index::INDEX_ENGINE_VERSION] = std::to_string(
            knowhere::Version::GetCurrentVersion().VersionNumber());

        auto& index_factory = milvus::indexbuilder::IndexFactory::GetInstance();
        auto index =
            index_factory.CreateIndex(milvus::DataType(dtype),
                                      config,
                                      milvus::storage::FileManagerContext());

        *res_index = index.release();
        status.error_code = Success;
        status.error_msg = "";
    } catch (SegcoreError& e) {
        auto status = CStatus();
        status.error_code = e.get_error_code();
        status.error_msg = strdup(e.what());
        return status;
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

milvus::storage::StorageConfig
get_storage_config(const milvus::proto::indexcgo::StorageConfig& config) {
    auto storage_config = milvus::storage::StorageConfig();
    storage_config.address = std::string(config.address());
    storage_config.bucket_name = std::string(config.bucket_name());
    storage_config.access_key_id = std::string(config.access_keyid());
    storage_config.access_key_value = std::string(config.secret_access_key());
    storage_config.root_path = std::string(config.root_path());
    storage_config.storage_type = std::string(config.storage_type());
    storage_config.cloud_provider = std::string(config.cloud_provider());
    storage_config.iam_endpoint = std::string(config.iamendpoint());
    storage_config.useSSL = config.usessl();
    storage_config.sslCACert = config.sslcacert();
    storage_config.useIAM = config.useiam();
    storage_config.region = config.region();
    storage_config.useVirtualHost = config.use_virtual_host();
    storage_config.requestTimeoutMs = config.request_timeout_ms();
    storage_config.gcp_credential_json =
        std::string(config.gcpcredentialjson());
    storage_config.max_connections = config.max_connections();
    storage_config.tls_min_version = std::string(config.ssl_tls_min_version());
    storage_config.use_crc32c_checksum = config.use_crc32c_checksum();
    return storage_config;
}

milvus::OptFieldT
get_opt_field(const ::google::protobuf::RepeatedPtrField<
              milvus::proto::indexcgo::OptionalFieldInfo>& field_infos) {
    milvus::OptFieldT opt_fields_map;
    for (const auto& field_info : field_infos) {
        auto field_id = field_info.fieldid();
        auto it = opt_fields_map.find(field_id);
        if (it == opt_fields_map.end()) {
            it = opt_fields_map
                     .emplace(field_id,
                              std::make_tuple(field_info.field_name(),
                                              static_cast<milvus::DataType>(
                                                  field_info.field_type()),
                                              static_cast<milvus::DataType>(
                                                  field_info.element_type()),
                                              std::vector<std::string>{}))
                     .first;
        }
        for (const auto& str : field_info.data_paths()) {
            std::get<3>(it->second).emplace_back(str);
        }
    }

    return opt_fields_map;
}

milvus::SegmentInsertFiles
get_segment_insert_files(
    const milvus::proto::indexcgo::SegmentInsertFiles& segment_insert_files) {
    milvus::SegmentInsertFiles files;
    for (const auto& column_group_files :
         segment_insert_files.field_insert_files()) {
        std::vector<std::string> paths;
        paths.reserve(column_group_files.file_paths().size());
        for (const auto& path : column_group_files.file_paths()) {
            paths.push_back(path);
        }
        files.emplace_back(std::move(paths));
    }
    return files;
}

milvus::Config
get_config(std::unique_ptr<milvus::proto::indexcgo::BuildIndexInfo>& info) {
    milvus::Config config;
    for (auto i = 0; i < info->index_params().size(); ++i) {
        const auto& param = info->index_params(i);
        config[param.key()] = param.value();
    }

    for (auto i = 0; i < info->type_params().size(); ++i) {
        const auto& param = info->type_params(i);
        config[param.key()] = param.value();
    }

    config[INSERT_FILES_KEY] = info->insert_files();
    if (info->opt_fields().size()) {
        config[VEC_OPT_FIELDS] = get_opt_field(info->opt_fields());
    }
    if (info->partition_key_isolation()) {
        config[PARTITION_KEY_ISOLATION_KEY] = info->partition_key_isolation();
    }
    config[INDEX_NUM_ROWS_KEY] = info->num_rows();
    config[STORAGE_VERSION_KEY] = info->storage_version();
    if (info->storage_version() == STORAGE_V2 ||
        info->storage_version() == STORAGE_V3) {
        config[SEGMENT_INSERT_FILES_KEY] =
            get_segment_insert_files(info->segment_insert_files());
        config[SEGMENT_MANIFEST_KEY] = info->manifest();
    }
    config[DIM_KEY] = info->dim();
    config[DATA_TYPE_KEY] = info->field_schema().data_type();
    config[ELEMENT_TYPE_KEY] = info->field_schema().element_type();
    if (!info->stats_base_path().empty()) {
        config[STATS_BASE_PATH_KEY] = info->stats_base_path();
    }

    if (!info->analyzer_extra_info().empty()) {
        config["analyzer_extra_info"] = info->analyzer_extra_info();
    }

    return config;
}

CStatus
CreateIndex(CIndex* res_index,
            const uint8_t* serialized_build_index_info,
            const uint64_t len) {
    SCOPE_CGO_CALL_METRIC();

    try {
        auto build_index_info =
            std::make_unique<milvus::proto::indexcgo::BuildIndexInfo>();
        auto res =
            build_index_info->ParseFromArray(serialized_build_index_info, len);
        AssertInfo(res, "Unmarshal build index info failed");

        auto field_type =
            static_cast<DataType>(build_index_info->field_schema().data_type());

        auto storage_config =
            get_storage_config(build_index_info->storage_config());
        auto config = get_config(build_index_info);

        auto engine_version = build_index_info->current_index_version();
        config[milvus::index::INDEX_ENGINE_VERSION] =
            std::to_string(engine_version);
        auto scalar_index_engine_version =
            build_index_info->current_scalar_index_version();
        config[milvus::index::SCALAR_INDEX_ENGINE_VERSION] =
            scalar_index_engine_version;
        auto tantivy_index_version =
            scalar_index_engine_version <= 1
                ? milvus::index::TANTIVY_INDEX_MINIMUM_VERSION
                : milvus::index::TANTIVY_INDEX_LATEST_VERSION;
        config[milvus::index::TANTIVY_INDEX_VERSION] = tantivy_index_version;

        // check index encoding config
        auto index_non_encoding_str =
            config.value(milvus::index::INDEX_NON_ENCODING, "false");
        bool index_non_encoding = index_non_encoding_str == "true";

        // init file manager
        milvus::storage::FieldDataMeta field_meta{
            build_index_info->collectionid(),
            build_index_info->partitionid(),
            build_index_info->segmentid(),
            build_index_info->field_schema().fieldid(),
            build_index_info->field_schema()};

        milvus::storage::IndexMeta index_meta{
            build_index_info->segmentid(),
            build_index_info->field_schema().fieldid(),
            build_index_info->buildid(),
            build_index_info->index_version(),
            "",
            build_index_info->field_schema().name(),
            field_type,
            build_index_info->dim(),
            index_non_encoding,
            build_index_info->index_store_path_version()};
        auto chunk_manager =
            milvus::storage::CreateChunkManager(storage_config);
        LOG_INFO("create chunk manager success, build_id: {}",
                 build_index_info->buildid());
        auto fs = milvus::storage::InitArrowFileSystem(storage_config);
        LOG_INFO("init arrow file system success, build_id: {}",
                 build_index_info->buildid());

        milvus::storage::FileManagerContext fileManagerContext(
            field_meta, index_meta, chunk_manager, fs);
        if (!build_index_info->stats_base_path().empty()) {
            fileManagerContext.set_stats_base_path(
                build_index_info->stats_base_path());
        }
        if (build_index_info->manifest() != "") {
            auto loon_properties = MakeInternalPropertiesFromStorageConfig(
                ToCStorageConfig(storage_config));
            // For external collections, inject extfs.{collID}.* from build_index_info
            if (!build_index_info->external_source().empty()) {
                InjectExternalSpecProperties(
                    *loon_properties,
                    build_index_info->collectionid(),
                    build_index_info->external_source(),
                    build_index_info->external_spec());
            }
            fileManagerContext.set_loon_ffi_properties(loon_properties);
        }

        if (build_index_info->has_storage_plugin_context()) {
            auto cipherPlugin =
                milvus::storage::PluginLoader::GetInstance().getCipherPlugin();
            AssertInfo(cipherPlugin != nullptr, "failed to get cipher plugin");
            cipherPlugin->Update(
                build_index_info->storage_plugin_context().encryption_zone_id(),
                build_index_info->storage_plugin_context().collection_id(),
                build_index_info->storage_plugin_context().encryption_key());

            auto plugin_context = std::make_shared<CPluginContext>();
            plugin_context->ez_id =
                build_index_info->storage_plugin_context().encryption_zone_id();
            plugin_context->collection_id =
                build_index_info->storage_plugin_context().collection_id();
            fileManagerContext.set_plugin_context(plugin_context);
        }

        auto index =
            milvus::indexbuilder::IndexFactory::GetInstance().CreateIndex(
                field_type, config, fileManagerContext);
        LOG_INFO("create index instance success, build_id: {}",
                 build_index_info->buildid());
        index->Build();
        LOG_INFO("build index done, build_id: {}", build_index_info->buildid());
        *res_index = index.release();
        auto status = CStatus();
        status.error_code = Success;
        status.error_msg = "";
        return status;
    } catch (SegcoreError& e) {
        auto status = CStatus();
        status.error_code = e.get_error_code();
        status.error_msg = strdup(e.what());
        return status;
    } catch (std::exception& e) {
        auto status = CStatus();
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
        return status;
    }
}

CStatus
BuildJsonKeyIndex(ProtoLayoutInterface result,
                  const uint8_t* serialized_build_index_info,
                  const uint64_t len) {
    SCOPE_CGO_CALL_METRIC();

    try {
        auto build_index_info =
            std::make_unique<milvus::proto::indexcgo::BuildIndexInfo>();
        auto res =
            build_index_info->ParseFromArray(serialized_build_index_info, len);
        AssertInfo(res, "Unmarshall build index info failed");

        auto field_type = static_cast<milvus::DataType>(
            build_index_info->field_schema().data_type());

        auto storage_config =
            get_storage_config(build_index_info->storage_config());
        auto config = get_config(build_index_info);

        auto loon_properties =
            MakePropertiesFromStorageConfig(ToCStorageConfig(storage_config));

        // init file manager
        milvus::storage::FieldDataMeta field_meta{
            build_index_info->collectionid(),
            build_index_info->partitionid(),
            build_index_info->segmentid(),
            build_index_info->field_schema().fieldid(),
            build_index_info->field_schema()};

        milvus::storage::IndexMeta index_meta{
            build_index_info->segmentid(),
            build_index_info->field_schema().fieldid(),
            build_index_info->buildid(),
            build_index_info->index_version(),
            "",
            build_index_info->field_schema().name(),
            field_type,
            build_index_info->dim(),
        };

        auto scalar_index_engine_version =
            build_index_info->current_scalar_index_version();
        config[milvus::index::SCALAR_INDEX_ENGINE_VERSION] =
            scalar_index_engine_version;
        auto tantivy_index_version =
            scalar_index_engine_version <= 1
                ? milvus::index::TANTIVY_INDEX_MINIMUM_VERSION
                : milvus::index::TANTIVY_INDEX_LATEST_VERSION;
        config[milvus::index::TANTIVY_INDEX_VERSION] = tantivy_index_version;

        auto chunk_manager =
            milvus::storage::CreateChunkManager(storage_config);
        auto fs = milvus::storage::InitArrowFileSystem(storage_config);

        milvus::storage::FileManagerContext fileManagerContext(
            field_meta, index_meta, chunk_manager, fs);
        fileManagerContext.set_stats_base_path(
            build_index_info->stats_base_path());

        if (build_index_info->manifest() != "") {
            auto loon_properties = MakeInternalPropertiesFromStorageConfig(
                ToCStorageConfig(storage_config));
            // For external collections, inject extfs.{collID}.* from build_index_info
            if (!build_index_info->external_source().empty()) {
                InjectExternalSpecProperties(
                    *loon_properties,
                    build_index_info->collectionid(),
                    build_index_info->external_source(),
                    build_index_info->external_spec());
            }
            fileManagerContext.set_loon_ffi_properties(loon_properties);
        }

        if (build_index_info->has_storage_plugin_context()) {
            auto cipherPlugin =
                milvus::storage::PluginLoader::GetInstance().getCipherPlugin();
            AssertInfo(cipherPlugin != nullptr, "failed to get cipher plugin");
            cipherPlugin->Update(
                build_index_info->storage_plugin_context().encryption_zone_id(),
                build_index_info->storage_plugin_context().collection_id(),
                build_index_info->storage_plugin_context().encryption_key());

            auto plugin_context = std::make_shared<CPluginContext>();
            plugin_context->ez_id =
                build_index_info->storage_plugin_context().encryption_zone_id();
            plugin_context->collection_id =
                build_index_info->storage_plugin_context().collection_id();
            fileManagerContext.set_plugin_context(plugin_context);
        }

        auto field_schema =
            FieldMeta::ParseFrom(build_index_info->field_schema());
        auto index = std::make_unique<index::JsonKeyStats>(
            fileManagerContext,
            false,
            build_index_info->json_stats_max_shredding_columns(),
            build_index_info->json_stats_shredding_ratio_threshold(),
            build_index_info->json_stats_write_batch_size(),
            tantivy_index_version);
        index->Build(config);
        auto create_index_result = index->Upload(config);
        create_index_result->SerializeAt(
            reinterpret_cast<milvus::ProtoLayout*>(result));
        auto status = CStatus();
        status.error_code = Success;
        status.error_msg = "";
        return status;
    } catch (SegcoreError& e) {
        auto status = CStatus();
        status.error_code = e.get_error_code();
        status.error_msg = strdup(e.what());
        return status;
    } catch (std::exception& e) {
        auto status = CStatus();
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
        return status;
    }
}

CStatus
BuildTextIndex(ProtoLayoutInterface result,
               const uint8_t* serialized_build_index_info,
               const uint64_t len) {
    SCOPE_CGO_CALL_METRIC();

    try {
        auto build_index_info =
            std::make_unique<milvus::proto::indexcgo::BuildIndexInfo>();
        auto res =
            build_index_info->ParseFromArray(serialized_build_index_info, len);
        AssertInfo(res, "Unmarshal build index info failed");

        auto field_type = static_cast<milvus::DataType>(
            build_index_info->field_schema().data_type());

        auto storage_config =
            get_storage_config(build_index_info->storage_config());
        auto config = get_config(build_index_info);

        // init file manager
        milvus::storage::FieldDataMeta field_meta{
            build_index_info->collectionid(),
            build_index_info->partitionid(),
            build_index_info->segmentid(),
            build_index_info->field_schema().fieldid(),
            build_index_info->field_schema()};

        milvus::storage::IndexMeta index_meta{
            build_index_info->segmentid(),
            build_index_info->field_schema().fieldid(),
            build_index_info->buildid(),
            build_index_info->index_version(),
            "",
            build_index_info->field_schema().name(),
            field_type,
            build_index_info->dim(),
        };
        auto chunk_manager =
            milvus::storage::CreateChunkManager(storage_config);
        auto fs = milvus::storage::InitArrowFileSystem(storage_config);

        milvus::storage::FileManagerContext fileManagerContext(
            field_meta, index_meta, chunk_manager, fs);
        fileManagerContext.set_stats_base_path(
            build_index_info->stats_base_path());

        if (build_index_info->manifest() != "") {
            auto loon_properties = MakeInternalPropertiesFromStorageConfig(
                ToCStorageConfig(storage_config));
            fileManagerContext.set_loon_ffi_properties(loon_properties);
        }

        if (build_index_info->has_storage_plugin_context()) {
            auto cipherPlugin =
                milvus::storage::PluginLoader::GetInstance().getCipherPlugin();
            AssertInfo(cipherPlugin != nullptr, "failed to get cipher plugin");
            cipherPlugin->Update(
                build_index_info->storage_plugin_context().encryption_zone_id(),
                build_index_info->storage_plugin_context().collection_id(),
                build_index_info->storage_plugin_context().encryption_key());
            auto plugin_context = std::make_shared<CPluginContext>();
            plugin_context->ez_id =
                build_index_info->storage_plugin_context().encryption_zone_id();
            plugin_context->collection_id =
                build_index_info->storage_plugin_context().collection_id();
            fileManagerContext.set_plugin_context(plugin_context);
        }

        auto scalar_index_engine_version =
            build_index_info->current_scalar_index_version();
        config[milvus::index::SCALAR_INDEX_ENGINE_VERSION] =
            scalar_index_engine_version;
        auto tantivy_index_version =
            scalar_index_engine_version <= 1
                ? milvus::index::TANTIVY_INDEX_MINIMUM_VERSION
                : milvus::index::TANTIVY_INDEX_LATEST_VERSION;
        config[milvus::index::TANTIVY_INDEX_VERSION] = tantivy_index_version;

        auto field_schema =
            FieldMeta::ParseFrom(build_index_info->field_schema());

        auto index = std::make_unique<index::TextMatchIndex>(
            fileManagerContext,
            tantivy_index_version,
            "milvus_tokenizer",
            field_schema.get_analyzer_params().c_str(),
            build_index_info->analyzer_extra_info().c_str());

        index->Build(config);
        auto create_index_result = index->Upload(config);
        create_index_result->SerializeAt(
            reinterpret_cast<milvus::ProtoLayout*>(result));
        auto status = CStatus();
        status.error_code = Success;
        status.error_msg = "";
        return status;
    } catch (SegcoreError& e) {
        auto status = CStatus();
        status.error_code = e.get_error_code();
        status.error_msg = strdup(e.what());
        return status;
    } catch (std::exception& e) {
        auto status = CStatus();
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
        return status;
    }
}

CStatus
DeleteIndex(CIndex index) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(index, "failed to delete index, passed index was null");
        auto cIndex =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        delete cIndex;
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
BuildFloatVecIndex(CIndex index,
                   int64_t float_value_num,
                   const float* vectors) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(index,
                   "failed to build float vector index, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto cIndex =
            dynamic_cast<milvus::indexbuilder::VecIndexCreator*>(real_index);
        auto dim = cIndex->dim();
        auto row_nums = float_value_num / dim;
        auto ds = knowhere::GenDataSet(row_nums, dim, vectors);
        cIndex->Build(ds);
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
BuildFloatVecIndexWithValidData(CIndex index,
                                int64_t float_value_num,
                                const float* vectors,
                                const bool* valid_data,
                                int64_t valid_data_len) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(index,
                   "failed to build float vector index, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto cIndex =
            dynamic_cast<milvus::indexbuilder::VecIndexCreator*>(real_index);
        auto dim = cIndex->dim();
        auto row_nums = float_value_num / dim;
        auto ds = knowhere::GenDataSet(row_nums, dim, vectors);
        cIndex->Build(ds, valid_data, valid_data_len);
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
BuildFloat16VecIndex(CIndex index,
                     int64_t float16_value_num,
                     const uint8_t* vectors) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(
            index,
            "failed to build float16 vector index, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto cIndex =
            dynamic_cast<milvus::indexbuilder::VecIndexCreator*>(real_index);
        auto dim = cIndex->dim();
        auto row_nums = float16_value_num / dim / 2;
        auto ds = knowhere::GenDataSet(row_nums, dim, vectors);
        cIndex->Build(ds);
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
BuildFloat16VecIndexWithValidData(CIndex index,
                                  int64_t float16_value_num,
                                  const uint8_t* vectors,
                                  const bool* valid_data,
                                  int64_t valid_data_len) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(
            index,
            "failed to build float16 vector index, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto cIndex =
            dynamic_cast<milvus::indexbuilder::VecIndexCreator*>(real_index);
        auto dim = cIndex->dim();
        auto row_nums = float16_value_num / dim / 2;
        auto ds = knowhere::GenDataSet(row_nums, dim, vectors);
        cIndex->Build(ds, valid_data, valid_data_len);
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
BuildBFloat16VecIndex(CIndex index,
                      int64_t bfloat16_value_num,
                      const uint8_t* vectors) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(
            index,
            "failed to build bfloat16 vector index, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto cIndex =
            dynamic_cast<milvus::indexbuilder::VecIndexCreator*>(real_index);
        auto dim = cIndex->dim();
        auto row_nums = bfloat16_value_num / dim / 2;
        auto ds = knowhere::GenDataSet(row_nums, dim, vectors);
        cIndex->Build(ds);
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
BuildBFloat16VecIndexWithValidData(CIndex index,
                                   int64_t bfloat16_value_num,
                                   const uint8_t* vectors,
                                   const bool* valid_data,
                                   int64_t valid_data_len) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(
            index,
            "failed to build bfloat16 vector index, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto cIndex =
            dynamic_cast<milvus::indexbuilder::VecIndexCreator*>(real_index);
        auto dim = cIndex->dim();
        auto row_nums = bfloat16_value_num / dim / 2;
        auto ds = knowhere::GenDataSet(row_nums, dim, vectors);
        cIndex->Build(ds, valid_data, valid_data_len);
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
BuildBinaryVecIndex(CIndex index, int64_t data_size, const uint8_t* vectors) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(
            index,
            "failed to build binary vector index, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto cIndex =
            dynamic_cast<milvus::indexbuilder::VecIndexCreator*>(real_index);
        auto dim = cIndex->dim();
        auto row_nums = (data_size * 8) / dim;
        auto ds = knowhere::GenDataSet(row_nums, dim, vectors);
        cIndex->Build(ds);
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
BuildBinaryVecIndexWithValidData(CIndex index,
                                 int64_t data_size,
                                 const uint8_t* vectors,
                                 const bool* valid_data,
                                 int64_t valid_data_len) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(
            index,
            "failed to build binary vector index, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto cIndex =
            dynamic_cast<milvus::indexbuilder::VecIndexCreator*>(real_index);
        auto dim = cIndex->dim();
        auto row_nums = (data_size * 8) / dim;
        auto ds = knowhere::GenDataSet(row_nums, dim, vectors);
        cIndex->Build(ds, valid_data, valid_data_len);
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
BuildSparseFloatVecIndex(CIndex index,
                         int64_t row_num,
                         int64_t dim,
                         const uint8_t* vectors) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(
            index,
            "failed to build sparse float vector index, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto cIndex =
            dynamic_cast<milvus::indexbuilder::VecIndexCreator*>(real_index);
        auto ds = knowhere::GenDataSet(row_num, dim, vectors);
        ds->SetIsSparse(true);
        cIndex->Build(ds);
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
BuildSparseFloatVecIndexWithValidData(CIndex index,
                                      int64_t row_num,
                                      int64_t dim,
                                      const uint8_t* vectors,
                                      const bool* valid_data,
                                      int64_t valid_data_len) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(
            index,
            "failed to build sparse float vector index, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto cIndex =
            dynamic_cast<milvus::indexbuilder::VecIndexCreator*>(real_index);
        auto ds = knowhere::GenDataSet(row_num, dim, vectors);
        ds->SetIsSparse(true);
        cIndex->Build(ds, valid_data, valid_data_len);
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
BuildInt8VecIndex(CIndex index, int64_t int8_value_num, const int8_t* vectors) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(index,
                   "failed to build int8 vector index, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto cIndex =
            dynamic_cast<milvus::indexbuilder::VecIndexCreator*>(real_index);
        auto dim = cIndex->dim();
        auto row_nums = int8_value_num / dim;
        auto ds = knowhere::GenDataSet(row_nums, dim, vectors);
        cIndex->Build(ds);
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
BuildInt8VecIndexWithValidData(CIndex index,
                               int64_t int8_value_num,
                               const int8_t* vectors,
                               const bool* valid_data,
                               int64_t valid_data_len) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(index,
                   "failed to build int8 vector index, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto cIndex =
            dynamic_cast<milvus::indexbuilder::VecIndexCreator*>(real_index);
        auto dim = cIndex->dim();
        auto row_nums = int8_value_num / dim;
        auto ds = knowhere::GenDataSet(row_nums, dim, vectors);
        cIndex->Build(ds, valid_data, valid_data_len);
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

// field_data:
//  1, serialized proto::schema::BoolArray, if type is bool;
//  2, serialized proto::schema::StringArray, if type is string;
//  3, raw pointer, if type is of fundamental except bool type;
// TODO: optimize here if necessary.
CStatus
BuildScalarIndex(CIndex c_index, int64_t size, const void* field_data) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(c_index,
                   "failed to build scalar index, passed index was null");

        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(c_index);
        const int64_t dim = 8;  // not important here
        auto dataset = knowhere::GenDataSet(size, dim, field_data);
        real_index->Build(dataset);

        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
SerializeIndexToBinarySet(CIndex index, CBinarySet* c_binary_set) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(
            index,
            "failed to serialize index to binary set, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto binary =
            std::make_unique<knowhere::BinarySet>(real_index->Serialize());
        *c_binary_set = binary.release();
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
LoadIndexFromBinarySet(CIndex index, CBinarySet c_binary_set) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(
            index,
            "failed to load index from binary set, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto binary_set = reinterpret_cast<knowhere::BinarySet*>(c_binary_set);
        real_index->Load(*binary_set);
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
CleanLocalData(CIndex index) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(index,
                   "failed to build float vector index, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto cIndex =
            dynamic_cast<milvus::indexbuilder::VecIndexCreator*>(real_index);
        cIndex->CleanLocalData();
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
SerializeIndexAndUpLoad(CIndex index, ProtoLayoutInterface result) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(
            index,
            "failed to serialize index to binary set, passed index was null");
        auto real_index =
            reinterpret_cast<milvus::indexbuilder::IndexCreatorBase*>(index);
        auto create_index_result = real_index->Upload();
        create_index_result->SerializeAt(
            reinterpret_cast<milvus::ProtoLayout*>(result));
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
LoadHeadIndex(CHeadIndex* res_index,
                      CStorageConfig c_storage_config,
                      const char* head_index_path) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(res_index,
                   "failed to load head index, passed index was null");
        AssertInfo(head_index_path != nullptr && strlen(head_index_path) > 0,
                   "failed to load head index, path is empty");

        auto storage_config = milvus::storage::StorageConfig();
        storage_config.address = c_storage_config.address == nullptr
                                     ? ""
                                     : std::string(c_storage_config.address);
        storage_config.bucket_name =
            c_storage_config.bucket_name == nullptr
                ? ""
                : std::string(c_storage_config.bucket_name);
        storage_config.access_key_id =
            c_storage_config.access_key_id == nullptr
                ? ""
                : std::string(c_storage_config.access_key_id);
        storage_config.access_key_value =
            c_storage_config.access_key_value == nullptr
                ? ""
                : std::string(c_storage_config.access_key_value);
        storage_config.root_path =
            c_storage_config.root_path == nullptr
                ? ""
                : std::string(c_storage_config.root_path);
        storage_config.storage_type =
            c_storage_config.storage_type == nullptr
                ? ""
                : std::string(c_storage_config.storage_type);
        storage_config.cloud_provider =
            c_storage_config.cloud_provider == nullptr
                ? ""
                : std::string(c_storage_config.cloud_provider);
        storage_config.iam_endpoint =
            c_storage_config.iam_endpoint == nullptr
                ? ""
                : std::string(c_storage_config.iam_endpoint);
        storage_config.region = c_storage_config.region == nullptr
                                    ? ""
                                    : std::string(c_storage_config.region);
        storage_config.useSSL = c_storage_config.useSSL;
        storage_config.sslCACert =
            c_storage_config.sslCACert == nullptr
                ? ""
                : std::string(c_storage_config.sslCACert);
        storage_config.useIAM = c_storage_config.useIAM;
        storage_config.useVirtualHost = c_storage_config.useVirtualHost;
        storage_config.requestTimeoutMs = c_storage_config.requestTimeoutMs;
        storage_config.gcp_credential_json =
            c_storage_config.gcp_credential_json == nullptr
                ? ""
                : std::string(c_storage_config.gcp_credential_json);
        storage_config.max_connections = c_storage_config.max_connections;
        storage_config.tls_min_version =
            c_storage_config.tls_min_version == nullptr
                ? ""
                : std::string(c_storage_config.tls_min_version);
        storage_config.use_crc32c_checksum =
            c_storage_config.use_crc32c_checksum;

        auto fs = milvus::storage::StorageV2FSCache::Instance().Get({
            storage_config.address,
            storage_config.bucket_name,
            storage_config.access_key_id,
            storage_config.access_key_value,
            storage_config.root_path,
            storage_config.storage_type,
            storage_config.cloud_provider,
            storage_config.iam_endpoint,
            storage_config.log_level,
            storage_config.region,
            storage_config.useSSL,
            storage_config.sslCACert,
            storage_config.useIAM,
            storage_config.useVirtualHost,
            storage_config.requestTimeoutMs,
            false,
            storage_config.gcp_credential_json,
            false,
            storage_config.max_connections,
            storage_config.tls_min_version,
            storage_config.use_crc32c_checksum,
        });
        auto file_manager = std::make_shared<GlobalHeadIndexFileManager>(
            fs, std::string(head_index_path));
        auto file_manager_pack =
            knowhere::Pack(std::shared_ptr<milvus::FileManager>(file_manager));
        auto index_or =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                knowhere::IndexEnum::INDEX_HNSW, 9, file_manager_pack);
        AssertInfo(index_or.has_value(),
                   "failed to create head index: {}",
                   index_or.what());

        auto index = std::move(index_or.value());
        knowhere::BinarySet empty_binary_set;
        auto deserialize_status = index.Deserialize(
            empty_binary_set, BuildHeadIndexKnowhereConfig(1, 1));
        AssertInfo(deserialize_status == knowhere::Status::success,
                   "failed to deserialize head index, status: {}",
                   KnowhereStatusString(deserialize_status));

        *res_index = new HeadIndexHandle{
            std::move(index),
            std::move(file_manager),
        };

        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
SearchHeadIndex(CHeadIndex index,
                        const float* query,
                        int64_t nq,
                        int64_t dim,
                        int64_t topk,
                        int64_t ef,
                        int64_t* ids,
                        float* distances) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        AssertInfo(index,
                   "failed to search head index, index is null");
        AssertInfo(query,
                   "failed to search head index, query is null");
        AssertInfo(ids, "failed to search head index, ids is null");
        AssertInfo(nq > 0, "failed to search head index, nq <= 0");
        AssertInfo(dim > 0, "failed to search head index, dim <= 0");
        AssertInfo(topk > 0, "failed to search head index, topk <= 0");

        auto* handle = reinterpret_cast<HeadIndexHandle*>(index);
        auto dataset = knowhere::GenDataSet(nq, dim, query);
        auto result =
            handle->index.Search(dataset,
                                 BuildHeadIndexKnowhereConfig(dim, topk, ef),
                                 knowhere::BitsetView(nullptr));
        AssertInfo(
            result.has_value(),
            "failed to search head index, status: {}, reason: {}",
            KnowhereStatusString(result.error()),
            result.what());
        const auto* result_ids = result.value()->GetIds();
        AssertInfo(result_ids != nullptr,
                   "failed to search head index, result ids is null");
        const auto* result_distances = result.value()->GetDistance();
        for (int64_t q = 0; q < nq; ++q) {
            for (int64_t k = 0; k < topk; ++k) {
                ids[q * topk + k] = result_ids[q * topk + k];
                if (distances != nullptr) {
                    distances[q * topk + k] =
                        result_distances != nullptr
                            ? result_distances[q * topk + k]
                            : 0.0f;
                }
            }
        }

        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}

CStatus
DeleteHeadIndex(CHeadIndex index) {
    SCOPE_CGO_CALL_METRIC();

    auto status = CStatus();
    try {
        if (index != nullptr) {
            delete reinterpret_cast<HeadIndexHandle*>(index);
        }
        status.error_code = Success;
        status.error_msg = "";
    } catch (std::exception& e) {
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
    }
    return status;
}
