#include "segcore/storagev1translator/DiskVecIndexTranslator.h"

#include "cachinglayer/Utils.h"
#include "common/Common.h"
#include "common/Consts.h"
#include "common/Slice.h"
#include "common/File.h"
#include "index/Meta.h"
#include "index/Utils.h"
#include "knowhere/config.h"
#include "log/Log.h"
#include "storage/DiskFileManagerImpl.h"
#include "storage/Util.h"
#include "knowhere/index/index.h"
#include "knowhere/index/index_factory.h"
namespace milvus::segcore::storagev1translator {

template <typename T>
DiskVecIndexTranslator<T>::DiskVecIndexTranslator(
    int64_t segment_id,
    int64_t field_id,
    knowhere::IndexType index_type,
    knowhere::IndexVersion index_version,
    Config config,
    std::shared_ptr<storage::DiskFileManagerImpl> file_manager)
    : segment_id_(segment_id),
      field_id_(field_id),
      key_(fmt::format("seg_{}_f_{}_index", segment_id, field_id)),
      index_type_(std::move(index_type)),
      index_version_(std::move(index_version)),
      config_(std::move(config)),
      file_manager_(std::move(file_manager)) {
}

template <typename T>
size_t
DiskVecIndexTranslator<T>::estimated_byte_size_of_cell(
    milvus::cachinglayer::cid_t cid) const {
    return 0;
}

template <typename T>
size_t
DiskVecIndexTranslator<T>::num_cells() const {
    return 1;
}

template <typename T>
milvus::cachinglayer::cid_t
DiskVecIndexTranslator<T>::cell_id_of(milvus::cachinglayer::uid_t uid) const {
    return 0;
}

template <typename T>
milvus::cachinglayer::StorageType
DiskVecIndexTranslator<T>::storage_type() const {
    return milvus::cachinglayer::StorageType::FILE;
}

template <typename T>
const std::string&
DiskVecIndexTranslator<T>::key() const {
    return key_;
}

template <typename T>
std::vector<std::pair<milvus::cachinglayer::cid_t,
                      std::unique_ptr<knowhere::Index<knowhere::IndexNode>>>>
DiskVecIndexTranslator<T>::get_cells(
    const std::vector<milvus::cachinglayer::cid_t>& cids) const {
    AssertInfo(cids.size() == 1 && cids[0] == 0,
               "DiskVecIndexTranslator only supports single cell");

    auto index = load_index_into_file();

    std::vector<
        std::pair<milvus::cachinglayer::cid_t,
                  std::unique_ptr<knowhere::Index<knowhere::IndexNode>>>>
        cells;
    cells.emplace_back(0, std::move(index));
    return cells;
}

template <typename T>
knowhere::Json
DiskVecIndexTranslator<T>::update_load_json() const {
    knowhere::Json load_config;
    load_config.update(config_);

    // set data path
    auto local_index_path_prefix = file_manager_->GetLocalIndexObjectPrefix();
    load_config[index::DISK_ANN_PREFIX_PATH] = local_index_path_prefix;

    if (index_type_ == knowhere::IndexEnum::INDEX_DISKANN) {
        // set base info
        load_config[index::DISK_ANN_PREPARE_WARM_UP] = false;
        load_config[index::DISK_ANN_PREPARE_USE_BFS_CACHE] = false;

        // set threads number
        auto num_threads = index::GetValueFromConfig<std::string>(
            load_config, index::DISK_ANN_LOAD_THREAD_NUM);
        AssertInfo(num_threads.has_value(),
                   "param " + std::string(index::DISK_ANN_LOAD_THREAD_NUM) +
                       "is empty");
        load_config[index::DISK_ANN_THREADS_NUM] =
            std::atoi(num_threads.value().c_str());

        // update search_beamwidth
        auto beamwidth = index::GetValueFromConfig<std::string>(
            load_config, index::DISK_ANN_QUERY_BEAMWIDTH);
    }

    if (config_.contains(index::MMAP_FILE_PATH)) {
        load_config.erase(index::MMAP_FILE_PATH);
        load_config[index::ENABLE_MMAP] = true;
    }

    return load_config;
}

template <typename T>
std::unique_ptr<knowhere::Index<knowhere::IndexNode>>
DiskVecIndexTranslator<T>::load_index_into_file() const {
    knowhere::Json load_config = update_load_json();

    // start read file span with active scope
    {
        auto index_files = index::GetValueFromConfig<std::vector<std::string>>(
            load_config, "index_files");
        AssertInfo(index_files.has_value(),
                   "index file paths is empty when load disk ann index data");
        file_manager_->CacheIndexToDisk(index_files.value());
    }

    LOG_INFO("load index into Knowhere...");

    auto index_obj = knowhere::IndexFactory::Instance().Create<T>(
        index_type_,
        index_version_,
        knowhere::Pack(std::shared_ptr<knowhere::FileManager>(file_manager_)));
    std::unique_ptr<knowhere::Index<knowhere::IndexNode>> index_ptr;

    if (index_obj.has_value()) {
        index_ptr = std::make_unique<knowhere::Index<knowhere::IndexNode>>(
            index_obj.value());
    } else {
        auto err = index_obj.error();
        if (err == knowhere::Status::invalid_index_error) {
            PanicInfo(ErrorCode::Unsupported, index_obj.what());
        }
        PanicInfo(ErrorCode::KnowhereError, index_obj.what());
    }

    auto stat = index_ptr->Deserialize(knowhere::BinarySet(), load_config);
    if (stat != knowhere::Status::success)
        PanicInfo(ErrorCode::UnexpectedError,
                  "failed to Deserialize index, " + KnowhereStatusString(stat));
    LOG_INFO("load vector index done");

    return index_ptr;
}

template class DiskVecIndexTranslator<float>;
template class DiskVecIndexTranslator<bin1>;
template class DiskVecIndexTranslator<float16>;
template class DiskVecIndexTranslator<bfloat16>;
template class DiskVecIndexTranslator<int8>;

}  // namespace milvus::segcore::storagev1translator
