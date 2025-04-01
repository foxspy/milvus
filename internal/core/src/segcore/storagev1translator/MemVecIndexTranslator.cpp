#include "segcore/storagev1translator/MemVecIndexTranslator.h"

#include "common/Common.h"
#include "common/Consts.h"
#include "common/Slice.h"
#include "common/File.h"
#include "log/Log.h"
#include "storage/Util.h"
#include "index/Meta.h"
#include "index/Utils.h"

#include "knowhere/index/index_factory.h"

namespace milvus::segcore::storagev1translator {

template <typename T>
MemVecIndexTranslator<T>::MemVecIndexTranslator(
    int64_t segment_id,
    int64_t field_id,
    knowhere::IndexType index_type,
    knowhere::IndexVersion index_version,
    Config config,
    milvus::cachinglayer::StorageType storage_type,
    std::shared_ptr<storage::MemFileManagerImpl> file_manager)
    : segment_id_(segment_id),
      field_id_(field_id),
      key_(fmt::format("seg_{}_f_{}_index", segment_id, field_id)),
      index_type_(std::move(index_type)),
      index_version_(index_version),
      config_(std::move(config)),
      storage_type_(storage_type),
      file_manager_(std::move(file_manager)) {
}

template <typename T>
size_t
MemVecIndexTranslator<T>::estimated_byte_size_of_cell(
    milvus::cachinglayer::cid_t cid) const {
    return 0;
}

template <typename T>
size_t
MemVecIndexTranslator<T>::num_cells() const {
    return 1;
}

template <typename T>
milvus::cachinglayer::cid_t
MemVecIndexTranslator<T>::cell_id_of(milvus::cachinglayer::uid_t uid) const {
    return 0;
}

template <typename T>
milvus::cachinglayer::StorageType
MemVecIndexTranslator<T>::storage_type() const {
    return storage_type_;
}

template <typename T>
const std::string&
MemVecIndexTranslator<T>::key() const {
    return key_;
}

template <typename T>
std::vector<std::pair<milvus::cachinglayer::cid_t,
                      std::unique_ptr<knowhere::Index<knowhere::IndexNode>>>>
MemVecIndexTranslator<T>::get_cells(
    const std::vector<milvus::cachinglayer::cid_t>& cids) const {
    AssertInfo(cids.size() == 1 && cids[0] == 0,
               "MemVecIndexTranslator only supports single cell");

    auto index = storage_type_ == milvus::cachinglayer::StorageType::MEMORY
                     ? load_index_into_memory()
                     : load_index_into_mmap();

    std::vector<
        std::pair<milvus::cachinglayer::cid_t,
                  std::unique_ptr<knowhere::Index<knowhere::IndexNode>>>>
        cells;
    cells.emplace_back(0, std::move(index));
    return cells;
}

template <typename T>
std::unique_ptr<knowhere::Index<knowhere::IndexNode>>
MemVecIndexTranslator<T>::load_index_into_memory() const {
    auto index_files = index::GetValueFromConfig<std::vector<std::string>>(
        config_, "index_files");
    AssertInfo(index_files.has_value(),
               "index file paths is empty when load index");

    std::unordered_set<std::string> pending_index_files(index_files->begin(),
                                                        index_files->end());

    LOG_INFO("load index files: {}", index_files->size());

    auto parallel_degree =
        static_cast<uint64_t>(DEFAULT_FIELD_MAX_MEMORY_LIMIT / FILE_SLICE_SIZE);
    std::map<std::string, FieldDataPtr> index_datas{};

    // try to read slice meta first
    std::string slice_meta_filepath;
    for (auto& file : pending_index_files) {
        auto file_name = file.substr(file.find_last_of('/') + 1);
        if (file_name == INDEX_FILE_SLICE_META) {
            slice_meta_filepath = file;
            pending_index_files.erase(file);
            break;
        }
    }

    // start read file span with active scope
    {
        // auto read_file_span =
        //     milvus::tracer::StartSpan("SegCoreReadIndexFile", &ctx);
        // auto read_scope =
        //     milvus::tracer::GetTracer()->WithActiveSpan(read_file_span);
        LOG_INFO("load with slice meta: {}", !slice_meta_filepath.empty());

        if (!slice_meta_filepath
                 .empty()) {  // load with the slice meta info, then we can load batch by batch
            std::string index_file_prefix = slice_meta_filepath.substr(
                0, slice_meta_filepath.find_last_of('/') + 1);

            auto result =
                file_manager_->LoadIndexToMemory({slice_meta_filepath});
            auto raw_slice_meta = result[INDEX_FILE_SLICE_META];
            Config meta_data = Config::parse(
                std::string(static_cast<const char*>(raw_slice_meta->Data()),
                            raw_slice_meta->Size()));

            for (auto& item : meta_data[META]) {
                std::string prefix = item[NAME];
                int slice_num = item[SLICE_NUM];
                auto total_len = static_cast<size_t>(item[TOTAL_LEN]);
                auto new_field_data = milvus::storage::CreateFieldData(
                    DataType::INT8, false, 1, total_len);

                std::vector<std::string> batch;
                batch.reserve(slice_num);
                for (auto i = 0; i < slice_num; ++i) {
                    std::string file_name = GenSlicedFileName(prefix, i);
                    batch.push_back(index_file_prefix + file_name);
                }

                auto batch_data = file_manager_->LoadIndexToMemory(batch);
                for (const auto& file_path : batch) {
                    const std::string file_name =
                        file_path.substr(file_path.find_last_of('/') + 1);
                    AssertInfo(batch_data.find(file_name) != batch_data.end(),
                               "lost index slice data: {}",
                               file_name);
                    auto data = batch_data[file_name];
                    new_field_data->FillFieldData(data->Data(), data->Size());
                }
                for (auto& file : batch) {
                    pending_index_files.erase(file);
                }

                AssertInfo(
                    new_field_data->IsFull(),
                    "index len is inconsistent after disassemble and assemble");
                index_datas[prefix] = new_field_data;
            }
        }

        if (!pending_index_files.empty()) {
            auto result =
                file_manager_->LoadIndexToMemory(std::vector<std::string>(
                    pending_index_files.begin(), pending_index_files.end()));
            for (auto&& index_data : result) {
                index_datas.insert(std::move(index_data));
            }
        }

        // read_file_span->End();
    }

    LOG_INFO("construct binary set...");
    BinarySet binary_set;
    for (auto& [key, data] : index_datas) {
        LOG_INFO("add index data to binary set: {}", key);
        auto size = data->Size();
        auto deleter = [&](uint8_t*) {};  // avoid repeated deconstruction
        auto buf = std::shared_ptr<uint8_t[]>(
            (uint8_t*)const_cast<void*>(data->Data()), deleter);
        binary_set.Append(key, buf, size);
    }

    // start engine load index span
    // auto span_load_engine =
    //     milvus::tracer::StartSpan("SegCoreEngineLoadIndex", &ctx);
    // auto engine_scope =
    //     milvus::tracer::GetTracer()->WithActiveSpan(span_load_engine);
    LOG_INFO("load index into Knowhere...");

    auto index_obj = knowhere::IndexFactory::Instance().Create<T>(
        index_type_, index_version_);

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

    auto stat = index_ptr->Deserialize(binary_set, config_);
    if (stat != knowhere::Status::success)
        PanicInfo(ErrorCode::UnexpectedError,
                  "failed to Deserialize index: {}",
                  KnowhereStatusString(stat));
    // span_load_engine->End();
    LOG_INFO("load vector index done");

    return index_ptr;
}

template <typename T>
std::unique_ptr<knowhere::Index<knowhere::IndexNode>>
MemVecIndexTranslator<T>::load_index_into_mmap() const {
    auto filepath =
        index::GetValueFromConfig<std::string>(config_, index::MMAP_FILE_PATH);
    AssertInfo(filepath.has_value(), "mmap filepath is empty when load index");

    std::filesystem::create_directories(
        std::filesystem::path(filepath.value()).parent_path());

    auto file = File::Open(filepath.value(), O_CREAT | O_TRUNC | O_RDWR);

    auto index_files = index::GetValueFromConfig<std::vector<std::string>>(
        config_, "index_files");
    AssertInfo(index_files.has_value(),
               "index file paths is empty when load index");

    std::unordered_set<std::string> pending_index_files(index_files->begin(),
                                                        index_files->end());

    LOG_INFO("load index files: {}", index_files.value().size());

    auto parallel_degree =
        static_cast<uint64_t>(DEFAULT_FIELD_MAX_MEMORY_LIMIT / FILE_SLICE_SIZE);

    // try to read slice meta first
    std::string slice_meta_filepath;
    for (auto& file : pending_index_files) {
        auto file_name = file.substr(file.find_last_of('/') + 1);
        if (file_name == INDEX_FILE_SLICE_META) {
            slice_meta_filepath = file;
            pending_index_files.erase(file);
            break;
        }
    }

    LOG_INFO("load with slice meta: {}", !slice_meta_filepath.empty());
    std::chrono::duration<double> load_duration_sum;
    std::chrono::duration<double> write_disk_duration_sum;
    if (!slice_meta_filepath
             .empty()) {  // load with the slice meta info, then we can load batch by batch
        std::string index_file_prefix = slice_meta_filepath.substr(
            0, slice_meta_filepath.find_last_of('/') + 1);
        std::vector<std::string> batch{};
        batch.reserve(parallel_degree);

        auto result = file_manager_->LoadIndexToMemory({slice_meta_filepath});
        auto raw_slice_meta = result[INDEX_FILE_SLICE_META];
        Config meta_data = Config::parse(
            std::string(static_cast<const char*>(raw_slice_meta->Data()),
                        raw_slice_meta->Size()));

        for (auto& item : meta_data[META]) {
            std::string prefix = item[NAME];
            int slice_num = item[SLICE_NUM];
            auto total_len = static_cast<size_t>(item[TOTAL_LEN]);
            auto HandleBatch = [&](int index) {
                auto start_load2_mem = std::chrono::system_clock::now();
                auto batch_data = file_manager_->LoadIndexToMemory(batch);
                load_duration_sum +=
                    (std::chrono::system_clock::now() - start_load2_mem);
                for (int j = index - batch.size() + 1; j <= index; j++) {
                    std::string file_name = GenSlicedFileName(prefix, j);
                    AssertInfo(batch_data.find(file_name) != batch_data.end(),
                               "lost index slice data");
                    auto data = batch_data[file_name];
                    auto start_write_file = std::chrono::system_clock::now();
                    auto written = file.Write(data->Data(), data->Size());
                    write_disk_duration_sum +=
                        (std::chrono::system_clock::now() - start_write_file);
                    AssertInfo(
                        written == data->Size(),
                        fmt::format("failed to write index data to disk {}: {}",
                                    filepath->data(),
                                    strerror(errno)));
                }
                for (auto& file : batch) {
                    pending_index_files.erase(file);
                }
                batch.clear();
            };

            for (auto i = 0; i < slice_num; ++i) {
                std::string file_name = GenSlicedFileName(prefix, i);
                batch.push_back(index_file_prefix + file_name);
                if (batch.size() >= parallel_degree) {
                    HandleBatch(i);
                }
            }
            if (batch.size() > 0) {
                HandleBatch(slice_num - 1);
            }
        }
    } else {
        //1. load files into memory
        auto start_load_files2_mem = std::chrono::system_clock::now();
        auto result = file_manager_->LoadIndexToMemory(std::vector<std::string>(
            pending_index_files.begin(), pending_index_files.end()));
        load_duration_sum +=
            (std::chrono::system_clock::now() - start_load_files2_mem);
        //2. write data into files
        auto start_write_file = std::chrono::system_clock::now();
        for (auto& [_, index_data] : result) {
            file.Write(index_data->Data(), index_data->Size());
        }
        write_disk_duration_sum +=
            (std::chrono::system_clock::now() - start_write_file);
    }
    // milvus::monitor::internal_storage_download_duration.Observe(
    //     std::chrono::duration_cast<std::chrono::milliseconds>(load_duration_sum)
    //         .count());
    // milvus::monitor::internal_storage_write_disk_duration.Observe(
    //     std::chrono::duration_cast<std::chrono::milliseconds>(
    //         write_disk_duration_sum)
    //         .count());
    file.Close();

    LOG_INFO("load index into Knowhere...");
    auto conf = config_;
    conf.erase(index::MMAP_FILE_PATH);
    conf[index::ENABLE_MMAP] = true;
    auto start_deserialize = std::chrono::system_clock::now();

    auto index_obj = knowhere::IndexFactory::Instance().Create<T>(
        index_type_, index_version_);
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

    auto stat = index_ptr->DeserializeFromFile(filepath.value(), conf);
    auto deserialize_duration =
        std::chrono::system_clock::now() - start_deserialize;
    if (stat != knowhere::Status::success) {
        PanicInfo(ErrorCode::UnexpectedError,
                  "failed to Deserialize index: {}",
                  KnowhereStatusString(stat));
    }
    // milvus::monitor::internal_storage_deserialize_duration.Observe(
    //     std::chrono::duration_cast<std::chrono::milliseconds>(
    //         deserialize_duration)
    //         .count());

    // auto dim = index_.Dim();
    // this->SetDim(index_.Dim());

    auto ok = unlink(filepath->data());
    AssertInfo(ok == 0,
               "failed to unlink mmap index file {}: {}",
               filepath.value(),
               strerror(errno));
    // LOG_INFO(
    //     "load vector index done, mmap_file_path:{}, download_duration:{}, "
    //     "write_files_duration:{}, deserialize_duration:{}",
    //     filepath.value(),
    //     std::chrono::duration_cast<std::chrono::milliseconds>(load_duration_sum)
    //         .count(),
    //     std::chrono::duration_cast<std::chrono::milliseconds>(
    //         write_disk_duration_sum)
    //         .count(),
    //     std::chrono::duration_cast<std::chrono::milliseconds>(
    //         deserialize_duration)
    //         .count());

    return index_ptr;
}

template class MemVecIndexTranslator<float>;
template class MemVecIndexTranslator<bin1>;
template class MemVecIndexTranslator<float16>;
template class MemVecIndexTranslator<bfloat16>;
template class MemVecIndexTranslator<int8>;

}  // namespace milvus::segcore::storagev1translator
