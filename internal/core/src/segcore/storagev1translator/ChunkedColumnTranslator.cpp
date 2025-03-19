#include "segcore/storagev1translator/ChunkedColumnTranslator.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "cachinglayer/Translator.h"
#include "cachinglayer/Utils.h"
#include "common/ChunkWriter.h"
#include "common/Types.h"
#include "common/SystemProperty.h"
#include "fmt/format.h"
#include "segcore/Utils.h"
#include "storage/ThreadPools.h"
#include "storage/MmapManager.h"

namespace milvus::segcore::storagev1translator {

ChunkedColumnTranslator::ChunkedColumnTranslator(
    int64_t segment_id,
    FieldMeta field_meta,
    FieldDataInfo field_data_info,
    std::vector<std::string> insert_files,
    milvus::cachinglayer::StorageType storage_type)
    : segment_id_(segment_id),
      key_(fmt::format("seg_{}_f_{}", segment_id, field_data_info.field_id)),
      field_meta_(field_meta),
      field_data_info_(field_data_info),
      insert_files_(insert_files),
      storage_type_(storage_type) {
    AssertInfo(!SystemProperty::Instance().IsSystem(
                   FieldId(field_data_info_.field_id)),
               "ChunkedColumnTranslator not supported for system field");
}

size_t
ChunkedColumnTranslator::num_cells() const {
    return 1;
}

milvus::cachinglayer::cid_t
ChunkedColumnTranslator::cell_id_of(milvus::cachinglayer::uid_t uid) const {
    return 0;
}

milvus::cachinglayer::StorageType
ChunkedColumnTranslator::storage_type() const {
    return storage_type_;
}

const std::string&
ChunkedColumnTranslator::key() const {
    return key_;
}

std::vector<std::pair<milvus::cachinglayer::cid_t,
                      std::unique_ptr<milvus::ChunkedColumnBase>>>
ChunkedColumnTranslator::get_cells(
    const std::vector<milvus::cachinglayer::cid_t>& cids) const {
    AssertInfo(cids.size() == 1 && cids[0] == 0,
               "ChunkedColumnTranslator only supports single cell");
    auto parallel_degree =
        static_cast<uint64_t>(DEFAULT_FIELD_MAX_MEMORY_LIMIT / FILE_SLICE_SIZE);
    auto& pool = ThreadPools::GetThreadPool(milvus::ThreadPoolPriority::MIDDLE);
    pool.Submit(LoadArrowReaderFromRemote,
                insert_files_,
                field_data_info_.arrow_reader_channel);
    LOG_INFO("segment {} submits load field {} task to thread pool",
             segment_id_,
             field_data_info_.field_id);

    auto column = storage_type_ == milvus::cachinglayer::StorageType::MEMORY
                      ? load_column_in_memory()
                      : load_column_in_mmap();
    std::vector<std::pair<milvus::cachinglayer::cid_t,
                          std::unique_ptr<milvus::ChunkedColumnBase>>>
        cells;
    cells.emplace_back(0, std::move(column));
    return cells;
}

std::unique_ptr<milvus::ChunkedColumnBase>
ChunkedColumnTranslator::load_column_in_memory() const {
    std::unique_ptr<milvus::ChunkedColumnBase> column{};

    auto data_type = field_meta_.get_data_type();
    switch (data_type) {
        case milvus::DataType::STRING:
        case milvus::DataType::VARCHAR:
        case milvus::DataType::TEXT: {
            column = std::make_unique<ChunkedVariableColumn<std::string>>(
                field_meta_);
            break;
        }
        case milvus::DataType::JSON: {
            column = std::make_unique<ChunkedVariableColumn<milvus::Json>>(
                field_meta_);
            break;
        }
        case milvus::DataType::ARRAY: {
            column = std::make_unique<ChunkedArrayColumn>(field_meta_);
            break;
        }
        case milvus::DataType::VECTOR_SPARSE_FLOAT: {
            column = std::make_unique<ChunkedSparseFloatColumn>(field_meta_);
            break;
        }
        default: {
            column = std::make_unique<ChunkedColumn>(field_meta_);
            break;
        }
    }
    std::shared_ptr<milvus::ArrowDataWrapper> r;
    while (field_data_info_.arrow_reader_channel->pop(r)) {
        auto chunk =
            create_chunk(field_meta_,
                         IsVectorDataType(data_type) &&
                                 !IsSparseFloatVectorDataType(data_type)
                             ? field_meta_.get_dim()
                             : 1,
                         r->reader);
        column->AddChunk(chunk);
    }
    AssertInfo(column->NumRows() == field_data_info_.row_count,
               fmt::format("data lost while loading column {}: loaded "
                           "num rows {} but expected {}",
                           field_data_info_.field_id,
                           column->NumRows(),
                           field_data_info_.row_count));
    return column;
}

std::unique_ptr<milvus::ChunkedColumnBase>
ChunkedColumnTranslator::load_column_in_mmap() const {
    auto filepath = std::filesystem::path(field_data_info_.mmap_dir_path) /
                    std::to_string(segment_id_) /
                    std::to_string(field_data_info_.field_id);
    auto dir = filepath.parent_path();
    std::filesystem::create_directories(dir);

    auto file = File::Open(filepath.string(), O_CREAT | O_TRUNC | O_RDWR);

    auto data_type = field_meta_.get_data_type();

    // write the field data to disk
    std::vector<uint64_t> indices{};
    std::vector<std::vector<uint64_t>> element_indices{};

    std::shared_ptr<milvus::ArrowDataWrapper> r;
    size_t file_offset = 0;
    std::vector<std::shared_ptr<Chunk>> chunks;
    while (field_data_info_.arrow_reader_channel->pop(r)) {
        auto chunk =
            create_chunk(field_meta_,
                         IsVectorDataType(data_type) &&
                                 !IsSparseFloatVectorDataType(data_type)
                             ? field_meta_.get_dim()
                             : 1,
                         file,
                         file_offset,
                         r->reader);
        file_offset += chunk->Size();
        chunks.push_back(chunk);
    }
    std::unique_ptr<ChunkedColumnBase> column{};
    auto num_rows = field_data_info_.row_count;
    switch (data_type) {
        case milvus::DataType::STRING:
        case milvus::DataType::VARCHAR:
        case milvus::DataType::TEXT: {
            column =
                std::make_unique<ChunkedVariableColumn<std::string>>(chunks);
            break;
        }
        case milvus::DataType::JSON: {
            column =
                std::make_unique<ChunkedVariableColumn<milvus::Json>>(chunks);
            break;
        }
        case milvus::DataType::ARRAY: {
            column = std::make_unique<ChunkedArrayColumn>(chunks);
            break;
        }
        case milvus::DataType::VECTOR_SPARSE_FLOAT: {
            column = std::make_unique<ChunkedSparseFloatColumn>(chunks);
            break;
        }
        default: {
            column = std::make_unique<ChunkedColumn>(chunks);
            break;
        }
    }
    auto ok = unlink(filepath.c_str());
    AssertInfo(ok == 0,
               fmt::format("failed to unlink mmap data file {}, err: {}",
                           filepath.c_str(),
                           strerror(errno)));
    return column;
}

}  // namespace milvus::segcore::storagev1translator
