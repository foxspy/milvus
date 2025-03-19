#pragma once

#include <string>
#include <vector>

#include "cachinglayer/Translator.h"
#include "cachinglayer/Utils.h"
#include "common/Types.h"
#include "mmap/ChunkedColumn.h"

namespace milvus::segcore::storagev1translator {

class ChunkedColumnTranslator
    : public milvus::cachinglayer::Translator<milvus::ChunkedColumnBase> {
 public:
    ChunkedColumnTranslator(int64_t segment_id,
                            FieldMeta field_meta,
                            FieldDataInfo field_data_info,
                            std::vector<std::string> insert_files,
                            milvus::cachinglayer::StorageType storage_type);

    size_t
    num_cells() const override;
    milvus::cachinglayer::cid_t
    cell_id_of(milvus::cachinglayer::uid_t uid) const override;
    milvus::cachinglayer::StorageType
    storage_type() const override;
    const std::string&
    key() const override;
    // each calling of this will trigger a new download.
    std::vector<std::pair<milvus::cachinglayer::cid_t,
                          std::unique_ptr<milvus::ChunkedColumnBase>>>
    get_cells(
        const std::vector<milvus::cachinglayer::cid_t>& cids) const override;

 private:
    std::unique_ptr<milvus::ChunkedColumnBase>
    load_column_in_memory() const;

    std::unique_ptr<milvus::ChunkedColumnBase>
    load_column_in_mmap() const;

    int64_t segment_id_;
    std::string key_;
    FieldMeta field_meta_;
    FieldDataInfo field_data_info_;
    std::vector<std::string> insert_files_;
    milvus::cachinglayer::StorageType storage_type_;
};

}  // namespace milvus::segcore::storagev1translator
