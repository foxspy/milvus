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

#pragma once

#include "cachinglayer/Translator.h"
#include "cachinglayer/Utils.h"
#include "common/Types.h"
#include "common/FieldMeta.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "knowhere/index/index.h"
#include "storage/DiskFileManagerImpl.h"

namespace milvus::segcore::storagev1translator {

template <typename T>
class DiskVecIndexTranslator : public milvus::cachinglayer::Translator<
                                   knowhere::Index<knowhere::IndexNode>> {
 public:
    DiskVecIndexTranslator(
        int64_t segment_id,
        int64_t field_id,
        knowhere::IndexType index_type,
        knowhere::IndexVersion index_version,
        Config config,
        std::shared_ptr<storage::DiskFileManagerImpl> file_manager);

 private:
    size_t
    estimated_byte_size_of_cell(milvus::cachinglayer::cid_t cid) const override;

    size_t
    num_cells() const override;

    milvus::cachinglayer::cid_t
    cell_id_of(milvus::cachinglayer::uid_t uid) const override;

    milvus::cachinglayer::StorageType
    storage_type() const override;

    const std::string&
    key() const override;

    std::vector<
        std::pair<milvus::cachinglayer::cid_t,
                  std::unique_ptr<knowhere::Index<knowhere::IndexNode>>>>
    get_cells(
        const std::vector<milvus::cachinglayer::cid_t>& cids) const override;

 private:
    std::unique_ptr<knowhere::Index<knowhere::IndexNode>>
    load_index_into_file() const;

    knowhere::Json
    update_load_json() const;

    int64_t segment_id_;
    int64_t field_id_;
    std::string key_;

    knowhere::IndexType index_type_;
    knowhere::IndexVersion index_version_;

    Config config_;
    milvus::cachinglayer::StorageType storage_type_;

    std::shared_ptr<storage::DiskFileManagerImpl> file_manager_;
};

}  // namespace milvus::segcore::storagev1translator