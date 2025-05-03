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

#include "cachinglayer/Translator.h"
#include "index/Index.h"
#include "segcore/ChunkedSegmentSealedImpl.h"

namespace milvus::index {

class IndexTranslator
    : public milvus::cachinglayer::Translator<milvus::index::IndexBase> {
 public:
    IndexTranslator(milvus::index::CreateIndexInfo index_info,
                    const milvus::segcore::LoadIndexInfo* load_index_info,
                    milvus::tracer::TraceContext ctx,
                    milvus::storage::FileManagerContext file_manager_context,
                    Config config);
    ~IndexTranslator() override = default;
    size_t
    num_cells() const override;
    milvus::cachinglayer::cid_t
    cell_id_of(milvus::cachinglayer::uid_t uid) const override;
    milvus::cachinglayer::ResourceUsage
    estimated_byte_size_of_cell(milvus::cachinglayer::cid_t cid) const override;
    const std::string&
    key() const override;
    std::vector<std::pair<milvus::cachinglayer::cid_t,
                          std::unique_ptr<milvus::index::IndexBase>>>
    get_cells(const std::vector<milvus::cachinglayer::cid_t>& cids) override;
    Meta*
    meta() override;

 private:
    struct IndexLoadInfo {
        bool enable_mmap;
        std::string mmap_dir_path;
        std::string index_id;
        std::string segment_id;
        std::string field_id;
    };

    milvus::index::CreateIndexInfo index_info_;
    milvus::segcore::LoadIndexInfo load_index_info_;
    milvus::tracer::TraceContext ctx_;
    milvus::storage::FileManagerContext file_manager_context_;
    Config config_;
    std::string index_key_;

    IndexLoadInfo index_load_info_;
};
}  // namespace milvus::index
