#include "index/IndexTranslator.h"
#include "index/IndexFactory.h"
#include "segcore/load_index_c.h"
#include <utility>

namespace milvus::index {

IndexTranslator::IndexTranslator(
    milvus::index::CreateIndexInfo index_info,
    const milvus::segcore::LoadIndexInfo* load_index_info,
    milvus::tracer::TraceContext ctx,
    milvus::storage::FileManagerContext file_manager_context,
    Config config)
    : index_info_(std::move(index_info)),
      ctx_(ctx),
      file_manager_context_(std::move(file_manager_context)),
      config_(std::move(config)),
      index_key_(fmt::format("seg_{}_i_{}",
                             load_index_info->segment_id,
                             load_index_info->field_id)),
      index_load_info_({load_index_info->enable_mmap,
                        load_index_info->mmap_dir_path,
                        std::to_string(load_index_info->index_id),
                        std::to_string(load_index_info->segment_id),
                        std::to_string(load_index_info->field_id)}) {
}

size_t
IndexTranslator::num_cells() const {
    return 1;
}

milvus::cachinglayer::cid_t
IndexTranslator::cell_id_of(milvus::cachinglayer::uid_t uid) const {
    return 0;
}

milvus::cachinglayer::ResourceUsage
IndexTranslator::estimated_byte_size_of_cell(
    milvus::cachinglayer::cid_t cid) const {
    auto request =
        EstimateLoadIndexResource((const CLoadIndexInfo)&load_index_info_);
    int64_t memory_cost = request.final_memory_cost * 1024 * 1024 * 1024;
    int64_t disk_cost = request.final_disk_cost * 1024 * 1024 * 1024;
    return milvus::cachinglayer::ResourceUsage{memory_cost, disk_cost};
}

const std::string&
IndexTranslator::key() const {
    return index_key_;
}

std::vector<std::pair<milvus::cachinglayer::cid_t,
                      std::unique_ptr<milvus::index::IndexBase>>>
IndexTranslator::get_cells(const std::vector<cid_t>& cids) {
    std::unique_ptr<milvus::index::IndexBase> index =
        milvus::index::IndexFactory::GetInstance().CreateIndex(
            index_info_, file_manager_context_);
    if (index_load_info_.enable_mmap && index->IsMmapSupported()) {
        AssertInfo(!index_load_info_.mmap_dir_path.empty(),
                   "mmap directory path is empty");
        auto filepath = std::filesystem::path(index_load_info_.mmap_dir_path) /
                        "index_files" / index_load_info_.index_id /
                        index_load_info_.segment_id / index_load_info_.field_id;

        config_[milvus::index::ENABLE_MMAP] = "true";
        config_[milvus::index::MMAP_FILE_PATH] = filepath.string();
    }

    LOG_DEBUG("load index with configs: {}", config_.dump());
    index->Load(ctx_, config_);

    std::vector<std::pair<cid_t, std::unique_ptr<milvus::index::IndexBase>>>
        result;
    result.emplace_back(std::make_pair(0, std::move(index)));
    return result;
}

Meta*
IndexTranslator::meta() {
    return nullptr;
}
}  // namespace milvus::index
