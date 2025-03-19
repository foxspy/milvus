#include "cachinglayer/EvictionManager.h"

#include <unordered_map>
#include <mutex>
#include <atomic>

namespace milvus::cachinglayer {

EvictionManager::EvictionManager(StorageType storage_type)
    : storage_type_(storage_type) {
}

void
EvictionManager::register_slot(uint64_t slot_id,
                               size_t num_cells) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (size_t i = 0; i < num_cells; ++i) {
        key_size_[GlobalCellKey(slot_id, i)] = 0;
    }
}

void
EvictionManager::unregister_slot(uint64_t slot_id,
                                  size_t num_cells) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (size_t i = 0; i < num_cells; ++i) {
        key_size_.erase(GlobalCellKey(slot_id, i));
    }
}

void
EvictionManager::notify_cell_inserted(const GlobalCellKey& key,
                                      size_t cell_size) {
    key_size_[key] = cell_size;
    space_usage_.fetch_add(cell_size);
}

void
EvictionManager::notify_cell_pinned(const GlobalCellKey& key) {
}

void
EvictionManager::notify_cell_unpinned(const GlobalCellKey& key) {
}

void
EvictionManager::notify_cell_evicted(const GlobalCellKey& key) {
    space_usage_.fetch_sub(key_size_[key]);
}

size_t
EvictionManager::bytes_used() const {
    return space_usage_.load();
}

StorageType
EvictionManager::storage_type() const {
    return storage_type_;
}

}  // namespace milvus::cachinglayer
