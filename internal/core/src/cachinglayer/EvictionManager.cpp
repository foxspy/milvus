#include "cachinglayer/EvictionManager.h"

#include <unordered_map>
#include <mutex>
#include <atomic>

namespace milvus::cachinglayer {

EvictionManager::EvictionManager(StorageType storage_type, size_t max_size)
    : dlist_(max_size, {std::chrono::seconds(10)}),
      storage_type_(storage_type) {
}

// void
// EvictionManager::register_slot(uint64_t slot_id,
//                                size_t num_cells) {
// }

// void
// EvictionManager::unregister_slot(uint64_t slot_id,
//                                   size_t num_cells) {
// }

size_t
EvictionManager::bytes_used() const {
    return space_usage_.load();
}

StorageType
EvictionManager::storage_type() const {
    return storage_type_;
}

internal::DList*
EvictionManager::dlist() {
    return &dlist_;
}

}  // namespace milvus::cachinglayer
