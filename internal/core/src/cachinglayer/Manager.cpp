#include "cachinglayer/Manager.h"

#include <memory>

#include "cachinglayer/CacheSlot.h"
#include "cachinglayer/Translator.h"
#include "cachinglayer/Utils.h"

namespace milvus::cachinglayer {

Manager&
Manager::GetInstance() {
    static Manager instance;
    return instance;
}

Manager::Manager() {
    for (int i = 0; i < static_cast<int>(StorageType::COUNT); ++i) {
        eviction_managers_[i] =
            std::make_unique<EvictionManager>(static_cast<StorageType>(i));
    }
}

Manager::~Manager() {
}

size_t
Manager::bytes_used(StorageType storage_type) const {
    return eviction_managers_[static_cast<int>(storage_type)]->bytes_used();
}

size_t
Manager::memory_overhead() const {
    // TODO: calculate memory overhead
    return 0;
}

EvictionManager*
Manager::get_eviction_manager(StorageType storage_type) const {
    return eviction_managers_[static_cast<int>(storage_type)].get();
}
}  // namespace milvus::cachinglayer
