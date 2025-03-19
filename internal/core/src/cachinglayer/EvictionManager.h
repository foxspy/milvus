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

#include <atomic>
#include <cstdint>
#include <unordered_map>
#include <mutex>

#include "cachinglayer/Utils.h"

namespace milvus::cachinglayer {

// EvictionManager decides when and which cells to evict under resource pressure.
//
// To avoid frequent eviction(and frequently making eviction decisions), EvictionManager perform eviction
// when resource usage is at high waterlevel(e.g. 90%) and purge to a lower waterlevel(e.g. 80%).
class EvictionManager {
 public:

    EvictionManager(StorageType storage_type);

    void
    register_slot(uint64_t slot_id, size_t num_cells);
    void
    unregister_slot(uint64_t slot_id,
                    size_t num_cells);
    void
    notify_cell_inserted(const GlobalCellKey& key,
                         size_t cell_size);
    void
    notify_cell_pinned(const GlobalCellKey& key);
    void
    notify_cell_unpinned(const GlobalCellKey& key);
    void
    notify_cell_evicted(const GlobalCellKey& key);

    size_t
    bytes_used() const;

    StorageType
    storage_type() const;

 private:
    std::unordered_map<GlobalCellKey, size_t> key_size_;
    std::mutex mutex_;
    std::atomic<size_t> space_usage_;

    StorageType storage_type_;
};

}  // namespace milvus::cachinglayer
