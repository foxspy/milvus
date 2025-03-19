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

#include <array>
#include <memory>

#include "cachinglayer/CacheSlot.h"
#include "cachinglayer/EvictionManager.h"
#include "cachinglayer/Translator.h"
#include "cachinglayer/Utils.h"

namespace milvus::cachinglayer {

// TODO: add config system, include config for the overall caching layer, and for individual cache slots.
class Manager {
 public:
    static Manager&
    GetInstance();

    Manager(const Manager&) = delete;
    Manager&
    operator=(const Manager&) = delete;
    Manager(Manager&&) = delete;
    Manager&
    operator=(Manager&&) = delete;

    ~Manager();

    template <typename CellT>
    std::unique_ptr<CacheSlot<CellT>>
    CreateCacheSlot(std::unique_ptr<Translator<CellT>> translator) {
        return std::make_unique<CacheSlot<CellT>>(
            std::move(translator),
            get_eviction_manager(translator->storage_type()));
    }

    size_t
    bytes_used(StorageType storage_type) const;

    // memory overhead for managing all cache slots/cells/translators/policies.
    size_t
    memory_overhead() const;

    EvictionManager*
    get_eviction_manager(StorageType storage_type) const;

 private:
    Manager();  // Private constructor
    std::array<std::unique_ptr<EvictionManager>,
               static_cast<int>(StorageType::COUNT)>
        eviction_managers_;
};  // class Manager

}  // namespace milvus::cachinglayer
