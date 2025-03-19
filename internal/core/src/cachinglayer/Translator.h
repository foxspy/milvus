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

#include <memory>
#include <vector>
#include <utility>
#include <type_traits>

#include "cachinglayer/Utils.h"

namespace milvus::cachinglayer {

template <typename CellT>
class Translator {
 public:
    using value_type = CellT;

    virtual size_t
    num_cells() const = 0;
    virtual cid_t
    cell_id_of(uid_t uid) const = 0;
    virtual StorageType
    storage_type() const = 0;
    // to identify a CacheSlot.
    virtual const std::string&
    key() const = 0;

    // Translator may choose to fetch more than requested cells.
    virtual std::vector<std::pair<cid_t, std::unique_ptr<CellT>>>
    get_cells(const std::vector<cid_t>& cids) const = 0;
    virtual ~Translator() = default;
};

}  // namespace milvus::cachinglayer
