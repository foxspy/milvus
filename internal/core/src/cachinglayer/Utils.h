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

#include <cstdint>

namespace milvus::cachinglayer {

using uid_t = int64_t;
using cid_t = int64_t;

enum class StorageType {
    MEMORY = 0,  // including Anonymous mmap
    FILE_MMAP = 1,
    FILE = 2,

    // must be the last one, value will be the number of storage types
    COUNT,
};

class GlobalCellKey {
 public:
    GlobalCellKey(uint64_t slot_id, cid_t cid) : slot_id_(slot_id), cid_(cid) {
    }

    bool
    operator==(const GlobalCellKey& other) const {
        return slot_id_ == other.slot_id_ && cid_ == other.cid_;
    }

    uint64_t
    get_slot_id() const {
        return slot_id_;
    }
    cid_t
    get_cid() const {
        return cid_;
    }

 private:
    uint64_t slot_id_;
    cid_t cid_;
};

}  // namespace milvus::cachinglayer

namespace std {
template <>
struct hash<milvus::cachinglayer::GlobalCellKey> {
    size_t
    operator()(const milvus::cachinglayer::GlobalCellKey& key) const {
        return hash<uint64_t>()(key.get_slot_id()) ^
               hash<milvus::cachinglayer::cid_t>()(key.get_cid());
    }
};
}  // namespace std
