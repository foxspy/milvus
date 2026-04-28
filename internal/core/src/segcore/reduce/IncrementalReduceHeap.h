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
#include <queue>
#include <unordered_map>
#include <vector>

#include "common/QueryResult.h"
#include "segcore/reduce_c.h"

namespace milvus::segcore {

// IncrementalReduceHeap maintains cross-batch top-K state for adaptive search.
// nq independent heaps, each holding up to topK entries.
//
// CSearch (via AsyncSearch) negates L2 distances so that all metrics follow
// "larger distance = better" after the search: L2 becomes negated L2 (closer
// vectors have larger, less-negative values), and IP/COSINE are unchanged.
// The heap always uses is_descending=true: root = MIN distance = worst-of-top-K.
// A new entry e beats the root iff e.distance > root.distance (e is better).
class IncrementalReduceHeap {
 public:
    IncrementalReduceHeap(CSearchPlan plan, int64_t nq, int64_t topK);
    ~IncrementalReduceHeap();

    // Merge a batch of per-segment search results into the heaps.
    // search_results is an array of CSearchResult pointers with length count.
    // Throws std::exception on failure.
    void
    MergeBatch(CSearchResult* search_results, int count);

    // Read current top-K thresholds (one per nq).
    // Returns -FLT_MAX sentinel for unfilled heaps (any real distance beats it).
    // When full, returns the worst-of-top-K distance (heap root = MIN distance).
    // Throws std::exception on failure.
    void
    GetThresholds(float* out_thresholds /* len == nq */);

    // Consume the heaps to produce final search result blobs.
    // After Finalize, the heap is not reusable.
    // Throws std::exception on failure.
    void
    Finalize(CSearchResultDataBlobs* out_result);

 private:
    struct Entry {
        float distance;
        int64_t segment_id;
        int64_t offset;
    };

    struct HeapCmp {
        bool is_descending;  // always true: comp(a,b)=a>b → min-at-root (worst-of-top-K)
        bool
        operator()(const Entry& a, const Entry& b) const;
    };

    CSearchPlan plan_;
    int64_t nq_;
    int64_t topK_;
    bool finalized_;
    HeapCmp cmp_;  // single comparator instance, shared by all nq heaps

    // One heap per nq; fixed-size top-K with worst-of-top-K at root.
    std::vector<std::priority_queue<Entry, std::vector<Entry>, HeapCmp>> heaps_;

    // Maps segment_id → one representative SearchResult* carrying segment_
    // pointer, used by Finalize to call FillPrimaryKey and FillTargetEntry.
    std::unordered_map<int64_t, milvus::SearchResult*> seg_map_;
};

}  // namespace milvus::segcore
