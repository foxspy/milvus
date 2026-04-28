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
#include <shared_mutex>
#include <string>
#include <vector>

namespace milvus::segcore {

// CentroidSearcher holds a set of segment centroids and provides
// SIMD-accelerated ordering of segments by distance to a query vector.
// It wraps knowhere::BruteForce internally.
//
// Thread safety: Update() and Order() are protected by a shared mutex.
// Multiple concurrent Order() calls are allowed; Update() is exclusive.
class CentroidSearcher {
 public:
    CentroidSearcher(int64_t dim, const std::string& metric_type);
    ~CentroidSearcher();

    // Replace all centroids. seg_ids and centroids are parallel arrays;
    // centroids is flat row-major [count × dim].
    // Called when PartitionStats are synced (low frequency).
    void
    Update(const int64_t* seg_ids, const float* centroids, int64_t count);

    // Return segment IDs ordered by distance to query_vec.
    // L2: ascending (nearest first). IP/COSINE: descending (best first).
    // nq query vectors, each of dimension dim_. Returns nq * count ordered IDs.
    // For adaptive search, nq=1 (order by first query vector).
    std::vector<int64_t>
    Order(const float* query_vec, int64_t nq) const;

    int64_t
    Count() const;

 private:
    int64_t dim_;
    std::string metric_type_;

    mutable std::shared_mutex mu_;
    std::vector<int64_t> seg_ids_;    // [N]
    std::vector<float> centroids_;    // [N × dim_], flat
};

}  // namespace milvus::segcore
