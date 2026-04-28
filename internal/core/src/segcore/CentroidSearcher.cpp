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

#include "segcore/CentroidSearcher.h"

#include <algorithm>
#include <numeric>
#include <shared_mutex>

#include "knowhere/comp/brute_force.h"
#include "knowhere/dataset.h"
#include "common/EasyAssert.h"

namespace milvus::segcore {

CentroidSearcher::CentroidSearcher(int64_t dim,
                                   const std::string& metric_type)
    : dim_(dim), metric_type_(metric_type) {
}

CentroidSearcher::~CentroidSearcher() = default;

void
CentroidSearcher::Update(const int64_t* seg_ids,
                          const float* centroids,
                          int64_t count) {
    std::unique_lock lock(mu_);
    seg_ids_.assign(seg_ids, seg_ids + count);
    centroids_.assign(centroids, centroids + count * dim_);
}

std::vector<int64_t>
CentroidSearcher::Order(const float* query_vec, int64_t nq) const {
    std::shared_lock lock(mu_);
    int64_t n = static_cast<int64_t>(seg_ids_.size());
    if (n == 0 || nq == 0) {
        return {};
    }

    // topK = n: we want ALL centroids ordered, not just top-few.
    int64_t topK = n;

    auto base_ds = knowhere::GenDataSet(n, dim_, centroids_.data());
    auto query_ds = knowhere::GenDataSet(nq, dim_, query_vec);

    // Output buffers: nq * topK
    std::vector<int64_t> offsets(nq * topK);
    std::vector<float> distances(nq * topK);

    knowhere::Json cfg;
    cfg[knowhere::meta::METRIC_TYPE] = metric_type_;
    cfg[knowhere::meta::TOPK] = topK;

    auto stat = knowhere::BruteForce::SearchWithBuf<float>(
        base_ds, query_ds,
        offsets.data(), distances.data(),
        cfg, nullptr, nullptr);

    AssertInfo(stat == knowhere::Status::success,
               "CentroidSearcher::Order BruteForce failed: {}",
               knowhere::Status2String(stat));

    // offsets[i] is the index into centroids_ (0..n-1), already sorted
    // by distance (best first per metric). Map to seg_ids.
    // For nq=1 (adaptive search), return first nq*topK = n entries.
    std::vector<int64_t> result(nq * topK);
    for (int64_t i = 0; i < nq * topK; ++i) {
        auto off = offsets[i];
        if (off >= 0 && off < n) {
            result[i] = seg_ids_[off];
        } else {
            result[i] = -1;  // invalid, shouldn't happen
        }
    }
    return result;
}

int64_t
CentroidSearcher::Count() const {
    std::shared_lock lock(mu_);
    return static_cast<int64_t>(seg_ids_.size());
}

}  // namespace milvus::segcore
