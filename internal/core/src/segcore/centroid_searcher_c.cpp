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

#include "segcore/centroid_searcher_c.h"
#include "segcore/CentroidSearcher.h"
#include "common/EasyAssert.h"

using namespace milvus::segcore;

CStatus
NewCentroidSearcher(int64_t dim,
                    const char* metric_type,
                    CCentroidSearcher* out) {
    try {
        auto* s = new CentroidSearcher(dim, std::string(metric_type));
        *out = static_cast<void*>(s);
        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

CStatus
CentroidSearcher_Update(CCentroidSearcher searcher,
                        const int64_t* seg_ids,
                        const float* centroids,
                        int64_t count) {
    try {
        auto* s = static_cast<CentroidSearcher*>(searcher);
        s->Update(seg_ids, centroids, count);
        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

CStatus
CentroidSearcher_Order(CCentroidSearcher searcher,
                       const float* query_vec,
                       int64_t nq,
                       int64_t* out_seg_ids,
                       int64_t* out_count) {
    try {
        auto* s = static_cast<CentroidSearcher*>(searcher);
        auto result = s->Order(query_vec, nq);
        *out_count = s->Count();
        std::copy(result.begin(), result.end(), out_seg_ids);
        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

int64_t
CentroidSearcher_Count(CCentroidSearcher searcher) {
    auto* s = static_cast<CentroidSearcher*>(searcher);
    return s->Count();
}

void
DeleteCentroidSearcher(CCentroidSearcher searcher) {
    delete static_cast<CentroidSearcher*>(searcher);
}
