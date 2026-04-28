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

#include <exception>

#include "common/EasyAssert.h"
#include "segcore/reduce/IncrementalReduceHeap.h"
#include "segcore/reduce_incremental_c.h"

using namespace milvus::segcore;

CStatus
NewReduceHeap(CSearchPlan plan, int64_t nq, int64_t topK, CReduceHeap* out) {
    try {
        auto* heap = new IncrementalReduceHeap(plan, nq, topK);
        *out = static_cast<void*>(heap);
        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

CStatus
ReduceHeap_MergeBatch(CReduceHeap heap,
                      CSearchResult* search_results,
                      int count) {
    try {
        auto* h = static_cast<IncrementalReduceHeap*>(heap);
        h->MergeBatch(search_results, count);
        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

CStatus
ReduceHeap_GetThresholds(CReduceHeap heap, float* out_thresholds) {
    try {
        auto* h = static_cast<IncrementalReduceHeap*>(heap);
        h->GetThresholds(out_thresholds);
        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

CStatus
ReduceHeap_Finalize(CReduceHeap heap, CSearchResultDataBlobs* out_result) {
    try {
        auto* h = static_cast<IncrementalReduceHeap*>(heap);
        h->Finalize(out_result);
        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

void
DeleteReduceHeap(CReduceHeap heap) {
    auto* h = static_cast<IncrementalReduceHeap*>(heap);
    delete h;
}
