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

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "common/type_c.h"
#include "segcore/plan_c.h"
#include "segcore/reduce_c.h"

typedef void* CReduceHeap;

CStatus
NewReduceHeap(CSearchPlan plan, int64_t nq, int64_t topK, CReduceHeap* out);

CStatus
ReduceHeap_MergeBatch(CReduceHeap heap,
                      CSearchResult* search_results,
                      int count);

CStatus
ReduceHeap_GetThresholds(CReduceHeap heap, float* out_thresholds);

CStatus
ReduceHeap_Finalize(CReduceHeap heap, CSearchResultDataBlobs* out_result);

void
DeleteReduceHeap(CReduceHeap heap);

#ifdef __cplusplus
}
#endif
