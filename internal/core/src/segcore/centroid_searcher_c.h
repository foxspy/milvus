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

typedef void* CCentroidSearcher;

CStatus
NewCentroidSearcher(int64_t dim,
                    const char* metric_type,
                    CCentroidSearcher* out);

CStatus
CentroidSearcher_Update(CCentroidSearcher searcher,
                        const int64_t* seg_ids,
                        const float* centroids,
                        int64_t count);

// Order segments by distance to query_vec (nq query vectors).
// out_seg_ids must be pre-allocated by caller: [nq * count].
// out_count receives the number of centroids (same as Count()).
CStatus
CentroidSearcher_Order(CCentroidSearcher searcher,
                       const float* query_vec,
                       int64_t nq,
                       int64_t* out_seg_ids,
                       int64_t* out_count);

int64_t
CentroidSearcher_Count(CCentroidSearcher searcher);

void
DeleteCentroidSearcher(CCentroidSearcher searcher);

#ifdef __cplusplus
}
#endif
