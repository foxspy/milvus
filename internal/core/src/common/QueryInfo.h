// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <vector>

#include "ArrayOffsets.h"
#include "common/Tracer.h"
#include "common/Types.h"
#include "knowhere/comp/search_hint.h"
#include "knowhere/config.h"

namespace milvus {

struct SearchIteratorV2Info {
    std::string token = "";
    uint32_t batch_size = 0;
    std::optional<float> last_bound = std::nullopt;
};

struct SearchInfo {
    int64_t topk_{0};
    int64_t group_size_{1};
    bool strict_group_size_{false};
    int64_t round_decimal_{0};
    FieldId field_id_;
    MetricType metric_type_;
    knowhere::Json search_params_;
    std::vector<FieldId>
        group_by_field_ids_;  // Group by field IDs (single or multi-field)
    tracer::TraceContext trace_ctx_;
    bool materialized_view_involved = false;
    bool iterative_filter_execution = false;
    std::optional<SearchIteratorV2Info> iterator_v2_info_ = std::nullopt;
    std::optional<std::string> json_path_;
    std::optional<milvus::DataType> json_type_;
    bool strict_cast_{false};
    std::shared_ptr<const IArrayOffsets> array_offsets_{
        nullptr};  // For element-level search
    bool global_refine_enable_{false};
    float search_topk_ratio_{0.0f};
    float refine_topk_ratio_{0.0f};
    // Global index head-index search hints (graph-search seeds) for this segment.
    // Each hint is a matched centroid's local row range plus query-to-centroid
    // distance; empty means default (graph) entry point. Set per-call by the global
    // index per-query search path. Carried per-call (never via the shared plan) so
    // concurrent per-segment searches do not race.
    std::vector<knowhere::SearchHint> search_hints_;
    // Worker-batched global index search: per-query graph-search seeds for this
    // segment. Outer index is the query row (0..num_queries-1); an empty inner
    // vector means that query uses the default entry point. Set per-call by the
    // worker-batched global index search path (segment x query marking). Preferred
    // over search_hints_ when non-empty. Carried per-call so concurrent per-segment
    // searches do not race.
    std::vector<std::vector<knowhere::SearchHint>> search_hints_per_query_;

    bool
    element_level() const {
        return array_offsets_ != nullptr;
    }

    bool
    has_group_by() const {
        return !group_by_field_ids_.empty();
    }
};

using SearchInfoPtr = std::shared_ptr<SearchInfo>;

}  // namespace milvus
