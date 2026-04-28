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

#include "segcore/reduce/IncrementalReduceHeap.h"

#include <algorithm>
#include <cfloat>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/Consts.h"
#include "common/EasyAssert.h"
#include "knowhere/comp/index_param.h"
#include "query/PlanImpl.h"
#include "segcore/SegmentInterface.h"
#include "segcore/reduce/Reduce.h"

namespace milvus::segcore {

bool
IncrementalReduceHeap::HeapCmp::operator()(const Entry& a,
                                            const Entry& b) const {
    // priority_queue is a max-heap: the element where operator() returns false
    // ends up at the top.  We always use is_descending=true so that root = MIN
    // (worst-of-top-K), regardless of metric type.
    //
    // After CSearch (AsyncSearch), all distances follow "larger = better":
    //   L2: negated by AsyncSearch → larger negated-dist = closer = better.
    //   IP/COSINE: unchanged → larger score = better.
    //
    // comp(a, b) = a.distance > b.distance.
    // comp(a, b)=false  ↔  a.distance <= b.distance  ↔  a is ≤ b.
    // Top = element where comp=false for all others = MIN distance = WORST.
    // A new entry e beats the top when comp(e, top)=true: e.distance > top = BETTER.
    return is_descending ? a.distance > b.distance : a.distance < b.distance;
}

IncrementalReduceHeap::IncrementalReduceHeap(CSearchPlan plan,
                                             int64_t nq,
                                             int64_t topK)
    : plan_(plan), nq_(nq), topK_(topK), finalized_(false) {
    auto* p = static_cast<milvus::query::Plan*>(plan);
    const auto& metric = p->plan_node_->search_info_.metric_type_;
    // After CSearch (via AsyncSearch), distances follow "larger = better":
    //   - L2: negated by AsyncSearch, so larger negated-distance = closer = better.
    //   - IP/COSINE: not negated, so larger score = better.
    // Therefore always use is_descending=true: root = min = worst-of-top-K.
    // cmp_(a, b) = a.distance > b.distance → max-heap: root = MIN distance = WORST.
    // cmp_(e, root): e.distance > root.distance → e is better → swap in.
    (void)metric;  // metric type stored in plan_, kept for future reference
    cmp_.is_descending = true;
    heaps_.reserve(nq);
    for (int64_t q = 0; q < nq; ++q) {
        heaps_.emplace_back(cmp_);
    }
}

IncrementalReduceHeap::~IncrementalReduceHeap() = default;

void
IncrementalReduceHeap::MergeBatch(CSearchResult* search_results, int count) {
    AssertInfo(!finalized_, "MergeBatch called after Finalize");
    for (int i = 0; i < count; ++i) {
        auto* sr = static_cast<milvus::SearchResult*>(search_results[i]);
        // Lazily record the segment pointer so Finalize can call FillPrimaryKey
        // and FillTargetEntry via the original SegmentInterface.
        auto* seg =
            static_cast<milvus::segcore::SegmentInterface*>(sr->segment_);
        int64_t seg_id = seg->get_segment_id();
        if (seg_map_.find(seg_id) == seg_map_.end()) {
            seg_map_[seg_id] = sr;
        }

        // Each SearchResult's flat layout: nq * topK, row-major.
        // Rows with INVALID_SEG_OFFSET are filtered-out placeholders.
        for (int64_t q = 0; q < nq_; ++q) {
            for (int64_t k = 0; k < topK_; ++k) {
                int64_t idx = q * topK_ + k;
                if (idx >= static_cast<int64_t>(sr->distances_.size())) {
                    break;
                }
                if (sr->seg_offsets_[idx] == INVALID_SEG_OFFSET) {
                    continue;
                }
                Entry e{sr->distances_[idx], seg_id, sr->seg_offsets_[idx]};
                auto& h = heaps_[q];
                if (static_cast<int64_t>(h.size()) < topK_) {
                    h.push(e);
                } else if (cmp_(e, h.top())) {
                    // new entry beats the current worst-of-top-K → swap in
                    h.pop();
                    h.push(e);
                }
            }
        }
    }
}

void
IncrementalReduceHeap::GetThresholds(float* out_thresholds) {
    AssertInfo(!finalized_, "GetThresholds called after Finalize");
    for (int64_t q = 0; q < nq_; ++q) {
        if (static_cast<int64_t>(heaps_[q].size()) < topK_) {
            // Heap not full: any real candidate beats this sentinel.
            out_thresholds[q] = cmp_.is_descending ? -FLT_MAX : FLT_MAX;
        } else {
            out_thresholds[q] = heaps_[q].top().distance;
        }
    }
}

void
IncrementalReduceHeap::Finalize(CSearchResultDataBlobs* out_result) {
    AssertInfo(!finalized_, "Finalize called twice");
    finalized_ = true;

    // Step 1: collect entries per (segment_id, nq).
    // seg_entries[seg_id][q] = list of (distance, offset) surviving top-K.
    std::unordered_map<int64_t,
                       std::vector<std::vector<std::pair<float, int64_t>>>>
        seg_entries;
    for (auto& [seg_id, _] : seg_map_) {
        seg_entries[seg_id].resize(nq_);
    }
    for (int64_t q = 0; q < nq_; ++q) {
        auto& h = heaps_[q];
        while (!h.empty()) {
            const auto& e = h.top();
            seg_entries[e.segment_id][q].emplace_back(e.distance, e.offset);
            h.pop();
        }
    }
    // Heap pops worst-first (root = worst-of-top-K = MIN distance).
    // ReduceHelper's K-way merge expects entries sorted best-first (larger = better).
    // Reverse each segment's per-nq list so best entries appear first.
    for (auto& [seg_id, per_nq] : seg_entries) {
        for (int64_t q = 0; q < nq_; ++q) {
            std::reverse(per_nq[q].begin(), per_nq[q].end());
        }
    }

    // Step 2: build one SearchResult per segment in the nq * topK flat layout
    // that ReduceHelper::FilterInvalidSearchResult expects.
    // Slots with no entry are filled with INVALID_SEG_OFFSET / 0.0f.
    std::vector<std::unique_ptr<milvus::SearchResult>> owned_srs;
    std::vector<milvus::SearchResult*> sr_ptrs;

    for (auto& [seg_id, per_nq] : seg_entries) {
        auto* orig_sr = seg_map_.at(seg_id);

        auto sr = std::make_unique<milvus::SearchResult>();
        sr->total_nq_ = nq_;
        sr->unity_topK_ = topK_;
        sr->total_data_cnt_ = orig_sr->total_data_cnt_;
        sr->segment_ = orig_sr->segment_;

        // Allocate nq_ * topK_ flat buffers, all slots initially invalid.
        int64_t flat_size = nq_ * topK_;
        sr->distances_.assign(flat_size, 0.0f);
        sr->seg_offsets_.assign(flat_size, INVALID_SEG_OFFSET);

        for (int64_t q = 0; q < nq_; ++q) {
            int64_t k = 0;
            for (auto& [dist, off] : per_nq[q]) {
                int64_t idx = q * topK_ + k;
                sr->distances_[idx] = dist;
                sr->seg_offsets_[idx] = off;
                ++k;
            }
        }

        // FilterInvalidSearchResult (called by FillPrimaryKey) will compact and
        // fill topk_per_nq_prefix_sum_.  Leave it empty here so the helper can
        // see it needs to be populated (it always overwrites it).
        sr->topk_per_nq_prefix_sum_.resize(nq_ + 1, 0);

        sr_ptrs.push_back(sr.get());
        owned_srs.push_back(std::move(sr));
    }

    if (sr_ptrs.empty()) {
        // No results at all: return empty blobs.
        auto* blobs = new SearchResultDataBlobs();
        blobs->blobs.resize(1);
        blobs->costs.resize(1);
        *out_result = blobs;
        return;
    }

    // Step 3: hand to ReduceHelper for PK filling, dedup, output-field fill,
    // and proto serialisation — identical to ReduceSearchResultsAndFillData.
    auto* plan = static_cast<milvus::query::Plan*>(plan_);
    int64_t slice_nq = nq_;
    int64_t slice_topK = topK_;
    auto reduce_helper = std::make_shared<ReduceHelper>(
        sr_ptrs, plan, &slice_nq, &slice_topK, 1, nullptr);
    reduce_helper->Reduce();
    reduce_helper->Marshal();
    *out_result = reduce_helper->GetSearchResultDataBlobs();
}

}  // namespace milvus::segcore
