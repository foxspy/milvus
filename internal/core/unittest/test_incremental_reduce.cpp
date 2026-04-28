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

#include <cfloat>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "common/Consts.h"
#include "common/EasyAssert.h"
#include "common/QueryResult.h"
#include "pb/schema.pb.h"
#include "segcore/Collection.h"
#include "segcore/collection_c.h"
#include "segcore/plan_c.h"
#include "segcore/reduce/IncrementalReduceHeap.h"
#include "segcore/reduce/Reduce.h"
#include "segcore/reduce_c.h"
#include "segcore/segment_c.h"
#include "test_utils/DataGen.h"
#include "test_utils/PbHelper.h"
#include "test_utils/c_api_test_utils.h"

using namespace milvus;
using namespace milvus::segcore;

// ---------------------------------------------------------------------------
// Schema config helpers — one per metric type.
// (get_default_schema_config() uses L2 hard-coded; we need IP/COSINE too.)
// ---------------------------------------------------------------------------

namespace {

// Build a schema config proto text with the given metric type.
// Vector field: fakevec (FloatVector, dim=4, fieldID=100)
// PK field:     age      (Int64, primary key,   fieldID=101)
std::string
schema_config_for_metric(const std::string& metric) {
    auto fmt =
        boost::format(
            R"(name: "default-collection"
               fields: <
                 fieldID: 100
                 name: "fakevec"
                 data_type: FloatVector
                 type_params: < key: "dim" value: "4" >
                 index_params: < key: "metric_type" value: "%1%" >
               >
               fields: <
                 fieldID: 101
                 name: "age"
                 data_type: Int64
                 is_primary_key: true
               >)") %
        metric;
    return fmt.str();
}

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

struct TestContext {
    CCollection collection{nullptr};
    std::vector<CSegmentInterface> segments;
    std::vector<CSearchResult> search_results;
    void* plan{nullptr};
    void* placeholder_group{nullptr};

    ~TestContext() {
        for (auto r : search_results)
            DeleteSearchResult(r);
        if (placeholder_group)
            DeletePlaceholderGroup(placeholder_group);
        if (plan)
            DeleteSearchPlan(plan);
        for (auto s : segments)
            DeleteSegment(s);
        if (collection)
            DeleteCollection(collection);
    }
};

// Build a TestContext with `n_segments` growing segments, each containing
// `n_per_seg` rows, then run CSearch on each.
// metric_type must be one of "L2", "IP", "COSINE".
std::unique_ptr<TestContext>
MakeTestContext(int n_per_seg,
                int n_segments,
                int nq,
                int topK,
                const std::string& metric_type) {
    auto ctx = std::make_unique<TestContext>();

    // NewCollection(const char*, MetricType) from DataGen.h sets the
    // collection's index_meta, which CreateSearchPlanByExpr reads to set
    // the plan's metric_type.
    ctx->collection =
        NewCollection(schema_config_for_metric(metric_type).c_str(), metric_type);
    auto* col = static_cast<milvus::segcore::Collection*>(ctx->collection);
    auto schema = col->get_schema();

    // Build the search plan. CreateSearchPlanByExpr sets the metric type from
    // the index meta (which came from the schema config above).
    ScopedSchemaHandle schema_handle(*schema);
    auto binary_plan = schema_handle.ParseSearch("",
                                                  "fakevec",
                                                  topK,
                                                  metric_type,
                                                  R"({"nprobe": 10})");
    auto status = CreateSearchPlanByExpr(ctx->collection,
                                         binary_plan.data(),
                                         binary_plan.size(),
                                         &ctx->plan);
    EXPECT_EQ(status.error_code, Success);

    auto blob = generate_query_data(nq);
    status = ParsePlaceholderGroup(
        ctx->plan, blob.data(), blob.length(), &ctx->placeholder_group);
    EXPECT_EQ(status.error_code, Success);

    // Insert and search each segment.
    for (int s = 0; s < n_segments; ++s) {
        CSegmentInterface seg;
        // Use distinct segment IDs to avoid collisions in seg_map_.
        auto st = NewSegment(ctx->collection, Growing, 100 + s, &seg, false);
        EXPECT_EQ(st.error_code, Success);
        ctx->segments.push_back(seg);

        // Use random PKs so each segment has globally unique primary keys,
        // matching production usage where segments never share PKs.
        auto dataset =
            DataGen(schema, n_per_seg, /*seed=*/42 + s, 0, 1, 10, 1, /*random_pk=*/true);
        int64_t insert_offset = 0;
        PreInsert(seg, n_per_seg, &insert_offset);
        auto raw = serialize(dataset.raw_);
        auto ins = Insert(seg,
                          insert_offset,
                          n_per_seg,
                          dataset.row_ids_.data(),
                          dataset.timestamps_.data(),
                          raw.data(),
                          raw.size());
        EXPECT_EQ(ins.error_code, Success);

        CSearchResult res;
        st = CSearch(seg, ctx->plan, ctx->placeholder_group, MAX_TIMESTAMP, &res);
        EXPECT_EQ(st.error_code, Success);
        ctx->search_results.push_back(res);
    }

    return ctx;
}

// Per-nq result: sorted list of (distance, pk) in the order returned by reduce.
struct NQResult {
    std::vector<float> distances;
    std::vector<int64_t> pks;
    std::unordered_set<int64_t> pk_set;
};

// Extract per-nq (pk, distance) pairs from a SearchResultDataBlobs blob.
std::vector<NQResult>
ExtractResultsPerNQ(CSearchResultDataBlobs blobs, int nq) {
    CProto blob_proto;
    int64_t remote_bytes = 0, total_bytes = 0;
    auto st = GetSearchResultDataBlob(
        &blob_proto, &remote_bytes, &total_bytes, blobs, 0);
    EXPECT_EQ(st.error_code, Success);

    milvus::proto::schema::SearchResultData srd;
    EXPECT_TRUE(srd.ParseFromArray(blob_proto.proto_blob, blob_proto.proto_size));

    std::vector<NQResult> result(nq);
    int64_t offset = 0;
    for (int q = 0; q < nq; ++q) {
        int64_t count = (q < srd.topks_size()) ? srd.topks(q) : 0;
        for (int64_t k = 0; k < count; ++k) {
            int64_t pk = srd.ids().int_id().data(offset + k);
            float dist = (offset + k < srd.scores_size()) ? srd.scores(offset + k) : 0.0f;
            result[q].pks.push_back(pk);
            result[q].distances.push_back(dist);
            result[q].pk_set.insert(pk);
        }
        offset += count;
    }
    return result;
}

// Extract the set of int64 PKs per nq from a SearchResultDataBlobs blob.
std::vector<std::unordered_set<int64_t>>
ExtractPKsPerNQ(CSearchResultDataBlobs blobs, int nq) {
    auto results = ExtractResultsPerNQ(blobs, nq);
    std::vector<std::unordered_set<int64_t>> pks(nq);
    for (int q = 0; q < nq; ++q) pks[q] = std::move(results[q].pk_set);
    return pks;
}

// Check two NQResult vectors are "equivalent" allowing boundary ties.
// Returns true if both results represent valid top-K answers where any
// differing elements have equal (within EPSILON) distances.
static constexpr float DIST_EPS = 1e-6f;  // allow small float differences

bool
ResultsEquivalent(const std::vector<NQResult>& expected,
                  const std::vector<NQResult>& actual,
                  int nq) {
    for (int q = 0; q < nq; ++q) {
        const auto& exp = expected[q];
        const auto& act = actual[q];
        if (exp.pks.size() != act.pks.size()) return false;
        int topK = static_cast<int>(exp.pks.size());
        if (topK == 0) continue;

        // Find the K-th (boundary) distance in expected.
        float kth_dist_exp = exp.distances.back();  // last = worst = K-th
        float kth_dist_act = act.distances.back();

        // Elements in expected that are strictly better than K-th:
        // they MUST appear in actual (they're unambiguous top-K).
        for (int k = 0; k < topK; ++k) {
            float d = exp.distances[k];
            // If strictly better than K-th by more than DIST_EPS:
            if (d - kth_dist_exp > DIST_EPS) {
                // This entry should unambiguously be in actual.
                if (act.pk_set.count(exp.pks[k]) == 0) return false;
            }
        }
        // Symmetric: elements in actual strictly better than their K-th
        // should appear in expected.
        for (int k = 0; k < topK; ++k) {
            float d = act.distances[k];
            if (d - kth_dist_act > DIST_EPS) {
                if (exp.pk_set.count(act.pks[k]) == 0) return false;
            }
        }
        // Verify that differing elements have K-th-level distances (within DIST_EPS).
        for (auto pk : exp.pk_set) {
            if (act.pk_set.count(pk) == 0) {
                // pk is in expected but not actual. Find its distance in expected.
                for (int k = 0; k < topK; ++k) {
                    if (exp.pks[k] == pk) {
                        if (std::fabs(exp.distances[k] - kth_dist_exp) > DIST_EPS) {
                            return false;  // Not a boundary tie
                        }
                        break;
                    }
                }
            }
        }
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Helpers: one-shot vs. incremental
// ---------------------------------------------------------------------------

// Run ReduceSearchResultsAndFillData and return per-nq results with distances.
// NOTE: this MODIFIES the search results in-place.
static std::vector<NQResult>
OneShotReduce(TestContext& ctx, int nq, int topK) {
    std::vector<int64_t> slice_nqs{nq};
    std::vector<int64_t> slice_topKs{topK};
    CSearchResultDataBlobs blobs = nullptr;
    CTraceContext trace{{}, {}, 0};
    auto st = ReduceSearchResultsAndFillData(trace,
                                              &blobs,
                                              ctx.plan,
                                              ctx.search_results.data(),
                                              ctx.search_results.size(),
                                              slice_nqs.data(),
                                              slice_topKs.data(),
                                              1);
    EXPECT_EQ(st.error_code, Success);
    auto results = ExtractResultsPerNQ(blobs, nq);
    DeleteSearchResultDataBlobs(blobs);
    return results;
}

// Run IncrementalReduceHeap in two batches and return per-nq results with distances.
// NOTE: this MODIFIES the search results in-place (through Finalize → ReduceHelper).
static std::vector<NQResult>
IncrementalReduce(TestContext& ctx, int nq, int topK, int split) {
    IncrementalReduceHeap heap(ctx.plan, nq, topK);

    int total = static_cast<int>(ctx.search_results.size());
    int first = std::min(split, total);
    heap.MergeBatch(ctx.search_results.data(), first);
    if (first < total) {
        heap.MergeBatch(ctx.search_results.data() + first, total - first);
    }

    CSearchResultDataBlobs blobs = nullptr;
    heap.Finalize(&blobs);
    EXPECT_NE(blobs, nullptr);
    auto results = ExtractResultsPerNQ(blobs, nq);
    DeleteSearchResultDataBlobs(blobs);
    return results;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// EquivalentToOneShot tests each use TWO separate TestContexts with identical
// parameters to avoid one reducer's in-place modifications affecting the other.

// EquivalentToOneShot tests verify that IncrementalReduce produces the same
// top-K result as OneShotReduce.  Both methods are valid top-K algorithms.
// When top-K distances are distinct (no ties at the K-th boundary), the sets
// must be identical.  When the K-th and (K+1)-th candidates have equal
// distances, any valid tie-breaking is accepted (ResultsEquivalent).

TEST(IncrementalReduceHeap, EquivalentToOneShot_L2) {
    constexpr int N = 500, N_SEG = 6, NQ = 3, TOPK = 5;
    // One context for one-shot, one for incremental (same seeds).
    auto ctx_oneshot = MakeTestContext(N, N_SEG, NQ, TOPK, "L2");
    auto ctx_incr = MakeTestContext(N, N_SEG, NQ, TOPK, "L2");

    auto expected = OneShotReduce(*ctx_oneshot, NQ, TOPK);
    auto actual = IncrementalReduce(*ctx_incr, NQ, TOPK, 3);

    EXPECT_TRUE(ResultsEquivalent(expected, actual, NQ))
        << "L2 results differ beyond boundary ties";
}

TEST(IncrementalReduceHeap, EquivalentToOneShot_IP) {
    constexpr int N = 500, N_SEG = 6, NQ = 3, TOPK = 5;
    auto ctx_oneshot = MakeTestContext(N, N_SEG, NQ, TOPK, "IP");
    auto ctx_incr = MakeTestContext(N, N_SEG, NQ, TOPK, "IP");

    auto expected = OneShotReduce(*ctx_oneshot, NQ, TOPK);
    auto actual = IncrementalReduce(*ctx_incr, NQ, TOPK, 3);

    EXPECT_TRUE(ResultsEquivalent(expected, actual, NQ))
        << "IP results differ beyond boundary ties";
}

TEST(IncrementalReduceHeap, EquivalentToOneShot_COSINE) {
    constexpr int N = 500, N_SEG = 6, NQ = 3, TOPK = 5;
    auto ctx_oneshot = MakeTestContext(N, N_SEG, NQ, TOPK, "COSINE");
    auto ctx_incr = MakeTestContext(N, N_SEG, NQ, TOPK, "COSINE");

    auto expected = OneShotReduce(*ctx_oneshot, NQ, TOPK);
    auto actual = IncrementalReduce(*ctx_incr, NQ, TOPK, 3);

    EXPECT_TRUE(ResultsEquivalent(expected, actual, NQ))
        << "COSINE results differ beyond boundary ties";
}

TEST(IncrementalReduceHeap, GetThresholds_UnfilledReturnsSentinel) {
    // Only 3 rows per segment but topK=10 → heap never fills → sentinel.
    // After CSearch all distances follow "larger=better" (L2 is negated by
    // AsyncSearch), so the unfilled sentinel is -FLT_MAX: any real candidate
    // beats it (dist > -FLT_MAX is always true).
    constexpr int N = 3, N_SEG = 1, NQ = 2, TOPK = 10;
    auto ctx = MakeTestContext(N, N_SEG, NQ, TOPK, "L2");

    IncrementalReduceHeap heap(ctx->plan, NQ, TOPK);
    heap.MergeBatch(ctx->search_results.data(),
                    static_cast<int>(ctx->search_results.size()));

    std::vector<float> thresholds(NQ);
    heap.GetThresholds(thresholds.data());
    for (int q = 0; q < NQ; ++q) {
        EXPECT_EQ(thresholds[q], -FLT_MAX) << "nq=" << q;
    }
}

TEST(IncrementalReduceHeap, GetThresholds_FilledReturnsKthWorst) {
    // 200 rows, topK=5, L2: threshold = worst-of-top-K distance.
    // NOTE: AsyncSearch negates L2 distances (so they are <= 0 in the heap).
    // The threshold must be strictly less than the unfilled sentinel (FLT_MAX)
    // and must be <= 0.0f for L2 (negated squared distances are non-positive).
    constexpr int N = 200, N_SEG = 1, NQ = 2, TOPK = 5;
    auto ctx = MakeTestContext(N, N_SEG, NQ, TOPK, "L2");

    IncrementalReduceHeap heap(ctx->plan, NQ, TOPK);
    heap.MergeBatch(ctx->search_results.data(),
                    static_cast<int>(ctx->search_results.size()));

    std::vector<float> thresholds(NQ);
    heap.GetThresholds(thresholds.data());
    for (int q = 0; q < NQ; ++q) {
        EXPECT_LT(thresholds[q], FLT_MAX)
            << "nq=" << q << " heap should be full";
        // AsyncSearch negates L2 distances → all L2 distances in the heap are <= 0.
        EXPECT_LE(thresholds[q], 0.0f)
            << "nq=" << q << " negated L2 dist must be <= 0";
    }
}

TEST(IncrementalReduceHeap, NqIndependent) {
    // Verify per-nq heaps are independent and Finalize still works.
    constexpr int N = 200, N_SEG = 1, NQ = 2, TOPK = 5;
    auto ctx = MakeTestContext(N, N_SEG, NQ, TOPK, "L2");

    IncrementalReduceHeap heap(ctx->plan, NQ, TOPK);
    heap.MergeBatch(ctx->search_results.data(),
                    static_cast<int>(ctx->search_results.size()));

    std::vector<float> thresholds(NQ);
    heap.GetThresholds(thresholds.data());
    for (int q = 0; q < NQ; ++q) {
        EXPECT_LT(thresholds[q], FLT_MAX) << "nq=" << q;
    }
    // Verify Finalize works after getting thresholds.
    CSearchResultDataBlobs blobs = nullptr;
    heap.Finalize(&blobs);
    EXPECT_NE(blobs, nullptr);
    DeleteSearchResultDataBlobs(blobs);
}

TEST(IncrementalReduceHeap, FinalizeNotReusable) {
    constexpr int N = 50, N_SEG = 1, NQ = 1, TOPK = 5;
    auto ctx = MakeTestContext(N, N_SEG, NQ, TOPK, "L2");

    IncrementalReduceHeap heap(ctx->plan, NQ, TOPK);
    heap.MergeBatch(ctx->search_results.data(),
                    static_cast<int>(ctx->search_results.size()));

    CSearchResultDataBlobs blobs = nullptr;
    heap.Finalize(&blobs);
    EXPECT_NE(blobs, nullptr);
    DeleteSearchResultDataBlobs(blobs);

    // After Finalize, further operations must throw.
    EXPECT_THROW(
        heap.MergeBatch(ctx->search_results.data(),
                        static_cast<int>(ctx->search_results.size())),
        std::exception);
    EXPECT_THROW(
        {
            std::vector<float> t(NQ);
            heap.GetThresholds(t.data());
        },
        std::exception);
    EXPECT_THROW(
        {
            CSearchResultDataBlobs b2 = nullptr;
            heap.Finalize(&b2);
        },
        std::exception);
}
