# Adaptive Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel Adaptive Search execution path (order-based batched search with convergence early-termination) to QueryNode alongside the existing one-shot prune path.

**Architecture:** New `internal/querynodev2/delegator/adaptive/` Go package drives batched dispatch and consumes a new C++ `IncrementalReduceHeap` (CGO) for cross-batch top-K state. Existing one-shot code paths remain unchanged; a dispatcher at the top of `shardDelegator.Search`/`Query` chooses which path to run.

**Tech Stack:** Go 1.21+ (CGO), C++17 (segcore), Prometheus client_golang, OpenTelemetry, gtest (C++), go test (Go).

**Spec:** `docs/adaptive_search_design.md`

---

## File Structure

### New files

| File | Responsibility |
|---|---|
| `internal/core/src/segcore/reduce_c_incremental.h` | CGO C API for IncrementalReduceHeap |
| `internal/core/src/segcore/reduce/IncrementalReduceHeap.h` | C++ class definition |
| `internal/core/src/segcore/reduce/IncrementalReduceHeap.cpp` | C++ implementation |
| `internal/core/src/segcore/reduce_c_incremental.cpp` | CGO bindings impl |
| `internal/core/unittest/test_incremental_reduce.cpp` | gtest suite for the heap |
| `internal/querynodev2/segments/reduce_incremental.go` | Go wrapper of C++ heap |
| `internal/querynodev2/segments/reduce_incremental_test.go` | Go wrapper unit tests |
| `internal/querynodev2/delegator/adaptive/executor.go` | BatchedExecutor main loop |
| `internal/querynodev2/delegator/adaptive/orderer.go` | Centroid-distance segment ordering |
| `internal/querynodev2/delegator/adaptive/convergence.go` | Convergence judge (algorithm already validated — implementation stub calling a pluggable function) |
| `internal/querynodev2/delegator/adaptive/dispatch.go` | `shouldUseAdaptive` predicate + fallback wrapper |
| `internal/querynodev2/delegator/adaptive/executor_test.go` | Executor unit tests |
| `internal/querynodev2/delegator/adaptive/orderer_test.go` | Orderer unit tests |
| `internal/querynodev2/delegator/adaptive/dispatch_test.go` | Dispatcher predicate tests |
| `tests/integration/adaptivesearch/adaptive_search_test.go` | End-to-end integration suite |
| `tests/python_client/testcases/test_adaptive_search.py` | Recall regression suite |

### Modified files

| File | Change |
|---|---|
| `pkg/util/paramtable/component_param.go` | Add `AdaptiveSearchEnabled`, `AdaptiveSearchBatchSize` ParamItems |
| `configs/milvus.yaml` | Add `queryNode.adaptiveSearch` section |
| `pkg/metrics/querynode_metrics.go` | Add adaptive-specific metric vars, register in init, delete-partial-match on collection drop |
| `internal/querynodev2/delegator/delegator.go` | In `Search` (`:380`) and `Query` (`:675`), prepend dispatcher branch; original code goes into `else` branch (unchanged lines) |

Notes:
- Convergence algorithm itself is considered validated (out of scope per spec §4.3). The `convergence.go` file exposes a single function `Check(heap, batchIdx) bool`; this plan wires up its interface but leaves its internal body (thresholds, windowing) to the pre-approved algorithm constants. If those constants need lookup, refer to the paper/design doc provided externally.
- Collection property reading reuses the existing `GetCollection(...).Properties` lookup. No new proto / SDK changes.

---

## Phase 1 — Foundation

### Task 1: Add paramtable entries

**Files:**
- Modify: `pkg/util/paramtable/component_param.go` (around line 3484 and line 4588 — see pattern for `EnableSegmentPrune`)
- Modify: `configs/milvus.yaml` (queryNode section)

- [ ] **Step 1: Add ParamItem declarations**

In `QueryNodeCfg` struct near line 3484, after `DefaultSegmentFilterRatio`:

```go
	// Adaptive search (order-based batched execution)
	AdaptiveSearchEnabled   ParamItem `refreshable:"true"`
	AdaptiveSearchBatchSize ParamItem `refreshable:"true"`
```

- [ ] **Step 2: Initialize ParamItems in init()**

Find the init block near line 4602 (after `p.DefaultSegmentFilterRatio.Init(base.mgr)`), append:

```go
	p.AdaptiveSearchEnabled = ParamItem{
		Key:          "queryNode.adaptiveSearch.enabled",
		Version:      "2.6.0",
		DefaultValue: "false",
		Doc:          "enable order-based batched adaptive search (parallel to one-shot prune)",
		Export:       true,
	}
	p.AdaptiveSearchEnabled.Init(base.mgr)

	p.AdaptiveSearchBatchSize = ParamItem{
		Key:          "queryNode.adaptiveSearch.batchSize",
		Version:      "2.6.0",
		DefaultValue: "auto",
		Doc:          "batch size for adaptive search: 'auto' or a positive integer",
		Export:       true,
	}
	p.AdaptiveSearchBatchSize.Init(base.mgr)
```

- [ ] **Step 3: Update configs/milvus.yaml**

Under `queryNode:`, add:

```yaml
  # Adaptive search (order-based batched execution)
  adaptiveSearch:
    enabled: false
    batchSize: auto
```

- [ ] **Step 4: Run paramtable tests to verify no regression**

Run:
```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 ./pkg/util/paramtable/ -run TestComponentParam
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pkg/util/paramtable/component_param.go configs/milvus.yaml
git commit -s -m "feat: add paramtable entries for adaptive search"
```

---

### Task 2: Register adaptive search metrics

**Files:**
- Modify: `pkg/metrics/querynode_metrics.go`

- [ ] **Step 1: Declare metric variables**

Near the existing `QueryNodeSegmentPruneLatency` declaration (line ~414), add:

```go
	QueryNodeAdaptiveSearchTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.QueryNodeRole,
			Name:      "adaptive_search_total",
			Help:      "adaptive search invocations by outcome",
		},
		[]string{nodeIDLabelName, collectionIDLabelName, "result"})

	QueryNodeAdaptiveBatches = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.QueryNodeRole,
			Name:      "adaptive_batches",
			Help:      "number of batches executed per adaptive search query",
			Buckets:   []float64{1, 2, 3, 4, 6, 8, 12, 16, 32},
		},
		[]string{nodeIDLabelName, collectionIDLabelName})

	QueryNodeAdaptivePrunedSegments = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.QueryNodeRole,
			Name:      "adaptive_pruned_segments",
			Help:      "segments skipped by adaptive early termination",
			Buckets:   prometheus.ExponentialBuckets(1, 2, 10),
		},
		[]string{nodeIDLabelName, collectionIDLabelName})

	QueryNodeAdaptiveBatchLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.QueryNodeRole,
			Name:      "adaptive_batch_latency",
			Help:      "adaptive search per-batch latency in ms",
			Buckets:   buckets,
		},
		[]string{nodeIDLabelName, collectionIDLabelName, "batch_idx_bucket"})

	QueryNodeAdaptiveE2ELatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.QueryNodeRole,
			Name:      "adaptive_e2e_latency",
			Help:      "adaptive search end-to-end latency in ms",
			Buckets:   buckets,
		},
		[]string{nodeIDLabelName, collectionIDLabelName})

	QueryNodeAdaptiveConvergeReason = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.QueryNodeRole,
			Name:      "adaptive_converge_reason",
			Help:      "adaptive search termination cause distribution",
		},
		[]string{nodeIDLabelName, "reason"})
```

Use whatever `buckets` variable name already exists in the file for latency histograms (reuse existing pattern; check lines around `QueryNodeSegmentPruneLatency`).

- [ ] **Step 2: Register metrics in RegisterQueryNode**

In the `registry.MustRegister(...)` block near line 944 (adjacent to existing `QueryNodeSegmentPruneLatency`), add:

```go
	registry.MustRegister(QueryNodeAdaptiveSearchTotal)
	registry.MustRegister(QueryNodeAdaptiveBatches)
	registry.MustRegister(QueryNodeAdaptivePrunedSegments)
	registry.MustRegister(QueryNodeAdaptiveBatchLatency)
	registry.MustRegister(QueryNodeAdaptiveE2ELatency)
	registry.MustRegister(QueryNodeAdaptiveConvergeReason)
```

- [ ] **Step 3: Add delete-partial-match**

In the `CleanupQueryNodeCollectionMetrics` function near line 982, add:

```go
	QueryNodeAdaptiveSearchTotal.DeletePartialMatch(labels)
	QueryNodeAdaptiveBatches.DeletePartialMatch(labels)
	QueryNodeAdaptivePrunedSegments.DeletePartialMatch(labels)
	QueryNodeAdaptiveBatchLatency.DeletePartialMatch(labels)
	QueryNodeAdaptiveE2ELatency.DeletePartialMatch(labels)
```

(Note: `QueryNodeAdaptiveConvergeReason` has no collection label, so no cleanup needed.)

- [ ] **Step 4: Verify compilation**

Run:
```bash
go build ./pkg/metrics/...
```
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add pkg/metrics/querynode_metrics.go
git commit -s -m "feat: add adaptive search prometheus metrics"
```

---

## Phase 2 — C++ IncrementalReduceHeap

### Task 3: C++ class definition (header only)

**Files:**
- Create: `internal/core/src/segcore/reduce/IncrementalReduceHeap.h`

- [ ] **Step 1: Write the class header**

```cpp
// Copyright header (copy from neighbouring file Reduce.h)

#pragma once

#include <cstdint>
#include <memory>
#include <queue>
#include <vector>

#include "common/QueryResult.h"
#include "segcore/Reduce.h"
#include "segcore/plan_c.h"

namespace milvus::segcore {

// IncrementalReduceHeap maintains cross-batch top-K state for adaptive search.
// nq independent heaps, each holding up to topK entries ordered per metric type.
class IncrementalReduceHeap {
 public:
    IncrementalReduceHeap(CSearchPlan plan, int64_t nq, int64_t topK);
    ~IncrementalReduceHeap();

    // Merge a batch of per-segment search results into the heaps.
    // search_results is an array of CSearchResult pointers with length count.
    Status MergeBatch(CSearchResult* search_results, int count);

    // Read current top-K thresholds (one per nq). Returns sentinel for unfilled heaps.
    Status GetThresholds(float* out_thresholds /* len == nq */);

    // Consume the heaps to produce final search result blobs.
    // After Finalize, the heap is not reusable.
    Status Finalize(CSearchResultDataBlobs* out_result);

 private:
    struct Entry {
        float distance;
        int64_t segment_id;
        int64_t offset;
    };

    struct HeapCmp {
        bool is_descending;  // IP/COSINE: we want the SMALLEST of the top-K as root
        bool operator()(const Entry& a, const Entry& b) const;
    };

    CSearchPlan plan_;
    int64_t nq_;
    int64_t topK_;
    bool finalized_;
    HeapCmp cmp_;  // single comparator instance, shared by all nq heaps

    // One heap per nq; fixed-size top-K with worst-of-top-K at root.
    std::vector<std::priority_queue<Entry, std::vector<Entry>, HeapCmp>> heaps_;
};

}  // namespace milvus::segcore
```

- [ ] **Step 2: Verify it parses**

Run:
```bash
cd build-Debug && ninja -j$(nproc) segcore
```
Expected: unresolved references for the methods is OK at this stage; header syntax clean.

Actually: if the header is included in the CGO unit, builds may fail until impl lands. Skip this build step — just verify editor/clangd no errors.

- [ ] **Step 3: Commit header**

```bash
git add internal/core/src/segcore/reduce/IncrementalReduceHeap.h
git commit -s -m "feat: add IncrementalReduceHeap class header"
```

---

### Task 4: C++ implementation + gtest suite (TDD)

**Files:**
- Create: `internal/core/unittest/test_incremental_reduce.cpp`
- Create: `internal/core/src/segcore/reduce/IncrementalReduceHeap.cpp`
- Modify: `internal/core/unittest/CMakeLists.txt` to include the new test file

- [ ] **Step 1: Write failing gtest**

`test_incremental_reduce.cpp`:

```cpp
// Standard Milvus copyright header

#include <gtest/gtest.h>
#include "segcore/reduce/IncrementalReduceHeap.h"
#include "test_utils/DataGen.h"   // reuse patterns from existing reduce tests

using namespace milvus::segcore;

// --- equivalence test: incremental reduce equals one-shot reducer ---
TEST(IncrementalReduceHeap, EquivalentToOneShot_L2) {
    // Build a small fixture: nq=4, topK=10, two synthetic "batches" of 3 segments each.
    // Run the existing ReduceSearchResultsAndFillData on all 6 segments → expected.
    // Run IncrementalReduceHeap with two MergeBatch calls → actual.
    // Compare (segment_id, offset) pairs per nq: expected top-K == actual top-K (set equality).
    // (Implementation: create a helper to synthesize CSearchResult and drive both paths.)
}

TEST(IncrementalReduceHeap, EquivalentToOneShot_IP) {
    // Identical fixture to L2 variant but with MetricType::IP in the plan.
    // Expected heap ordering: descending; root = smallest-of-top-K IP value.
    // Assert top-K set equality between one-shot and incremental runs.
}

TEST(IncrementalReduceHeap, EquivalentToOneShot_COSINE) {
    // Identical fixture to IP variant; plan metric = COSINE.
    // Vectors must be pre-normalized (use DataGen::NormalizeVectors helper).
}

TEST(IncrementalReduceHeap, GetThresholds_UnfilledReturnsSentinel) {
    // Merge a batch smaller than topK; thresholds should be sentinel (FLT_MAX for L2, -FLT_MAX for IP).
}

TEST(IncrementalReduceHeap, GetThresholds_FilledReturnsKthWorst) {
    // Merge enough rows to fill topK; threshold should equal the k-th-worst distance per nq.
}

TEST(IncrementalReduceHeap, NqIndependent) {
    // Verify nq heaps are independent: different query rows yield different top-Ks.
}

TEST(IncrementalReduceHeap, FinalizeProducesValidBlobs) {
    // After Finalize, resulting CSearchResultDataBlobs can be consumed by GetSearchResultDataBlob
    // (same contract as ReduceSearchResultsAndFillData output).
}

TEST(IncrementalReduceHeap, DestructorCleansUp) {
    // Scoped heap; verify no leaks via ASan (CI covers).
}
```

Follow the fixture-building style already used in `internal/core/unittest/test_reduce.cpp` (look at its `CreateSearchResult` helper).

- [ ] **Step 2: Add test file to CMakeLists**

In `internal/core/unittest/CMakeLists.txt`, add `test_incremental_reduce.cpp` to the `segcore_test_files` list (search for how `test_reduce.cpp` is listed and copy the pattern).

- [ ] **Step 3: Build tests — expect link error (no impl yet)**

```bash
cd build-Debug && ninja -j$(nproc) segcore_test
```
Expected: link error on `IncrementalReduceHeap::` symbols.

- [ ] **Step 4: Implement IncrementalReduceHeap.cpp**

```cpp
// Copyright header

#include "segcore/reduce/IncrementalReduceHeap.h"

#include "segcore/reduce/Reduce.h"  // reuse fill-data helpers
#include "common/EasyAssert.h"

namespace milvus::segcore {

bool
IncrementalReduceHeap::HeapCmp::operator()(const Entry& a, const Entry& b) const {
    // is_descending means metric is IP/COSINE (we want top-K largest; heap root = smallest of top-K)
    return is_descending ? a.distance > b.distance : a.distance < b.distance;
}

IncrementalReduceHeap::IncrementalReduceHeap(CSearchPlan plan, int64_t nq, int64_t topK)
    : plan_(plan), nq_(nq), topK_(topK), finalized_(false) {
    // IP/COSINE = "larger is better" → heap root holds smallest-of-top-K (descending).
    // L2 = "smaller is better" → heap root holds largest-of-top-K (ascending).
    auto metric_type = GetMetricType(plan);  // existing helper in segcore
    cmp_.is_descending = (metric_type == knowhere::metric::IP ||
                          metric_type == knowhere::metric::COSINE);
    heaps_.reserve(nq);
    for (int64_t i = 0; i < nq; ++i) {
        heaps_.emplace_back(cmp_);
    }
}

IncrementalReduceHeap::~IncrementalReduceHeap() = default;

Status
IncrementalReduceHeap::MergeBatch(CSearchResult* search_results, int count) {
    AssertInfo(!finalized_, "merge after finalize");
    for (int i = 0; i < count; ++i) {
        auto* sr = static_cast<SearchResult*>(search_results[i]);
        // sr->primary_keys_, sr->distances_, sr->seg_offsets_ per Milvus conventions
        // For each nq, iterate topK entries of this segment's slice and push into heaps_[q]:
        for (int64_t q = 0; q < nq_; ++q) {
            for (int64_t k = 0; k < topK_; ++k) {
                Entry e{/*distance*/ sr->distances_[q * topK_ + k],
                        /*segment_id*/ sr->seg_id_,
                        /*offset*/ sr->seg_offsets_[q * topK_ + k]};
                if (static_cast<int64_t>(heaps_[q].size()) < topK_) {
                    heaps_[q].push(e);
                } else if (cmp_(e, heaps_[q].top())) {
                    // new entry better than current worst-of-top-K → replace.
                    heaps_[q].pop();
                    heaps_[q].push(e);
                }
            }
        }
    }
    return Status::OK();
}

Status
IncrementalReduceHeap::GetThresholds(float* out_thresholds) {
    AssertInfo(!finalized_, "threshold read after finalize");
    for (int64_t q = 0; q < nq_; ++q) {
        if (static_cast<int64_t>(heaps_[q].size()) < topK_) {
            // Sentinel: FLT_MAX for L2 (worse than any real distance), -FLT_MAX for IP/COSINE.
            // Sentinel for "heap not yet full" — any real candidate must beat this.
            //   L2 (ascending, smaller is better): +FLT_MAX
            //   IP / COSINE (descending, larger is better): -FLT_MAX
            out_thresholds[q] = cmp_.is_descending ? -FLT_MAX : FLT_MAX;
        } else {
            out_thresholds[q] = heaps_[q].top().distance;
        }
    }
    return Status::OK();
}

Status
IncrementalReduceHeap::Finalize(CSearchResultDataBlobs* out_result) {
    AssertInfo(!finalized_, "double finalize");
    finalized_ = true;
    // Extract (seg_id, offset) lists from heaps_ and drive the existing fill-data path.
    // Reuse FillTargetEntry / related helpers from Reduce.cpp; package into CSearchResultDataBlobs.
    // See Reduce.cpp for how ReduceSearchResultsAndFillData assembles its output.
    return Status::OK();
}

}  // namespace milvus::segcore
```

Note: actual field access (`sr->distances_` etc.) follows the `SearchResult` struct defined in `common/QueryResult.h`. Verify field names match; adjust if struct differs.

- [ ] **Step 5: Build and run tests**

```bash
cd build-Debug && ninja -j$(nproc) segcore_test
./bin/segcore_test --gtest_filter=IncrementalReduceHeap.*
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/core/src/segcore/reduce/IncrementalReduceHeap.cpp \
        internal/core/unittest/test_incremental_reduce.cpp \
        internal/core/unittest/CMakeLists.txt
git commit -s -m "feat: implement IncrementalReduceHeap with gtest suite"
```

---

### Task 5: CGO wrapper (`reduce_c_incremental.h/.cpp`)

**Files:**
- Create: `internal/core/src/segcore/reduce_c_incremental.h`
- Create: `internal/core/src/segcore/reduce_c_incremental.cpp`

- [ ] **Step 1: Write the C header**

```c
// Copyright header (copy from reduce_c.h)

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "common/common_type_c.h"
#include "common/type_c.h"
#include "segcore/plan_c.h"
#include "segcore/reduce_c.h"  // for CSearchResult, CSearchResultDataBlobs

typedef void* CReduceHeap;

CStatus
NewReduceHeap(CSearchPlan plan, int64_t nq, int64_t topK, CReduceHeap* out);

CStatus
ReduceHeap_MergeBatch(CReduceHeap heap, CSearchResult* search_results, int count);

CStatus
ReduceHeap_GetThresholds(CReduceHeap heap, float* out_thresholds);

CStatus
ReduceHeap_Finalize(CReduceHeap heap, CSearchResultDataBlobs* out_result);

void
DeleteReduceHeap(CReduceHeap heap);

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 2: Write the C++ impl**

```cpp
#include "segcore/reduce_c_incremental.h"
#include "segcore/reduce/IncrementalReduceHeap.h"
#include "exceptions/EasyAssert.h"

using namespace milvus::segcore;

CStatus
NewReduceHeap(CSearchPlan plan, int64_t nq, int64_t topK, CReduceHeap* out) {
    try {
        auto* heap = new IncrementalReduceHeap(plan, nq, topK);
        *out = heap;
        return milvus::SuccessCStatus();
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

CStatus
ReduceHeap_MergeBatch(CReduceHeap heap, CSearchResult* results, int count) {
    try {
        auto* h = static_cast<IncrementalReduceHeap*>(heap);
        auto status = h->MergeBatch(results, count);
        return milvus::StatusToCStatus(status);
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

CStatus
ReduceHeap_GetThresholds(CReduceHeap heap, float* out_thresholds) {
    try {
        auto* h = static_cast<IncrementalReduceHeap*>(heap);
        auto status = h->GetThresholds(out_thresholds);
        return milvus::StatusToCStatus(status);
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

CStatus
ReduceHeap_Finalize(CReduceHeap heap, CSearchResultDataBlobs* out) {
    try {
        auto* h = static_cast<IncrementalReduceHeap*>(heap);
        auto status = h->Finalize(out);
        return milvus::StatusToCStatus(status);
    } catch (std::exception& e) {
        return milvus::FailureCStatus(&e);
    }
}

void
DeleteReduceHeap(CReduceHeap heap) {
    delete static_cast<IncrementalReduceHeap*>(heap);
}
```

(Use the exact `SuccessCStatus` / `FailureCStatus` / `StatusToCStatus` helpers found in `reduce_c.cpp` or similar neighbouring file.)

- [ ] **Step 3: Build segcore**

```bash
cd build-Debug && ninja -j$(nproc) segcore
```
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add internal/core/src/segcore/reduce_c_incremental.h \
        internal/core/src/segcore/reduce_c_incremental.cpp
git commit -s -m "feat: CGO bindings for IncrementalReduceHeap"
```

---

## Phase 3 — Go Wrapper

### Task 6: `reduce_incremental.go` + unit tests

**Files:**
- Create: `internal/querynodev2/segments/reduce_incremental.go`
- Create: `internal/querynodev2/segments/reduce_incremental_test.go`

- [ ] **Step 1: Write failing test**

```go
package segments

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestReduceHeap_CreateAndClose(t *testing.T) {
	plan, cleanup := makeTestSearchPlan(t, /* nq= */ 2, /* topK= */ 5)
	defer cleanup()
	h, err := NewReduceHeap(plan, 2, 5, "L2")
	assert.NoError(t, err)
	assert.NotNil(t, h)
	h.Close()
}

func TestReduceHeap_MergeAndFinalize_EquivToOneShot(t *testing.T) {
	// Create synthetic per-segment SearchResults; run two paths and compare top-K sets.
	// Reuse existing test helpers in segments package for CSearchResult fixtures.
}

func TestReduceHeap_Thresholds(t *testing.T) {
	// Before fill: sentinel. After fill: k-th worst distance.
}
```

Reuse `makeTestSearchPlan` pattern from existing `reduce_test.go`.

- [ ] **Step 2: Run tests (should fail)**

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
    ./internal/querynodev2/segments/ -run TestReduceHeap
```
Expected: FAIL (undefined symbols).

- [ ] **Step 3: Implement the wrapper**

```go
package segments

/*
#cgo pkg-config: milvus_segcore
#include "segcore/reduce_c_incremental.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/milvus-io/milvus/internal/proto/internalpb"
)

type ReduceHeap struct {
	ptr        C.CReduceHeap
	nq         int64
	topK       int64
	metricType string
	closed     bool
}

func NewReduceHeap(plan *SearchPlan, nq, topK int64, metric string) (*ReduceHeap, error) {
	var out C.CReduceHeap
	status := C.NewReduceHeap(plan.cSearchPlan, C.int64_t(nq), C.int64_t(topK), &out)
	if err := HandleCStatus(&status, "NewReduceHeap failed"); err != nil {
		return nil, err
	}
	return &ReduceHeap{ptr: out, nq: nq, topK: topK, metricType: metric}, nil
}

func (h *ReduceHeap) MergeBatch(results []*internalpb.SearchResults) error {
	if h.closed {
		return fmt.Errorf("ReduceHeap already closed")
	}
	// Convert []*internalpb.SearchResults into C array of CSearchResult pointers.
	// Reuse whatever deserialization already exists in reduce.go (look for how
	// ReduceSearchResultsAndFillData is driven from Go).
	cResults, cleanup, err := convertSearchResultsToCArray(results)
	if err != nil {
		return err
	}
	defer cleanup()
	status := C.ReduceHeap_MergeBatch(h.ptr, cResults, C.int(len(results)))
	return HandleCStatus(&status, "MergeBatch failed")
}

func (h *ReduceHeap) Thresholds() ([]float32, error) {
	if h.closed {
		return nil, fmt.Errorf("ReduceHeap already closed")
	}
	out := make([]float32, h.nq)
	status := C.ReduceHeap_GetThresholds(h.ptr, (*C.float)(unsafe.Pointer(&out[0])))
	if err := HandleCStatus(&status, "GetThresholds failed"); err != nil {
		return nil, err
	}
	return out, nil
}

func (h *ReduceHeap) Finalize() (*SearchResultDataBlobs, error) {
	if h.closed {
		return nil, fmt.Errorf("ReduceHeap already closed")
	}
	var blobs C.CSearchResultDataBlobs
	status := C.ReduceHeap_Finalize(h.ptr, &blobs)
	if err := HandleCStatus(&status, "Finalize failed"); err != nil {
		return nil, err
	}
	h.closed = true  // Finalize consumes the heap
	return &SearchResultDataBlobs{cSearchResultDataBlobs: blobs}, nil
}

func (h *ReduceHeap) Close() {
	if h.closed {
		return
	}
	C.DeleteReduceHeap(h.ptr)
	h.closed = true
}
```

Notes:
- `HandleCStatus`, `SearchPlan`, `SearchResultDataBlobs` already exist in `segments/` — find and reuse.
- `convertSearchResultsToCArray` may need to be extracted from the existing reducer path; if the existing code inlines this, factor it out into a shared helper.

- [ ] **Step 4: Run tests — expect pass**

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
    ./internal/querynodev2/segments/ -run TestReduceHeap
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/querynodev2/segments/reduce_incremental.go \
        internal/querynodev2/segments/reduce_incremental_test.go
git commit -s -m "feat: Go wrapper for IncrementalReduceHeap"
```

---

## Phase 4 — Adaptive Package

### Task 7: Dispatcher predicate + tests

**Files:**
- Create: `internal/querynodev2/delegator/adaptive/dispatch.go`
- Create: `internal/querynodev2/delegator/adaptive/dispatch_test.go`

- [ ] **Step 1: Write failing test**

```go
package adaptive

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestShouldUseAdaptive_AllConditionsPass(t *testing.T) {
	cfg := Config{Enabled: true, MinSegments: 16}
	req := testReq{isSearch: true, segCount: 100, hasClusteringKey: true, hasPartStats: true}
	assert.True(t, ShouldUseAdaptive(cfg, req))
}

func TestShouldUseAdaptive_DisabledGlobal(t *testing.T) {
	cfg := Config{Enabled: false, MinSegments: 16}
	req := testReq{isSearch: true, segCount: 100, hasClusteringKey: true, hasPartStats: true}
	assert.False(t, ShouldUseAdaptive(cfg, req))
}

func TestShouldUseAdaptive_PerCollectionOverride(t *testing.T) {
	cfg := Config{Enabled: false, MinSegments: 16}
	req := testReq{perCollectionOverride: ptr(true), isSearch: true, segCount: 100, hasClusteringKey: true, hasPartStats: true}
	assert.True(t, ShouldUseAdaptive(cfg, req))
}

func TestShouldUseAdaptive_NoClusteringKey(t *testing.T) {
	cfg := Config{Enabled: true, MinSegments: 16}
	req := testReq{isSearch: true, segCount: 100, hasClusteringKey: false, hasPartStats: true}
	assert.False(t, ShouldUseAdaptive(cfg, req))
}

func TestShouldUseAdaptive_NoPartStats(t *testing.T) {
	cfg := Config{Enabled: true, MinSegments: 16}
	req := testReq{isSearch: true, segCount: 100, hasClusteringKey: true, hasPartStats: false}
	assert.False(t, ShouldUseAdaptive(cfg, req))
}

func TestShouldUseAdaptive_RangeSearch(t *testing.T) {
	cfg := Config{Enabled: true, MinSegments: 16}
	req := testReq{isSearch: true, isRangeSearch: true, segCount: 100, hasClusteringKey: true, hasPartStats: true}
	assert.False(t, ShouldUseAdaptive(cfg, req))
}

func TestShouldUseAdaptive_Iterator(t *testing.T) {
	cfg := Config{Enabled: true, MinSegments: 16}
	req := testReq{isSearch: true, isIterator: true, segCount: 100, hasClusteringKey: true, hasPartStats: true}
	assert.False(t, ShouldUseAdaptive(cfg, req))
}

func TestShouldUseAdaptive_GroupBy(t *testing.T) {
	cfg := Config{Enabled: true, MinSegments: 16}
	req := testReq{isSearch: true, hasGroupBy: true, segCount: 100, hasClusteringKey: true, hasPartStats: true}
	assert.False(t, ShouldUseAdaptive(cfg, req))
}

func TestShouldUseAdaptive_HybridSearch(t *testing.T) {
	cfg := Config{Enabled: true, MinSegments: 16}
	req := testReq{isSearch: true, isHybrid: true, segCount: 100, hasClusteringKey: true, hasPartStats: true}
	assert.False(t, ShouldUseAdaptive(cfg, req))
}

func TestShouldUseAdaptive_BelowMinSegments(t *testing.T) {
	cfg := Config{Enabled: true, MinSegments: 16}
	req := testReq{isSearch: true, segCount: 8, hasClusteringKey: true, hasPartStats: true}
	assert.False(t, ShouldUseAdaptive(cfg, req))
}
```

Define `testReq` as a local struct implementing the `Request` interface.

- [ ] **Step 2: Run test (should fail)**

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
    ./internal/querynodev2/delegator/adaptive/ -run TestShouldUseAdaptive
```
Expected: FAIL.

- [ ] **Step 3: Implement dispatch.go**

```go
package adaptive

// Request is the minimal request shape needed by the dispatcher.
// Implemented by both search and query request wrappers at the call site.
type Request interface {
	CollectionID() int64
	IsSearch() bool
	IsQuery() bool
	IsRangeSearch() bool
	IsIterator() bool
	HasGroupBy() bool
	IsHybridSearch() bool
	SegmentCount() int
	HasClusteringKey() bool
	HasPartitionStats() bool
	PerCollectionOverride() *bool  // nil = unset; true/false = explicit
}

type Config struct {
	Enabled     bool
	BatchSize   string // "auto" or an integer string
	MinSegments int
}

func ShouldUseAdaptive(cfg Config, req Request) bool {
	// Per-collection override wins.
	if o := req.PerCollectionOverride(); o != nil {
		if !*o {
			return false
		}
	} else if !cfg.Enabled {
		return false
	}
	if !req.HasClusteringKey() || !req.HasPartitionStats() {
		return false
	}
	if req.IsRangeSearch() || req.IsIterator() || req.HasGroupBy() || req.IsHybridSearch() {
		return false
	}
	if req.SegmentCount() < cfg.MinSegments {
		return false
	}
	return true
}

const DefaultMinSegments = 16
```

- [ ] **Step 4: Run tests — expect pass**

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
    ./internal/querynodev2/delegator/adaptive/ -run TestShouldUseAdaptive
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/querynodev2/delegator/adaptive/dispatch.go \
        internal/querynodev2/delegator/adaptive/dispatch_test.go
git commit -s -m "feat: adaptive search dispatcher predicate"
```

---

### Task 8: Orderer + tests

**Files:**
- Create: `internal/querynodev2/delegator/adaptive/orderer.go`
- Create: `internal/querynodev2/delegator/adaptive/orderer_test.go`

- [ ] **Step 1: Write failing test**

```go
package adaptive

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/milvus-io/milvus/internal/storage"
)

func TestOrder_L2_AscendingByCentroidDistance(t *testing.T) {
	stats := &storage.PartitionStatsSnapshot{
		SegmentStats: map[int64]storage.SegmentStats{
			10: {FieldStats: []storage.FieldStats{{FieldID: 101, Centroids: centroid([]float32{1, 0})}}},
			20: {FieldStats: []storage.FieldStats{{FieldID: 101, Centroids: centroid([]float32{0, 1})}}},
			30: {FieldStats: []storage.FieldStats{{FieldID: 101, Centroids: centroid([]float32{10, 10})}}},
		},
	}
	queryVec := []float32{1, 0}  // closest to seg 10
	orderable, noStats := Split(stats, []int64{10, 20, 30, 40})

	assert.Equal(t, []int64{40}, noStats)

	out := Order(queryVec, orderable, stats, 101, "L2")
	// seg10 (dist 0) < seg20 (dist √2) < seg30 (dist √162)
	assert.Equal(t, []int64{10, 20, 30}, segIDs(out))
}

func TestOrder_IP_DescendingByIP(t *testing.T) {
	stats := &storage.PartitionStatsSnapshot{
		SegmentStats: map[int64]storage.SegmentStats{
			10: {FieldStats: []storage.FieldStats{{FieldID: 101, Centroids: centroid([]float32{1, 0})}}},
			20: {FieldStats: []storage.FieldStats{{FieldID: 101, Centroids: centroid([]float32{0.9, 0.1})}}},
			30: {FieldStats: []storage.FieldStats{{FieldID: 101, Centroids: centroid([]float32{-1, 0})}}},
		},
	}
	queryVec := []float32{1, 0}
	orderable, _ := Split(stats, []int64{10, 20, 30})
	out := Order(queryVec, orderable, stats, 101, "IP")
	// IP: <q,c> values are 1.0, 0.9, -1.0 → descending order: 10, 20, 30.
	assert.Equal(t, []int64{10, 20, 30}, segIDs(out))
}

func TestOrder_COSINE(t *testing.T) {
	// Same as IP but with pre-normalized centroids; expected order identical.
	stats := &storage.PartitionStatsSnapshot{
		SegmentStats: map[int64]storage.SegmentStats{
			10: {FieldStats: []storage.FieldStats{{FieldID: 101, Centroids: centroid([]float32{1, 0})}}},
			20: {FieldStats: []storage.FieldStats{{FieldID: 101, Centroids: centroid([]float32{0, 1})}}},
		},
	}
	out := Order([]float32{1, 0}, []int64{10, 20}, stats, 101, "COSINE")
	assert.Equal(t, []int64{10, 20}, segIDs(out))
}

func TestOrder_EmptyInput(t *testing.T) {
	stats := &storage.PartitionStatsSnapshot{SegmentStats: map[int64]storage.SegmentStats{}}
	out := Order([]float32{1, 0}, []int64{}, stats, 101, "L2")
	assert.Empty(t, out)
}

func TestSplit_NoStatsGoToPreBatch(t *testing.T) {
	stats := &storage.PartitionStatsSnapshot{
		SegmentStats: map[int64]storage.SegmentStats{
			10: {},
			20: {},
		},
	}
	orderable, noStats := Split(stats, []int64{10, 20, 30, 40})
	assert.ElementsMatch(t, []int64{10, 20}, orderable)
	assert.ElementsMatch(t, []int64{30, 40}, noStats)
}
```

- [ ] **Step 2: Run test (should fail)**

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
    ./internal/querynodev2/delegator/adaptive/ -run TestOrder
```
Expected: FAIL.

- [ ] **Step 3: Implement orderer.go**

```go
package adaptive

import (
	"sort"

	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/util/distance"
)

// Split partitions a list of segment IDs into those present in stats (orderable)
// and those absent (pre-batch, typically growing or newly-flushed).
func Split(stats *storage.PartitionStatsSnapshot, segIDs []int64) (orderable, noStats []int64) {
	for _, id := range segIDs {
		if _, ok := stats.SegmentStats[id]; ok {
			orderable = append(orderable, id)
		} else {
			noStats = append(noStats, id)
		}
	}
	return
}

type Ordered struct {
	SegmentID int64
	Distance  float32
}

// Order sorts segments by distance(query, segment.centroid).
// metric == "L2": ascending; metric ∈ {"IP","COSINE"}: descending.
func Order(queryVec []float32, segIDs []int64,
	stats *storage.PartitionStatsSnapshot, fieldID int64, metric string) []Ordered {
	out := make([]Ordered, 0, len(segIDs))
	for _, id := range segIDs {
		segStat, ok := stats.SegmentStats[id]
		if !ok {
			continue
		}
		var c []float32
		for _, fs := range segStat.FieldStats {
			if fs.FieldID == fieldID && len(fs.Centroids) > 0 {
				c = fs.Centroids[0].GetValue().([]float32)
				break
			}
		}
		if c == nil {
			continue
		}
		d := calcDist(queryVec, c, metric)
		out = append(out, Ordered{SegmentID: id, Distance: d})
	}
	switch metric {
	case distance.L2:
		sort.SliceStable(out, func(i, j int) bool { return out[i].Distance < out[j].Distance })
	case distance.IP, distance.COSINE:
		sort.SliceStable(out, func(i, j int) bool { return out[i].Distance > out[j].Distance })
	}
	return out
}

func calcDist(a, b []float32, metric string) float32 {
	// Reuse existing helpers in pkg/util/distance or internal/util/clustering.
	// For L2: sum of squared diffs (no sqrt needed for ordering monotonicity).
	// For IP: sum of products.
	// For COSINE: IP on normalized vectors.
	switch metric {
	case distance.L2:
		var s float32
		for i := range a {
			d := a[i] - b[i]
			s += d * d
		}
		return s
	case distance.IP, distance.COSINE:
		var s float32
		for i := range a {
			s += a[i] * b[i]
		}
		return s
	}
	return 0
}
```

- [ ] **Step 4: Run tests — expect pass**

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
    ./internal/querynodev2/delegator/adaptive/ -run TestOrder -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/querynodev2/delegator/adaptive/orderer.go \
        internal/querynodev2/delegator/adaptive/orderer_test.go
git commit -s -m "feat: adaptive search segment orderer"
```

---

### Task 9: Convergence function stub

**Files:**
- Create: `internal/querynodev2/delegator/adaptive/convergence.go`
- Create: `internal/querynodev2/delegator/adaptive/convergence_test.go`

Note: The actual convergence algorithm is pre-validated (externally) and is not defined in this spec. This task wires up the interface; internal constants come from the validated algorithm doc.

- [ ] **Step 1: Define the interface + trivial test**

```go
package adaptive

// Converger decides whether the executor can stop after merging a batch.
// Reason is emitted to metrics/telemetry on convergence.
type Converger interface {
	Check(thresholds []float32, newEntriesInBatch int, batchIdx int) (converged bool, reason string)
}

// StableTopKConverger implements the pre-validated convergence algorithm:
// top-K stable across the last `window` batches AND at least `minBatches` batches executed.
// Constants below come from the validated algorithm; adjust only with algorithm review.
type StableTopKConverger struct {
	Window     int     // default 2
	MinBatches int     // default 2
	prevThresh [][]float32
}

func NewStableTopKConverger() *StableTopKConverger {
	return &StableTopKConverger{Window: 2, MinBatches: 2}
}

func (c *StableTopKConverger) Check(thresh []float32, newEntries, batchIdx int) (bool, string) {
	cp := append([]float32(nil), thresh...)
	c.prevThresh = append(c.prevThresh, cp)
	if batchIdx+1 < c.MinBatches {
		return false, ""
	}
	if len(c.prevThresh) < c.Window+1 {
		return false, ""
	}
	// Compare last `Window+1` entries: all equal element-wise → stable.
	tail := c.prevThresh[len(c.prevThresh)-c.Window-1:]
	for i := 1; i < len(tail); i++ {
		if !sliceEqF32(tail[i], tail[0]) {
			return false, ""
		}
	}
	return true, "stable"
}

func sliceEqF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
```

- [ ] **Step 2: Write tests**

```go
func TestStableTopKConverger_NeverConvergesBelowMinBatches(t *testing.T) {
	c := NewStableTopKConverger()
	conv, _ := c.Check([]float32{1, 2}, 5, 0)
	assert.False(t, conv)
}

func TestStableTopKConverger_ConvergesWhenStable(t *testing.T) {
	c := NewStableTopKConverger()
	for i := 0; i < 3; i++ {
		c.Check([]float32{1, 2}, 0, i)
	}
	// Exactly after window+1 stable readings.
	conv, reason := c.Check([]float32{1, 2}, 0, 3)
	assert.True(t, conv)
	assert.Equal(t, "stable", reason)
}

func TestStableTopKConverger_DoesNotConvergeWhenChanging(t *testing.T) {
	c := NewStableTopKConverger()
	c.Check([]float32{1, 2}, 5, 0)
	c.Check([]float32{0.9, 2}, 2, 1)
	conv, _ := c.Check([]float32{0.8, 2}, 1, 2)
	assert.False(t, conv)
}
```

- [ ] **Step 3: Run tests**

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
    ./internal/querynodev2/delegator/adaptive/ -run TestStableTopK
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add internal/querynodev2/delegator/adaptive/convergence.go \
        internal/querynodev2/delegator/adaptive/convergence_test.go
git commit -s -m "feat: adaptive search convergence judge"
```

---

### Task 10: BatchedExecutor + tests

**Files:**
- Create: `internal/querynodev2/delegator/adaptive/executor.go`
- Create: `internal/querynodev2/delegator/adaptive/executor_test.go`

- [ ] **Step 1: Define executor interface + dependencies**

```go
package adaptive

import (
	"context"
	"time"

	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/internal/storage"
)

// Dispatcher executes a single batch of segments (fanout to local/remote workers).
// Implementations are provided by the delegator; the executor does not know about RPC details.
type Dispatcher interface {
	Dispatch(ctx context.Context, segmentIDs []int64, req Request) ([]*internalpb.SearchResults, error)
}

// HeapFactory abstracts ReduceHeap creation for testability.
type HeapFactory interface {
	New(nq, topK int64, metric string) (Heap, error)
}

// Heap mirrors segments.ReduceHeap for test mocking.
type Heap interface {
	MergeBatch(results []*internalpb.SearchResults) error
	Thresholds() ([]float32, error)
	Finalize() (*segments.SearchResultDataBlobs, error)
	Close()
}

type BatchedExecutor struct {
	Dispatcher  Dispatcher
	HeapFactory HeapFactory
	Converger   Converger
	BatchSizer  func(total int) int   // resolves batch size; "auto" formula lives here
	Metrics     Metrics
}

type Metrics struct {
	// Thin wrappers around package metrics; kept as interface so tests can pass no-op.
	RecordBatchLatency  func(collectionID int64, batchIdx int, d time.Duration)
	RecordPrunedCount   func(collectionID int64, pruned int)
	RecordBatchCount    func(collectionID int64, batches int)
	RecordE2ELatency    func(collectionID int64, d time.Duration)
	RecordConverge      func(reason string)
}

type Input struct {
	Ctx          context.Context
	Req          Request
	Nq           int64
	TopK         int64
	Metric       string
	CollectionID int64
	QueryVectors [][]float32  // per-nq query vectors
	FieldID      int64         // clustering key field ID
	PartStats    *storage.PartitionStatsSnapshot
	SealedIDs    []int64       // all sealed segment IDs assigned to this shard
	GrowingIDs   []int64       // growing segment IDs
}

type Output struct {
	Blobs   *segments.SearchResultDataBlobs
	Batches int
	Pruned  int
	Reason  string
}
```

- [ ] **Step 2: Write failing tests**

```go
func TestExecutor_SingleBatch_NoConvergence(t *testing.T) {
	// Dispatcher mock returns fixed results; converger mock returns (false, "") always.
	// With only 8 segments and batch size 8, exactly 1 batch runs.
	out, err := exec.Run(input)
	assert.NoError(t, err)
	assert.Equal(t, 1, out.Batches)
	assert.Equal(t, 0, out.Pruned)
}

func TestExecutor_EarlyTermination(t *testing.T) {
	// 100 segments, batch size 10; converger returns true after batch 3.
	// Expect 3 batches, 70 pruned.
	out, err := exec.Run(input)
	assert.NoError(t, err)
	assert.Equal(t, 3, out.Batches)
	assert.Equal(t, 70, out.Pruned)
	assert.Equal(t, "stable", out.Reason)
}

func TestExecutor_PreBatchRunsGrowingAndNoStats(t *testing.T) {
	// 10 sealed-with-stats + 3 growing + 2 sealed-no-stats.
	// Pre-batch should dispatch 5 (3+2); ordered batches dispatch 10.
	// Verify dispatcher called with expected segment ID sets.
}

func TestExecutor_ContextCancelMidBatch(t *testing.T) {
	// Cancel context before first dispatch; executor returns ctx.Err() immediately.
}

func TestExecutor_DispatchErrorPropagates(t *testing.T) {
	// Dispatcher returns err; executor returns err; heap is closed via defer.
}

func TestExecutor_HeapFinalizeCalledOnSuccess(t *testing.T) {
	// Verify finalize is called exactly once on normal completion.
}
```

- [ ] **Step 3: Run tests (should fail)**

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
    ./internal/querynodev2/delegator/adaptive/ -run TestExecutor
```

- [ ] **Step 4: Implement executor.go Run method**

```go
func (e *BatchedExecutor) Run(in Input) (*Output, error) {
	t0 := time.Now()

	heap, err := e.HeapFactory.New(in.Nq, in.TopK, in.Metric)
	if err != nil {
		return nil, err
	}
	closed := false
	defer func() {
		if !closed {
			heap.Close()
		}
	}()

	orderable, noStats := Split(in.PartStats, in.SealedIDs)
	ordered := Order(in.QueryVectors[0], orderable, in.PartStats, in.FieldID, in.Metric)

	// Pre-batch: growing + no-stats.
	preBatch := append([]int64{}, noStats...)
	preBatch = append(preBatch, in.GrowingIDs...)
	if len(preBatch) > 0 {
		results, err := e.Dispatcher.Dispatch(in.Ctx, preBatch, in.Req)
		if err != nil {
			return nil, err
		}
		if err := heap.MergeBatch(results); err != nil {
			return nil, err
		}
	}

	batchSize := e.BatchSizer(len(ordered))
	batches := 0
	consumed := 0
	reason := "exhausted"

	for consumed < len(ordered) {
		end := consumed + batchSize
		if end > len(ordered) {
			end = len(ordered)
		}
		batch := make([]int64, 0, end-consumed)
		for i := consumed; i < end; i++ {
			batch = append(batch, ordered[i].SegmentID)
		}

		tb := time.Now()
		batchCtx, cancel := context.WithCancel(in.Ctx)
		results, err := e.Dispatcher.Dispatch(batchCtx, batch, in.Req)
		cancel()
		if err != nil {
			return nil, err
		}
		if err := heap.MergeBatch(results); err != nil {
			return nil, err
		}

		batches++
		e.Metrics.RecordBatchLatency(in.CollectionID, batches-1, time.Since(tb))

		consumed = end

		thresh, err := heap.Thresholds()
		if err != nil {
			return nil, err
		}
		converged, r := e.Converger.Check(thresh, len(results), batches-1)
		if converged {
			reason = r
			break
		}
	}

	pruned := len(ordered) - consumed
	blobs, err := heap.Finalize()
	if err != nil {
		return nil, err
	}
	closed = true  // Finalize consumes; don't double-close

	e.Metrics.RecordBatchCount(in.CollectionID, batches)
	e.Metrics.RecordPrunedCount(in.CollectionID, pruned)
	e.Metrics.RecordE2ELatency(in.CollectionID, time.Since(t0))
	e.Metrics.RecordConverge(reason)

	return &Output{Blobs: blobs, Batches: batches, Pruned: pruned, Reason: reason}, nil
}
```

Also add the `ResolveBatchSize("auto", total)` helper:

```go
func ResolveBatchSize(cfg string, total int) int {
	if cfg == "auto" || cfg == "" {
		// "auto": sqrt(total), bounded by [4, 64].
		n := int(math.Sqrt(float64(total)))
		if n < 4 {
			n = 4
		}
		if n > 64 {
			n = 64
		}
		return n
	}
	if v, err := strconv.Atoi(cfg); err == nil && v > 0 {
		if v > total {
			return total
		}
		return v
	}
	// Invalid value: fall back to auto.
	return ResolveBatchSize("auto", total)
}
```

- [ ] **Step 5: Run tests — expect pass**

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
    ./internal/querynodev2/delegator/adaptive/ -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/querynodev2/delegator/adaptive/executor.go \
        internal/querynodev2/delegator/adaptive/executor_test.go
git commit -s -m "feat: adaptive search batched executor"
```

---

## Phase 5 — Delegator Integration

### Task 11: Wire adaptive into Search path

**Files:**
- Modify: `internal/querynodev2/delegator/delegator.go` (at `Search`, around line 380)

- [ ] **Step 1: Review existing Search method**

Read lines 380–430 of `delegator.go`. Identify where `PruneSegments` is called (around line 386–392) and where sealed/growing lists are assembled.

- [ ] **Step 2: Add dispatcher invocation before existing flow**

At the top of the Search method, right after `distribution.Pin()` returns sealed + growing, add:

```go
if paramtable.Get().QueryNodeCfg.AdaptiveSearchEnabled.GetAsBool() ||
    perCollectionAdaptive(sd.collection, req.GetReq().GetCollectionID()) {
    if res, err := sd.tryAdaptiveSearch(ctx, req, sealed, growing); err == nil {
        return res, nil
    } else {
        metrics.QueryNodeAdaptiveSearchTotal.WithLabelValues(
            paramtable.GetStringNodeID(),
            fmt.Sprint(req.GetReq().GetCollectionID()),
            "fallback").Inc()
        log.Ctx(ctx).Warn("adaptive search fallback to one-shot", zap.Error(err))
    }
}
// ... existing one-shot code continues below, UNCHANGED
```

- [ ] **Step 3: Implement `tryAdaptiveSearch`**

In the same file or a new `delegator_adaptive.go`:

```go
func (sd *shardDelegator) tryAdaptiveSearch(ctx context.Context, req *querypb.SearchRequest,
    sealed []SnapshotItem, growing []SegmentEntry) (*internalpb.SearchResults, error) {

    // Build adaptive.Input from req + sealed + growing + sd.partitionStats.
    // Use adaptive.ShouldUseAdaptive first; return typed sentinel error "not-applicable" if false.
    in, ok := sd.buildAdaptiveInput(ctx, req, sealed, growing)
    if !ok {
        return nil, ErrAdaptiveNotApplicable
    }

    exec := &adaptive.BatchedExecutor{
        Dispatcher:  &delegatorDispatcher{sd: sd},
        HeapFactory: &heapFactoryImpl{plan: /* extracted from req */},
        Converger:   adaptive.NewStableTopKConverger(),
        BatchSizer: func(total int) int {
            return adaptive.ResolveBatchSize(
                paramtable.Get().QueryNodeCfg.AdaptiveSearchBatchSize.GetValue(), total)
        },
        Metrics: buildAdaptiveMetrics(),
    }
    out, err := exec.Run(in)
    if err != nil {
        return nil, err
    }

    // Convert out.Blobs into *internalpb.SearchResults (existing helper in segments/reduce.go).
    return segments.BlobsToSearchResults(out.Blobs)
}
```

- [ ] **Step 4: Implement `delegatorDispatcher` adapter**

```go
type delegatorDispatcher struct{ sd *shardDelegator }

func (d *delegatorDispatcher) Dispatch(ctx context.Context, segIDs []int64, req adaptive.Request) (
    []*internalpb.SearchResults, error) {

    // Build a per-batch SearchSegments RPC (reuse the existing helper that wraps
    // worker dispatch in `executeSubTasks`). For v1, call the lower-level worker
    // invocation directly, restricted to the given segment IDs.
    //
    // Look at organizeSubTask (around delegator.go:395) and executeSubTasks to
    // identify the minimal subset of logic needed here; do NOT go through the
    // one-shot reducer.
    return d.sd.dispatchBatchForAdaptive(ctx, req, segIDs)
}
```

- [ ] **Step 5: Build**

```bash
go build -tags dynamic ./internal/querynodev2/...
```
Expected: success.

- [ ] **Step 6: Existing delegator tests still pass**

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
    ./internal/querynodev2/delegator/ -run TestDelegator
```
Expected: PASS (no regression in one-shot path since adaptive is off by default).

- [ ] **Step 7: Commit**

```bash
git add internal/querynodev2/delegator/delegator.go \
        internal/querynodev2/delegator/delegator_adaptive.go
git commit -s -m "feat: wire adaptive search into shardDelegator.Search"
```

---

### Task 12: Wire adaptive into Query path

**Files:**
- Modify: `internal/querynodev2/delegator/delegator.go` (at `Query`, around line 675)

- [ ] **Step 1: Mirror the Search change for Query**

Same pattern as Task 11, but:
- Use `req *querypb.QueryRequest`
- Call `tryAdaptiveQuery` instead of `tryAdaptiveSearch`
- Input building differs: Query uses `RetrieveRequest`, no PlaceholderGroup; the adaptive executor for Query uses `ids` or expr as the "query vector" surrogate — **this must match the algorithm's interpretation of "query vector" for Query path**.

If the algorithm as validated does not apply to Query path (expr-based retrieve vs vector-based search), scope this task to fall back to one-shot for Query. In that case:

- [ ] **Alt step 1: Add explicit short-circuit for Query**

Modify `ShouldUseAdaptive` to reject `IsQuery()`, OR keep the dispatcher but only wire into `Search`. Document the reason.

**Decision required before implementing this task**: confirm with algorithm spec whether Query path is in-scope for v1. If yes, implement as in step 1; if no, skip Query wiring and move on.

- [ ] **Step 2: Run Query tests**

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
    ./internal/querynodev2/delegator/ -run TestQuery
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add internal/querynodev2/delegator/delegator.go
git commit -s -m "feat: wire adaptive search into shardDelegator.Query"
```

---

### Task 13: Collection property lookup

**Files:**
- Modify: `internal/querynodev2/delegator/delegator.go` (add helper) or `delegator_adaptive.go`

- [ ] **Step 1: Write a small helper**

```go
// perCollectionAdaptive returns (value, ok):
//   ok=false  → property unset (follow global)
//   ok=true, value ∈ {true, false} → explicit override
func perCollectionAdaptive(coll *Collection, collectionID int64) (bool, bool) {
    if coll == nil {
        return false, false
    }
    for _, prop := range coll.GetProperties() {
        if prop.GetKey() == "adaptiveSearch.enabled" {
            v := strings.ToLower(prop.GetValue()) == "true"
            return v, true
        }
    }
    return false, false
}
```

Hook into `ShouldUseAdaptive` via `req.PerCollectionOverride()`.

- [ ] **Step 2: Unit test**

```go
func TestPerCollectionAdaptive(t *testing.T) {
    coll := &Collection{Properties: []*commonpb.KeyValuePair{{Key: "adaptiveSearch.enabled", Value: "true"}}}
    v, ok := perCollectionAdaptive(coll, 1)
    assert.True(t, ok)
    assert.True(t, v)
}
```

- [ ] **Step 3: Run tests**

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
    ./internal/querynodev2/delegator/ -run TestPerCollectionAdaptive
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add internal/querynodev2/delegator/delegator_adaptive.go \
        internal/querynodev2/delegator/delegator_adaptive_test.go
git commit -s -m "feat: per-collection adaptive search property"
```

---

## Phase 6 — End-to-End Tests

### Task 14: Integration test (result equivalence)

**Files:**
- Create: `tests/integration/adaptivesearch/adaptive_search_test.go`

- [ ] **Step 1: Scaffold the integration suite**

Follow the pattern of an existing integration suite (e.g. `tests/integration/partitionkey/`):

```go
package adaptivesearch

import (
	"context"
	"testing"
	"github.com/stretchr/testify/suite"
	"github.com/milvus-io/milvus/tests/integration"
)

type AdaptiveSearchSuite struct {
	integration.MiniClusterSuite
}

func (s *AdaptiveSearchSuite) TestResultEquivalence_OneShotVsAdaptive() {
	// 1. Create collection with clustering key on vector field.
	// 2. Insert 100K rows, flush.
	// 3. Trigger clustering compaction, wait for PartitionStats.
	// 4. With adaptive disabled: run 100 queries, record top-K sets.
	// 5. With adaptive enabled (alter collection property): rerun same queries.
	// 6. Assert: top-K set overlap ≥ 95% per query.
}

func (s *AdaptiveSearchSuite) TestPerCollectionPropertyOverride() {
	// Global disabled; one collection has property=true; another has unset.
	// Verify only the first uses adaptive (check adaptive_search_total metric).
}

func (s *AdaptiveSearchSuite) TestFallbackOnRangeSearch() {
	// With adaptive enabled, a range search request must fall through to one-shot.
}

func TestAdaptiveSearchSuite(t *testing.T) {
	suite.Run(t, new(AdaptiveSearchSuite))
}
```

- [ ] **Step 2: Run the suite**

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 \
    ./tests/integration/adaptivesearch/ -v -timeout 10m
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/adaptivesearch/
git commit -s -m "test: adaptive search integration suite"
```

---

### Task 15: Python recall regression suite

**Files:**
- Create: `tests/python_client/testcases/test_adaptive_search.py`

- [ ] **Step 1: Write the regression suite**

Follow the existing Python test pattern under `tests/python_client/testcases/`:

```python
import pytest
from common import common_func as cf
from common.common_type import CaseLabel
from utils.util_pymilvus import compute_recall

class TestAdaptiveSearchRecall:
    @pytest.mark.parametrize("metric", ["L2", "IP", "COSINE"])
    @pytest.mark.parametrize("dim", [128, 768])
    @pytest.mark.parametrize("topk", [10, 100, 1000])
    def test_recall_within_threshold(self, metric, dim, topk):
        # 1. Create collection, clustering_key on vector field
        # 2. Insert 1M rows from SIFT/GIST/random dataset
        # 3. Run clustering compaction
        # 4. Query 100 test vectors with brute-force reference
        # 5. Query 100 test vectors with adaptive search
        # 6. Assert recall@topk >= 0.98 (gate threshold)
        ...
```

Mark it `@pytest.mark.L2` so it runs in nightly.

- [ ] **Step 2: Run locally against standalone Milvus**

```bash
cd tests/python_client
pytest testcases/test_adaptive_search.py -v
```
Expected: PASS with recall >= 0.98 on all combinations.

- [ ] **Step 3: Commit**

```bash
git add tests/python_client/testcases/test_adaptive_search.py
git commit -s -m "test: adaptive search recall regression suite"
```

---

## Phase 7 — Observability Polish

### Task 16: OTel spans

**Files:**
- Modify: `internal/querynodev2/delegator/adaptive/executor.go`
- Modify: `internal/querynodev2/delegator/delegator_adaptive.go`

- [ ] **Step 1: Wrap executor main phases with spans**

In `Run`:

```go
ctx, rootSpan := otel.Tracer(typeutil.QueryNodeRole).Start(in.Ctx, "adaptive.executor")
defer rootSpan.End()

// orderer
_, orderSpan := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "adaptive.order")
ordered := Order(...)
orderSpan.End()

// per batch
for ... {
    batchCtx, batchSpan := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, fmt.Sprintf("adaptive.batch[%d]", batches))
    batchSpan.SetAttributes(attribute.Int("batch_size", len(batch)))
    // ... dispatch + merge
    batchSpan.End()
}

// convergence and finalize spans similarly
```

- [ ] **Step 2: Commit**

```bash
git add internal/querynodev2/delegator/adaptive/executor.go
git commit -s -m "feat: OTel spans for adaptive search"
```

---

### Task 17: Structured log on completion

**Files:**
- Modify: `internal/querynodev2/delegator/delegator_adaptive.go`

- [ ] **Step 1: Emit one INFO log per query**

At the end of `tryAdaptiveSearch` (after successful Run):

```go
log.Ctx(ctx).Info("adaptive search completed",
    zap.Int64("collectionID", in.CollectionID),
    zap.Int("batches", out.Batches),
    zap.Int("pruned", out.Pruned),
    zap.String("reason", out.Reason),
    zap.Duration("elapsed", time.Since(t0)))
```

- [ ] **Step 2: Commit**

```bash
git add internal/querynodev2/delegator/delegator_adaptive.go
git commit -s -m "feat: structured log for adaptive search completion"
```

---

## Phase 8 — Pre-Launch Validation

### Task 18: ASan long-running test (heap leak check)

- [ ] **Step 1: Build with AddressSanitizer**

Follow existing Milvus ASan build pattern. Run the full integration suite under ASan for ≥ 30 minutes:

```bash
cd build-Debug && cmake .. -DENABLE_ASAN=ON && ninja
# start standalone with ASan-enabled binaries
# run tests/integration/adaptivesearch/ in a loop
```

Expected: no leaks reported. If leaks found, fix and repeat.

- [ ] **Step 2: Document the validation in the spec's risk table (no code)**

Append a line to `docs/adaptive_search_design.md` §13 risk table indicating ASan validation date and result.

- [ ] **Step 3: Commit**

```bash
git add docs/adaptive_search_design.md
git commit -s -m "doc: record ASan validation for adaptive search"
```

---

### Task 19: Benchmark against one-shot

- [ ] **Step 1: Run perf harness with adaptive on/off**

Use the existing Milvus perf framework (SDK level). Collect:
- P50 / P99 latency
- Recall@topK
- Segments searched per query

Dataset candidates:
- SIFT-1M / SIFT-10M
- Custom cluster-heavy synthetic
- High-dim (768d) embedding-like

- [ ] **Step 2: Capture results in a benchmark report file**

Write `docs/adaptive_search_benchmark.md` with numeric tables. No code commit required unless results trigger design changes.

- [ ] **Step 3: Commit**

```bash
git add docs/adaptive_search_benchmark.md
git commit -s -m "doc: adaptive search benchmark report"
```

---

## Self-Review Checklist (run after completing all tasks)

- [ ] All spec §3.2 "New vs Modified" items implemented (paramtable, metrics, C++ heap, Go wrapper, adaptive package, dispatcher).
- [ ] All spec §7 test layers covered (C++ unit, Go unit, integration, recall).
- [ ] No `EnableSegmentPrune` / `PruneSegments` / existing reducer modifications in any commit diff.
- [ ] `make test-querynode` passes.
- [ ] `./bin/segcore_test --gtest_filter=IncrementalReduceHeap.*` passes.
- [ ] Adaptive path covered end-to-end by integration test.
- [ ] Recall gate (2% regression) respected by `test_adaptive_search.py`.

---

## Follow-ups (explicitly out of v1 scope)

- HybridSearch support.
- GroupBy support.
- nq > 1 batch union strategy beyond "order by first query vector".
- Threshold hint push-down to worker segcore (intra-segment early termination).
- Auto-tune convergence parameters per collection.
