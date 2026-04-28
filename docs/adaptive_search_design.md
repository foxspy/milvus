# Adaptive Search Design (Order-based Batched Execution)

Status: Draft
Owner: @xianliang.li
Last updated: 2026-04-15

## 1. Overview

Milvus's clustering compaction produces per-segment `PartitionStats` containing centroid information for vector clustering keys. The query side uses these today for a one-shot segment prune (`segment_pruner.go`): sort segments by centroid distance, pick `sqrt(N)·filterRatio` segments, run all in parallel.

This design introduces a **second, parallel execution path** — Adaptive Search — that uses the same centroids to drive an **ordered, batched search** with convergence-based early termination. The new path is added alongside the existing one-shot path without modifying it; a lightweight dispatcher decides per query which to use, and falls back to the existing path on any short-circuit condition.

## 2. Goals / Non-Goals

### Goals

- Reduce the number of segments actually searched by using already-computed top-K results to short-circuit later batches.
- Zero changes to producer side (clustering compaction, `FieldStats`, `PartitionStatsSnapshot` JSON format).
- Zero changes to existing one-shot execution code path; new path is additive and can be fully disabled by flag.
- Support both `Search` and `Query` vector paths.
- Default-off rollout, with per-collection override via collection property.

### Non-Goals

- No geometric pruning (no radius, no lower-bound arithmetic). Centroid distance is used purely as an **ordering signal**.
- No mid-batch cancellation for convergence (only whole-batch wait + decide + break).
- No support in v1 for: Range search, Iterator search, GroupBy, HybridSearch — these fall back to the existing path.
- No worker RPC protocol change. No segcore search interface change.
- No proto changes.

## 3. Architecture

### 3.1 Two Parallel Paths

```
shardDelegator.Search / Query
        |
  shouldUseAdaptive? ───────────yes──▶  Adaptive path (NEW, isolated)
        |                                    |
        no (short-circuit)                Return
        |
        ▼
  Existing one-shot path (UNCHANGED)
        |
      Return
```

Short-circuit into the existing path occurs when any of these is true:
- Global flag `queryNode.adaptiveSearch.enabled = false`
- Collection property `adaptiveSearch.enabled ≠ "true"`
- No clustering key on the collection
- `PartitionStats` not loaded for the shard
- Request is Range / Iterator / GroupBy / HybridSearch
- Total sealed segment count < `minSegmentsForAdaptive` (constant = 16)
- Runtime error in the adaptive executor (graceful fallback to one-shot)

### 3.2 New vs Modified

| Scope | What |
|---|---|
| New code (Go) | `internal/querynodev2/delegator/adaptive/` package: executor, orderer, convergence, CGO heap wrapper |
| New code (Go, segments) | `internal/querynodev2/segments/reduce_incremental.go` |
| New code (C++) | `internal/core/src/segcore/reduce_c_incremental.{h,cpp}` |
| New code (paramtable) | `queryNode.adaptiveSearch.enabled` + `.batchSize` |
| New code (collection property) | `adaptiveSearch.enabled` per-collection |
| New code (metrics, OTel spans) | Adaptive-specific observability |
| Modified (minimum) | `shardDelegator.Search` (`delegator.go:380`) and `Query` (`:675`): prepend a dispatcher; `else`-branch contains unchanged original code |
| Unchanged | `segment_pruner.go`, `scalar_pruner.go`, existing reducer (`ReduceSearchResultsAndFillData`), `SearchSegments` RPC, producer (`clustering_compactor.go`), stats proto, delegator sync logic |

### 3.3 Segment Categories (for batching)

The adaptive path treats sealed segments in one of two buckets:
- **Orderable**: sealed segments present in `PartitionStats` with a valid centroid → participate in centroid-distance ordering and batching.
- **Pre-batch**: growing segments and sealed segments absent from `PartitionStats` → searched once in a pre-batch (cannot be ordered), results merged into the heap before the ordered loop begins.

Position-agnostic: local sealed and remote sealed segments are mixed in the same batch. Rationale: simpler executor; latency cost of mixing is acceptable in v1.

## 4. Control Flow

### 4.1 Adaptive Executor Main Loop

```go
func (e *batchedExecutor) Search(ctx, req, sealed, growing) (Result, error) {
    heap, err := NewReduceHeap(plan, nq, topK, metric)
    if err != nil { return nil, err }
    defer heap.Close()

    // Order the orderable subset; split out pre-batch.
    orderable, preBatch := split(sealed, e.partStats)
    orderedSegs := e.orderer.Order(req.Vectors, orderable, e.partStats)

    // Pre-batch: growing + no-stats sealed.
    preSegs := preBatch ∪ growing
    if len(preSegs) > 0 {
        results, err := e.dispatch(ctx, preSegs, req)
        if err != nil { return nil, err }
        if err := heap.MergeBatch(results); err != nil { return nil, err }
    }

    // Ordered batches.
    batchSize := e.config.ResolveBatchSize(len(orderedSegs))
    for batchIdx, batch := range chunks(orderedSegs, batchSize) {
        batchCtx, cancel := context.WithCancel(ctx)
        results, err := e.dispatch(batchCtx, batch, req)
        cancel()
        if err != nil { return nil, err }
        if err := heap.MergeBatch(results); err != nil { return nil, err }

        if e.converger.Check(heap, batchIdx) {
            break
        }
    }

    return heap.Finalize()
}
```

Semantics:
- Batches are strictly sequential. Within a batch, dispatch uses the existing parallel fanout to workers.
- Convergence does not trigger mid-batch cancellation; only "don't issue the next batch".
- Runtime errors propagate; the outer dispatcher catches and falls back to the one-shot path (best-effort).

### 4.2 Orderer

```go
type Orderer interface {
    // Returns segments sorted by distance(query_vector[0], segment.centroid).
    // For L2: ascending; for IP/COSINE: descending.
    // Uses only nq=1's first query vector for ordering; nq>1 is a known degradation.
    Order(queryVectors [][]byte, segs []SegmentEntry, stats *storage.PartitionStatsSnapshot) []SegmentEntry
}
```

nq > 1: v1 orders by the first query vector only. nq independent orderings would balloon batch union; accepted as a known limitation.

### 4.3 Convergence

Algorithm-layer convergence judgment is validated externally and is out of scope of this design. The executor calls an opaque `Check(heap, batchIdx) bool` that returns whether to stop. The concrete judge lives in `adaptive/convergence.go` and is implementation-level, not spec-level.

## 5. C++ CGO API

New header: `internal/core/src/segcore/reduce_c_incremental.h`

```c
typedef void* CReduceHeap;

CStatus NewReduceHeap(CSearchPlan plan,
                      int64_t nq,
                      int64_t topK,
                      CReduceHeap* out);

CStatus ReduceHeap_MergeBatch(CReduceHeap heap,
                              CSearchResultDataBlobs* search_results,
                              int count);

CStatus ReduceHeap_GetThresholds(CReduceHeap heap,
                                 float* out_thresholds /* len == nq */);

CStatus ReduceHeap_Finalize(CReduceHeap heap,
                            CSearchResultDataBlobs* out_result);

void DeleteReduceHeap(CReduceHeap heap);
```

Properties:
- `ReduceHeap` internally holds `nq` independent top-K heaps, direction determined by the metric type from `CSearchPlan`.
- `MergeBatch` expands each `SearchResultDataBlobs` into (nq × topK) entries, pushes into corresponding heaps, evicts when full.
- `GetThresholds` returns the k-th entry distance per nq (sentinel if the heap is not full).
- `Finalize` consumes the heap to produce the final `CSearchResultDataBlobs`. After `Finalize`, the heap is not reusable.
- Existing `ReduceSearchResultsAndFillData` is not modified. Where possible the new code reuses its `fill-data` subroutine internally without extending its external interface.

## 6. Go Wrapper

New file: `internal/querynodev2/segments/reduce_incremental.go`

```go
type ReduceHeap struct { /* opaque, holds C handle */ }

func NewReduceHeap(plan *SearchPlan, nq, topK int64, metric string) (*ReduceHeap, error)

func (h *ReduceHeap) MergeBatch(results []*internalpb.SearchResults) error
func (h *ReduceHeap) Thresholds() ([]float32, error)
func (h *ReduceHeap) Finalize()    (*SearchResultDataBlobs, error)
func (h *ReduceHeap) Close()
```

Lifecycle: `New → MergeBatch * N → Finalize → (released)`, with `defer Close()` as a safety net for error paths. No `runtime.SetFinalizer` — explicit `Close` is preferred to keep C resource release deterministic.

## 7. Dispatch & RPC

The adaptive executor reuses the existing worker dispatch surface:
- Local sealed segments → in-process segcore call (same as one-shot).
- Remote sealed segments → existing `QueryNode.SearchSegments` RPC.
- Growing segments → existing streaming-side forward mechanism.

No RPC protocol changes. No worker-side changes. Each batch is one round of fanout (α mode: unary RPCs, N times).

## 8. Configuration

### 8.1 Paramtable (`configs/milvus.yaml`)

```yaml
queryNode:
  adaptiveSearch:
    enabled: false       # global on/off, default OFF
    batchSize: auto      # "auto" | positive integer
```

Code location: `pkg/v2/util/paramtable/component_param.go`, under `QueryNodeCfg`, placed adjacent to `EnableSegmentPrune`.

Hot reload: follows existing paramtable behavior (seconds-level effective without restart).

### 8.2 Collection Property

Key: `adaptiveSearch.enabled`, value `"true"` or `"false"`.

```python
client.alter_collection_properties(
    collection_name="my_coll",
    properties={"adaptiveSearch.enabled": "true"}
)
```

Precedence (per-collection > global):
- Property `"true"`  → use adaptive (overrides global off)
- Property `"false"` → use one-shot (overrides global on)
- Property unset    → follow global flag

### 8.3 Batch Size Resolution

- `"auto"` (default): internal formula based on segment count. Concrete formula is an implementation detail; it is not part of the user-visible contract.
- Positive integer: respected verbatim (capped at segment count).
- `< 1`: treated as invalid; fall back to auto.

## 9. Dispatcher

```go
func (sd *shardDelegator) shouldUseAdaptive(req searchOrQueryReq) bool {
    if !sd.adaptiveEnabledFor(req.CollectionID) { return false }
    if sd.clusteringKeyField == nil             { return false }
    if len(sd.partitionStats) == 0              { return false }
    if req.IsRangeSearch() || req.IsIterator()  { return false }
    if req.HasGroupBy() || req.IsHybridSearch() { return false }
    if len(sd.sealedSegments) < minSegmentsForAdaptive { return false }
    return true
}

// in Search/Query entry:
if sd.shouldUseAdaptive(req) {
    if result, err := sd.tryAdaptive(ctx, req); err == nil {
        return result, nil
    }
    // On adaptive error, emit fallback metric and fall through to one-shot.
}
// ... existing one-shot code unchanged
```

`minSegmentsForAdaptive = 16` (constant; not a config).

## 10. Observability

### 10.1 Metrics

All new; existing `QueryNodeSegmentPrune{Bias,Ratio,Latency}` are unchanged and continue to be written by the one-shot path only.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `milvus_querynode_adaptive_search_total` | Counter | `collection`, `result`={success,fallback,error} | Adaptive entry invocations by outcome |
| `milvus_querynode_adaptive_batches` | Histogram | `collection` | Number of batches actually executed per query |
| `milvus_querynode_adaptive_pruned_segments` | Histogram | `collection` | Segments skipped due to early termination |
| `milvus_querynode_adaptive_batch_latency` | Histogram | `collection`, `batch_idx_bucket` | Per-batch latency (index bucketed 0 / 1 / 2 / 3+) |
| `milvus_querynode_adaptive_e2e_latency` | Histogram | `collection` | End-to-end adaptive path latency |
| `milvus_querynode_adaptive_converge_reason` | Counter | `reason`={stable,exhausted,error} | Termination cause distribution |

### 10.2 OpenTelemetry Spans

```
span: shardDelegator.search
  ├── span: shouldUseAdaptive          attrs: enabled, fallback_reason
  └── (if adaptive)
      span: adaptive.executor
        ├── span: adaptive.order
        ├── span: adaptive.preBatch
        ├── span: adaptive.batch[i]    attrs: batch_size, segment_count
        │   ├── span: dispatch
        │   └── span: heap.merge
        ├── span: adaptive.converge    attrs: reason
        └── span: heap.finalize
```

Reuses `otel.Tracer(typeutil.QueryNodeRole)`. No new tracer.

### 10.3 Logs

- `INFO`: one structured line per adaptive-path query with batches executed, segments skipped, convergence reason, total latency.
- `WARN`: fallback triggered with reason.
- `DEBUG`: per-batch detail; disabled by default.

## 11. Testing

| Layer | Content | Location |
|---|---|---|
| C++ unit | `IncrementalReduceHeap` API correctness; equivalence to one-shot reducer on same inputs | `internal/core/unittest/test_incremental_reduce.cpp` |
| C++ unit | Coverage of `nq ∈ {1, 16, 64}`, `topK ∈ {1, 100, 1000}`, metrics {L2, IP, COSINE} | same |
| Go unit | `BatchedExecutor` main loop with mocked heap/dispatch: convergence, cancel, error paths | `internal/querynodev2/delegator/adaptive/executor_test.go` |
| Go unit | `Orderer` correctness; empty stats; growing/no-stats segment split | `adaptive/orderer_test.go` |
| Go unit | Dispatcher all short-circuit branches | `delegator_adaptive_dispatch_test.go` |
| Go integration | Result equivalence: adaptive vs one-shot top-K set ≈ equivalent on same dataset | `tests/integration/adaptivesearch/` new suite |
| Go integration | Per-collection property override; runtime flag flip | same |
| Recall regression | Multi-dataset × multi-metric × multi-topK brute-force comparison | `tests/python_client/testcases/test_adaptive_search.py` new suite; nightly |
| Perf | Adaptive on/off comparison of latency and throughput | Existing perf framework, new cases |

**Recall gate**: the offline recall regression suite is a launch gate. If `recall@K` regression vs one-shot exceeds 2% on any dataset in the suite, the release is blocked until the cause is fixed.

## 12. Rollout

### Phase 0 – Development (≈4 weeks)

Feature-complete code with unit/integration tests. C++ unit tests green. Recall regression suite integrated into nightly CI.

### Phase 1 – Internal Dogfood (2 weeks)

Global flag stays `false`. One or two internal test clusters enable per-collection on a few collections. Monitor:
- `adaptive_search_total{result=fallback}` ratio < 1%
- `recall_estimate` stable vs baseline

### Phase 2 – Community Beta (4 weeks)

Documentation and release notes published. Community users invited to enable on staging collections. Collect feedback on recall and adjust internal constants if needed.

### Phase 3 – Production Recommended (4–8 weeks post Phase 2)

Marked stable in docs. Global default remains `false`; docs describe when to enable. per-collection property is the recommended form.

### Phase 4 – Global Default (future minor release)

After observing broad production adoption, evaluate changing the global default to `true` in a subsequent minor release.

### Rollback

- Global flag set to `false` → all collections immediately revert to one-shot.
- Per-collection property removed or set to `"false"` → that collection reverts.
- No redeploy or binary rollback required.

## 13. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| C++ heap memory leak | Medium | High | Explicit `Close` + long-running tests + Valgrind/ASan in CI |
| Unobserved recall regression | Medium | High | Gate on offline recall suite; online recall sampling in Phase 1/2 |
| Cancel unresponsive inside long segcore traversal | Medium | Medium | Empirical verification on HNSW/DiskANN; document adaptive latency caveats if not immediate |
| Collection property parsed incorrectly | Low | Medium | Explicit schema validation + dispatcher tests |
| Adaptive slower than one-shot on small collections | Medium | Low | `minSegmentsForAdaptive=16` guard |
| nq > 1 yields less benefit | Medium | Low | Documented limitation; v2 can address |

## 14. Open Questions (Deferred Beyond v1)

- HybridSearch / GroupBy support.
- nq > 1 batch union strategy.
- Threshold hint push-down to worker segcore for intra-segment early termination.
- Automated convergence parameter tuning per collection.

## 15. References

- Clustering compaction producer: `internal/datanode/compactor/clustering_compactor.go`
- Current one-shot prune: `internal/querynodev2/delegator/segment_pruner.go`, `scalar_pruner.go`
- Existing reducer: `internal/core/src/segcore/reduce/ReduceSearchResultsAndFillData.*`
- Delegator entry points: `internal/querynodev2/delegator/delegator.go:380` (Search), `:675` (Query)
- Upstream clustering compaction write-up: `cluster_compaction.md` (project root)
