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

package delegator

import (
	"container/heap"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/querynodev2/delegator/adaptive"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/clustering"
	"github.com/milvus-io/milvus/internal/util/segcore"
	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v2/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/timerecord"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"google.golang.org/protobuf/proto"
)

const minSegmentsForAdaptive = 16

// shouldUseAdaptiveSearch checks if the adaptive search path should be used.
// This is the dispatcher predicate invoked at the top of sd.search().
func (sd *shardDelegator) shouldUseAdaptiveSearch(req *querypb.SearchRequest) bool {
	// Per-collection property override takes precedence over global flag.
	if sd.collection == nil {
		return false
	}
	globalEnabled := paramtable.Get().QueryNodeCfg.AdaptiveSearchEnabled.GetAsBool()
	collOverride := getCollectionPropertyBool(sd.collection.Schema().GetProperties(), "adaptiveSearch.enabled")
	switch {
	case collOverride != nil && !*collOverride:
		return false // explicitly disabled for this collection
	case collOverride != nil && *collOverride:
		// explicitly enabled — proceed to remaining checks
	case !globalEnabled:
		return false // no override + global disabled
	}

	// Must have clustering key on vector field.
	schema := sd.collection.Schema()
	clusteringKeyField := clustering.GetClusteringKeyField(schema)
	if clusteringKeyField == nil {
		return false
	}
	if !typeutil.IsVectorType(clusteringKeyField.GetDataType()) {
		return false // adaptive ordering only applies to vector clustering keys
	}

	// Must have partition stats loaded.
	sd.partitionStatsMut.RLock()
	hasStats := len(sd.partitionStats) > 0
	sd.partitionStatsMut.RUnlock()
	if !hasStats {
		return false
	}

	// Request-level short-circuits.
	searchReq := req.GetReq()
	if searchReq.GetGroupByFieldId() > 0 {
		return false // GroupBy not supported
	}
	if searchReq.GetIsIterator() {
		return false
	}
	if searchReq.GetIsAdvanced() {
		return false // HybridSearch
	}

	return true
}

// adaptiveSearch replaces executeSearchSubTasks with an ordered, batched dispatch.
// It orders sealed segments by centroid distance, dispatches in batches, and stops
// when convergence is detected. Growing segments and sealed segments without partition
// stats are dispatched as a pre-batch.
//
// Returns []*internalpb.SearchResults in the same format as executeSearchSubTasks.
func (sd *shardDelegator) adaptiveSearch(
	ctx context.Context,
	req *querypb.SearchRequest,
	sealed []SnapshotItem,
	growing []SegmentEntry,
	sealedRowCount map[int64]int64,
) ([]*internalpb.SearchResults, error) {
	ctx, rootSpan := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "adaptive.search")
	defer rootSpan.End()

	log := sd.getLogger(ctx)
	tr := timerecord.NewTimeRecorder("adaptiveSearch")

	// 1. Extract query vector + clustering info from request.
	schema := sd.collection.Schema()
	clusteringKeyField := clustering.GetClusteringKeyField(schema)
	queryVec, err := extractFirstQueryVector(req.GetReq(), clusteringKeyField)
	if err != nil {
		log.Warn("adaptive search failed to extract query vector, falling back", zap.Error(err))
		return nil, err
	}

	metricType := req.GetReq().GetMetricType()
	dim := int64(len(queryVec))

	// 2. Collect all sealed segment IDs and build segID → SnapshotItem index.
	allSegIDs := make([]int64, 0)
	segToSnapshot := make(map[int64]int) // segID → index in sealed
	for i, item := range sealed {
		for _, seg := range item.Segments {
			allSegIDs = append(allSegIDs, seg.SegmentID)
			segToSnapshot[seg.SegmentID] = i
		}
	}

	if len(allSegIDs) < minSegmentsForAdaptive {
		return nil, fmt.Errorf("too few segments for adaptive search: %d", len(allSegIDs))
	}

	// 3. Order segments by centroid distance (via CentroidSearcher / fallback).
	_, orderSpan := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "adaptive.order")
	sd.partitionStatsMut.RLock()
	orderedSegIDs := orderSegments(sd.partitionStats, allSegIDs, queryVec, dim,
		clusteringKeyField.GetFieldID(), metricType)
	sd.partitionStatsMut.RUnlock()

	// Snapshot of segments that DO have centroid stats (orderable). Built
	// BEFORE truncation so that segments which simply lack centroids still
	// fall into the pre-batch, while segments pruned by truncation are
	// dropped entirely.
	orderableSet := make(map[int64]struct{}, len(orderedSegIDs))
	for _, id := range orderedSegIDs {
		orderableSet[id] = struct{}{}
	}

	// Truncate to ceil(sqrt(N) * filterRatio) — the same heuristic as one-shot
	// segment_pruner. Remaining segments are pruned outright (no search).
	if n := len(orderedSegIDs); n > 0 {
		filterRatio := paramtable.Get().QueryNodeCfg.DefaultSegmentFilterRatio.GetAsFloat()
		target := int(math.Ceil(math.Sqrt(float64(n)) * filterRatio))
		if target > 0 && target < n {
			orderedSegIDs = orderedSegIDs[:target]
		}
	}
	orderSpan.SetAttributes(attribute.Int("ordered_count", len(orderedSegIDs)))
	orderSpan.End()

	// 4. Pre-batch contains ONLY sealed segments that genuinely have no
	// centroid stats (not in orderableSet). Truncated segments are neither
	// in orderableSet (still there) — wait, they are. Must distinguish by
	// using orderableSet (pre-truncation) for pre-batch, and post-truncation
	// orderedSegIDs for main loop.
	var preBatchSealed []SnapshotItem
	for i, item := range sealed {
		var noStatSegs []SegmentEntry
		for _, seg := range item.Segments {
			if _, ok := orderableSet[seg.SegmentID]; !ok {
				noStatSegs = append(noStatSegs, seg)
			}
		}
		if len(noStatSegs) > 0 {
			preBatchSealed = append(preBatchSealed, SnapshotItem{
				NodeID:   sealed[i].NodeID,
				Segments: noStatSegs,
			})
		}
	}

	// 5. Execute pre-batch (growing + no-stats sealed).
	var allResults []*internalpb.SearchResults
	if len(preBatchSealed) > 0 || len(growing) > 0 {
		_, preBatchSpan := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "adaptive.preBatch")
		preBatchResults, err := sd.executeSearchSubTasks(ctx, req, preBatchSealed, growing, sealedRowCount)
		preBatchSpan.End()
		if err != nil {
			return nil, err
		}
		log.Info("adaptive search pre-batch result topk",
			zap.Int("result_messages", len(preBatchResults)),
			zap.Int("scores", searchResultsScoreCount(preBatchResults)),
			zap.String("topk_summary", summarizeSearchResultsTopKScores(preBatchResults, metricType, int(req.GetReq().GetNq()), int(req.GetReq().GetTopk()), 3, 5)))
		allResults = append(allResults, preBatchResults...)
	}

	// 6. Batch loop over ordered segments.
	// Running top-K heap feeds the convergence checker a real cross-batch
	// threshold (not a per-batch local one). Its output is decoupled from
	// allResults — proxy-side reduce stays identical.
	nq := req.GetReq().GetNq()
	topK := req.GetReq().GetTopk()
	runHeap := newRunningTopK(nq, topK)
	// Seed with pre-batch results so running top-K reflects everything seen.
	if len(allResults) > 0 {
		runHeap.merge(allResults)
		if thresholds, full := runHeap.thresholds(); full {
			log.Info("adaptive search pre-batch threshold",
				zap.Int("result_messages", len(allResults)),
				zap.Int("scores", searchResultsScoreCount(allResults)),
				zap.String("threshold_summary", summarizeAdaptiveThresholds(thresholds, metricType)))
		} else {
			log.Info("adaptive search pre-batch threshold not ready",
				zap.Int("result_messages", len(allResults)),
				zap.Int("scores", searchResultsScoreCount(allResults)))
		}
	}

	batchSizeCfg := paramtable.Get().QueryNodeCfg.AdaptiveSearchBatchSize.GetValue()
	batchSize := adaptive.ResolveBatchSize(batchSizeCfg, len(orderedSegIDs))
	converger := adaptive.NewStableTopKConverger()
	batches := 0

	for i := 0; i < len(orderedSegIDs); i += batchSize {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		end := i + batchSize
		if end > len(orderedSegIDs) {
			end = len(orderedSegIDs)
		}
		batchSegIDs := orderedSegIDs[i:end]

		// Build SnapshotItem list for this batch (grouped by nodeID).
		batchSealed := buildSealedForBatch(sealed, batchSegIDs, segToSnapshot)

		// P1: once running heap is full, push its worst-of-top-K as
		// cardinal PruneControl fields so the graph-search index can
		// early-stop candidates that can't beat the current top-K floor.
		batchReq := req
		pruneControlEnabled := false
		pruneThresholdSummary := ""
		if currentThr, full := runHeap.thresholds(); full {
			batchReq = cloneReqWithPruneControl(req, currentThr, metricType)
			pruneControlEnabled = batchReq != req
			pruneThresholdSummary = summarizeAdaptiveThresholds(currentThr, metricType)
		}
		log.Info("adaptive search dispatch batch",
			zap.Int("batch", batches),
			zap.Int("segment_start", i),
			zap.Int("segment_end", end),
			zap.Int("segment_count", len(batchSegIDs)),
			zap.Int("remaining_after_batch", len(orderedSegIDs)-end),
			zap.Bool("prune_control_enabled", pruneControlEnabled),
			zap.String("prune_threshold_summary", pruneThresholdSummary))

		_, batchSpan := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, fmt.Sprintf("adaptive.batch[%d]", batches))
		batchSpan.SetAttributes(attribute.Int("segment_count", len(batchSegIDs)))
		batchResults, err := sd.executeSearchSubTasks(ctx, batchReq, batchSealed, nil, sealedRowCount)
		batchSpan.End()
		if err != nil {
			return nil, err
		}
		log.Info("adaptive search batch result topk",
			zap.Int("batch", batches),
			zap.Int("result_messages", len(batchResults)),
			zap.Int("scores", searchResultsScoreCount(batchResults)),
			zap.String("topk_summary", summarizeSearchResultsTopKScores(batchResults, metricType, int(nq), int(topK), 3, 5)))
		allResults = append(allResults, batchResults...)
		batches++

		// Convergence check using running (cross-batch) top-K thresholds.
		runHeap.merge(batchResults)
		thresholds, heapFull := runHeap.thresholds()
		if !heapFull {
			// Heap not full yet — no meaningful threshold; skip convergence.
			log.Info("adaptive search batch threshold not ready",
				zap.Int("batch", batches-1),
				zap.Int("result_messages", len(batchResults)),
				zap.Int("scores", searchResultsScoreCount(batchResults)))
			continue
		}
		converged, reason := converger.Check(thresholds, len(batchResults), batches-1)
		log.Info("adaptive search batch threshold",
			zap.Int("batch", batches-1),
			zap.Int("batches", batches),
			zap.Int("result_messages", len(batchResults)),
			zap.Int("scores", searchResultsScoreCount(batchResults)),
			zap.String("threshold_summary", summarizeAdaptiveThresholds(thresholds, metricType)),
			zap.Bool("converged", converged),
			zap.String("reason", reason),
			zap.Int("pruned_if_stop", len(orderedSegIDs)-end))
		if converged {
			pruned := len(orderedSegIDs) - end
			log.Info("adaptive search converged",
				zap.Int("batches", batches),
				zap.Int("pruned", pruned),
				zap.String("reason", reason),
				zap.Duration("elapsed", tr.ElapseSpan()))

			metrics.QueryNodeAdaptiveSearchTotal.WithLabelValues(
				paramtable.GetStringNodeID(),
				fmt.Sprint(sd.collectionID),
				"success").Inc()
			metrics.QueryNodeAdaptiveBatches.WithLabelValues(
				paramtable.GetStringNodeID(),
				fmt.Sprint(sd.collectionID)).Observe(float64(batches))
			metrics.QueryNodeAdaptivePrunedSegments.WithLabelValues(
				paramtable.GetStringNodeID(),
				fmt.Sprint(sd.collectionID)).Observe(float64(pruned))
			metrics.QueryNodeAdaptiveE2ELatency.WithLabelValues(
				paramtable.GetStringNodeID(),
				fmt.Sprint(sd.collectionID)).Observe(float64(tr.ElapseSpan().Milliseconds()))

			return allResults, nil
		}
	}

	// Exhausted all batches without convergence.
	log.Info("adaptive search exhausted all batches",
		zap.Int("batches", batches),
		zap.Duration("elapsed", tr.ElapseSpan()))

	metrics.QueryNodeAdaptiveSearchTotal.WithLabelValues(
		paramtable.GetStringNodeID(),
		fmt.Sprint(sd.collectionID),
		"success").Inc()
	metrics.QueryNodeAdaptiveBatches.WithLabelValues(
		paramtable.GetStringNodeID(),
		fmt.Sprint(sd.collectionID)).Observe(float64(batches))
	metrics.QueryNodeAdaptiveE2ELatency.WithLabelValues(
		paramtable.GetStringNodeID(),
		fmt.Sprint(sd.collectionID)).Observe(float64(tr.ElapseSpan().Milliseconds()))

	return allResults, nil
}

func summarizeAdaptiveThresholds(thresholds []float32, metric string) string {
	if len(thresholds) == 0 {
		return ""
	}
	internalMin, internalMax := thresholds[0], thresholds[0]
	internalSum := float64(0)
	userMin, userMax := adaptiveUserThreshold(thresholds[0], metric), adaptiveUserThreshold(thresholds[0], metric)
	userSum := float64(0)
	for _, threshold := range thresholds {
		if threshold < internalMin {
			internalMin = threshold
		}
		if threshold > internalMax {
			internalMax = threshold
		}
		internalSum += float64(threshold)

		userThreshold := adaptiveUserThreshold(threshold, metric)
		if userThreshold < userMin {
			userMin = userThreshold
		}
		if userThreshold > userMax {
			userMax = userThreshold
		}
		userSum += float64(userThreshold)
	}
	return fmt.Sprintf("metric=%s internal_min=%.6f internal_max=%.6f internal_avg=%.6f user_min=%.6f user_max=%.6f user_avg=%.6f",
		metric,
		internalMin,
		internalMax,
		internalSum/float64(len(thresholds)),
		userMin,
		userMax,
		userSum/float64(len(thresholds)))
}

func adaptiveUserThreshold(threshold float32, metric string) float32 {
	switch strings.ToUpper(metric) {
	case "IP", "COSINE", "BM25":
		return threshold
	default:
		return -threshold
	}
}

func searchResultsScoreCount(results []*internalpb.SearchResults) int {
	count := 0
	for _, result := range results {
		if result == nil || result.GetResultData() == nil {
			continue
		}
		count += len(result.GetResultData().GetScores())
	}
	return count
}

type adaptiveResultTopKSummary struct {
	ResultIndex int                        `json:"result_index"`
	NQ          int                        `json:"nq"`
	ScoreCount  int                        `json:"score_count"`
	Queries     []adaptiveQueryTopKSummary `json:"queries"`
}

type adaptiveQueryTopKSummary struct {
	QueryIndex          int       `json:"query_index"`
	TopK                int       `json:"topk"`
	ScoreBest           float32   `json:"score_best"`
	ScoreWorst          float32   `json:"score_worst"`
	ScoreMedian         float32   `json:"score_median"`
	ScoreHead           []float32 `json:"score_head"`
	ScoreTail           []float32 `json:"score_tail"`
	CosineDistanceBest  *float32  `json:"cosine_distance_best,omitempty"`
	CosineDistanceWorst *float32  `json:"cosine_distance_worst,omitempty"`
	CosineDistanceHead  []float32 `json:"cosine_distance_head,omitempty"`
	CosineDistanceTail  []float32 `json:"cosine_distance_tail,omitempty"`
}

func summarizeSearchResultsTopKScores(results []*internalpb.SearchResults, metric string, reqNQ int, reqTopK int, maxQueries int, maxScores int) string {
	if len(results) == 0 {
		return "[]"
	}
	summaries := make([]adaptiveResultTopKSummary, 0, len(results))
	for resultIdx, result := range results {
		if result == nil || result.GetResultData() == nil {
			continue
		}
		data := result.GetResultData()
		scores := data.GetScores()
		topks := data.GetTopks()
		nq := len(topks)
		if nq == 0 {
			nq = reqNQ
		}
		if nq <= 0 {
			continue
		}

		summary := adaptiveResultTopKSummary{
			ResultIndex: resultIdx,
			NQ:          nq,
			ScoreCount:  len(scores),
			Queries:     make([]adaptiveQueryTopKSummary, 0, minInt(nq, maxQueries)),
		}
		offset := 0
		for queryIdx := 0; queryIdx < nq && queryIdx < maxQueries; queryIdx++ {
			queryTopK := reqTopK
			if queryIdx < len(topks) {
				queryTopK = int(topks[queryIdx])
			}
			if queryTopK <= 0 {
				continue
			}
			end := offset + queryTopK
			if end > len(scores) {
				end = len(scores)
			}
			if offset >= end {
				break
			}

			queryScores := scores[offset:end]
			querySummary := adaptiveQueryTopKSummary{
				QueryIndex:  queryIdx,
				TopK:        len(queryScores),
				ScoreBest:   roundAdaptiveScore(queryScores[0]),
				ScoreWorst:  roundAdaptiveScore(queryScores[len(queryScores)-1]),
				ScoreMedian: roundAdaptiveScore(queryScores[len(queryScores)/2]),
				ScoreHead:   roundAdaptiveScores(headFloat32(queryScores, maxScores)),
				ScoreTail:   roundAdaptiveScores(tailFloat32(queryScores, maxScores)),
			}
			if strings.EqualFold(metric, "COSINE") {
				best := roundAdaptiveScore(1 - queryScores[0])
				worst := roundAdaptiveScore(1 - queryScores[len(queryScores)-1])
				querySummary.CosineDistanceBest = &best
				querySummary.CosineDistanceWorst = &worst
				querySummary.CosineDistanceHead = roundAdaptiveDistances(headFloat32(queryScores, maxScores))
				querySummary.CosineDistanceTail = roundAdaptiveDistances(tailFloat32(queryScores, maxScores))
			}
			summary.Queries = append(summary.Queries, querySummary)
			offset += queryTopK
		}
		summaries = append(summaries, summary)
	}
	bytes, err := json.Marshal(summaries)
	if err != nil {
		return fmt.Sprintf("marshal_error=%v", err)
	}
	return string(bytes)
}

func headFloat32(values []float32, n int) []float32 {
	if n <= 0 || len(values) <= n {
		return values
	}
	return values[:n]
}

func tailFloat32(values []float32, n int) []float32 {
	if n <= 0 || len(values) <= n {
		return values
	}
	return values[len(values)-n:]
}

func roundAdaptiveScores(values []float32) []float32 {
	rounded := make([]float32, 0, len(values))
	for _, value := range values {
		rounded = append(rounded, roundAdaptiveScore(value))
	}
	return rounded
}

func roundAdaptiveDistances(scores []float32) []float32 {
	distances := make([]float32, 0, len(scores))
	for _, score := range scores {
		distances = append(distances, roundAdaptiveScore(1-score))
	}
	return distances
}

func roundAdaptiveScore(value float32) float32 {
	return float32(math.Round(float64(value)*1e6) / 1e6)
}

func minInt(left int, right int) int {
	if left < right {
		return left
	}
	return right
}

// orderSegments uses partition stats centroids to order segments by distance.
// Falls back to input order if ordering fails.
func orderSegments(
	partitionStats map[UniqueID]*storage.PartitionStatsSnapshot,
	segIDs []int64,
	queryVec []float32,
	dim int64,
	fieldID int64,
	metric string,
) []int64 {
	// Collect centroids across all partitions.
	var ids []int64
	var centroids []float32
	seen := make(map[int64]struct{})

	for _, ps := range partitionStats {
		if ps == nil {
			continue
		}
		for _, segID := range segIDs {
			if _, ok := seen[segID]; ok {
				continue
			}
			segStat, exists := ps.SegmentStats[segID]
			if !exists {
				continue
			}
			for _, fs := range segStat.FieldStats {
				if fs.FieldID == fieldID && len(fs.Centroids) > 0 {
					if c, ok := fs.Centroids[0].GetValue().([]float32); ok && int64(len(c)) == dim {
						ids = append(ids, segID)
						centroids = append(centroids, c...)
						seen[segID] = struct{}{}
					}
					break
				}
			}
		}
	}

	if len(ids) == 0 {
		return nil
	}

	// Use CentroidSearcher for SIMD-accelerated ordering.
	searcher, err := segcore.NewCentroidSearcher(dim, metric)
	if err != nil {
		return ids // fallback: unordered
	}
	defer searcher.Close()

	if err := searcher.Update(ids, centroids); err != nil {
		return ids
	}
	ordered, err := searcher.Order(queryVec, 1)
	if err != nil {
		return ids
	}
	return ordered
}

// buildSealedForBatch filters sealed SnapshotItems to include only segments
// in batchSegIDs, preserving NodeID grouping.
func buildSealedForBatch(
	sealed []SnapshotItem,
	batchSegIDs []int64,
	segToSnapshot map[int64]int,
) []SnapshotItem {
	batchSet := make(map[int64]struct{}, len(batchSegIDs))
	for _, id := range batchSegIDs {
		batchSet[id] = struct{}{}
	}

	// Group by NodeID (original SnapshotItem structure).
	nodeSegs := make(map[int64][]SegmentEntry)
	for _, segID := range batchSegIDs {
		snapIdx, ok := segToSnapshot[segID]
		if !ok {
			continue
		}
		item := sealed[snapIdx]
		for _, seg := range item.Segments {
			if seg.SegmentID == segID {
				nodeSegs[item.NodeID] = append(nodeSegs[item.NodeID], seg)
				break
			}
		}
	}

	result := make([]SnapshotItem, 0, len(nodeSegs))
	for nodeID, segs := range nodeSegs {
		result = append(result, SnapshotItem{
			NodeID:   nodeID,
			Segments: segs,
		})
	}
	return result
}

// extractFirstQueryVector parses the first query vector from the search request.
func extractFirstQueryVector(
	searchReq *internalpb.SearchRequest,
	clusteringKeyField *schemapb.FieldSchema,
) ([]float32, error) {
	var vectorsHolder commonpb.PlaceholderGroup
	if err := proto.Unmarshal(searchReq.GetPlaceholderGroup(), &vectorsHolder); err != nil {
		return nil, err
	}
	if len(vectorsHolder.GetPlaceholders()) == 0 || len(vectorsHolder.GetPlaceholders()[0].GetValues()) == 0 {
		return nil, fmt.Errorf("empty placeholder group")
	}

	vecBytes := vectorsHolder.GetPlaceholders()[0].GetValues()[0]
	dimStr, err := funcutil.GetAttrByKeyFromRepeatedKV("dim", clusteringKeyField.GetTypeParams())
	if err != nil {
		return nil, err
	}
	dim := 0
	fmt.Sscanf(dimStr, "%d", &dim)
	if dim <= 0 {
		return nil, fmt.Errorf("invalid dimension: %s", dimStr)
	}

	// Deserialize float32 vector from bytes.
	vec := clustering.DeserializeFloatVector(vecBytes)
	if len(vec) != dim {
		return nil, fmt.Errorf("vector dim mismatch: got %d, expected %d", len(vec), dim)
	}
	return vec, nil
}

// Default PruneControl knobs sent down to the vector index (cardinal HNSW).
// prune_k_step > 0 is the sentinel that enables pruning; init_step is the
// minimum number of graph pops before the prune check is allowed to fire.
const (
	adaptivePruneInitStepDefault = 16
	adaptivePruneKStepDefault    = 4
)

func adaptivePruneParam(envKey string, defaultValue int) int {
	if v := os.Getenv(envKey); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
			return parsed
		}
	}
	return defaultValue
}

// cloneReqWithPruneControl returns a deep copy of req with cardinal PruneControl
// fields injected into VectorANNS.QueryInfo.search_params:
//   - prune_distance_thres: loosest per-nq worst-of-top-K score (see below)
//   - prune_init_step:      minimum pops before pruning may fire
//   - prune_k_step:         consecutive over-threshold pops to early-stop (>0 enables)
//
// Threshold conversion: cardinal consumes the value in user-metric space.
// Milvus's internal score is "higher-is-better" (L2 negated, IP/COSINE
// direct). For positively-related metrics the internal value equals the
// user-space similarity and we pass it through; for L2-family metrics we
// negate to recover the actual distance.
//
// On any parse/marshal failure, returns the original req unchanged.
func cloneReqWithPruneControl(req *querypb.SearchRequest, thresholds []float32, metric string) *querypb.SearchRequest {
	if len(thresholds) == 0 {
		return req
	}
	// Loosest per-nq threshold (min in higher-is-better form) so no nq
	// loses a legitimate top-K candidate across queries.
	minThr := thresholds[0]
	for _, t := range thresholds[1:] {
		if t < minThr {
			minThr = t
		}
	}
	// Convert internal score → user-metric distance threshold.
	userThr := minThr
	switch strings.ToUpper(metric) {
	case "IP", "COSINE", "BM25":
		// internal score == user similarity (higher-is-better)
	default:
		// L2/HAMMING/JACCARD: internal score = -user_distance.
		userThr = -minThr
	}

	var plan planpb.PlanNode
	if err := proto.Unmarshal(req.GetReq().GetSerializedExprPlan(), &plan); err != nil {
		return req
	}
	anns := plan.GetVectorAnns()
	if anns == nil || anns.GetQueryInfo() == nil {
		return req
	}
	qi := anns.QueryInfo

	params := map[string]interface{}{}
	if qi.SearchParams != "" {
		if err := json.Unmarshal([]byte(qi.SearchParams), &params); err != nil {
			return req
		}
	}
	params["prune_distance_thres"] = userThr
	if _, ok := params["prune_init_step"]; !ok {
		params["prune_init_step"] = adaptivePruneParam("MILVUS_ADAPTIVE_PRUNE_INIT_STEP", adaptivePruneInitStepDefault)
	}
	if _, ok := params["prune_k_step"]; !ok {
		params["prune_k_step"] = adaptivePruneParam("MILVUS_ADAPTIVE_PRUNE_K_STEP", adaptivePruneKStepDefault)
	}

	newParamsBytes, err := json.Marshal(params)
	if err != nil {
		return req
	}
	qi.SearchParams = string(newParamsBytes)

	newPlanBytes, err := proto.Marshal(&plan)
	if err != nil {
		return req
	}

	cloned := proto.Clone(req).(*querypb.SearchRequest)
	cloned.Req.SerializedExprPlan = newPlanBytes
	return cloned
}

// runningTopK tracks the cross-batch top-K score per nq in memory. It is
// decoupled from the SearchResults returned to the caller — we keep those
// intact so the proxy-side reduce remains correct. The heap is only used
// to feed the convergence checker a *running* threshold (not per-batch).
type runningTopK struct {
	nq, topK int
	heaps    []*float32MinHeap // one min-heap per nq, root = worst-of-topK
}

func newRunningTopK(nq, topK int64) *runningTopK {
	r := &runningTopK{nq: int(nq), topK: int(topK), heaps: make([]*float32MinHeap, nq)}
	for q := range r.heaps {
		h := make(float32MinHeap, 0, topK)
		r.heaps[q] = &h
	}
	return r
}

// merge updates the per-nq heap with every (score) tuple in the batch.
// Ids are irrelevant for threshold computation, so we ignore them.
func (r *runningTopK) merge(results []*internalpb.SearchResults) {
	for _, sr := range results {
		data := sr.GetResultData()
		if data == nil {
			continue
		}
		scores := data.GetScores()
		topks := data.GetTopks()
		offset := int64(0)
		for q := 0; q < r.nq && q < len(topks); q++ {
			k := topks[q]
			for i := int64(0); i < k; i++ {
				if offset+i >= int64(len(scores)) {
					break
				}
				r.push(q, scores[offset+i])
			}
			offset += k
		}
	}
}

func (r *runningTopK) push(q int, score float32) {
	h := r.heaps[q]
	if h.Len() < r.topK {
		heap.Push(h, score)
		return
	}
	// heap full → only push if beats the current worst (root of min-heap)
	if score > (*h)[0] {
		(*h)[0] = score
		heap.Fix(h, 0)
	}
}

// thresholds returns the running worst-of-top-K per nq. The second return
// value is true iff *all* heaps are full (otherwise the threshold is not
// a meaningful stopping signal yet).
func (r *runningTopK) thresholds() ([]float32, bool) {
	out := make([]float32, r.nq)
	full := true
	for q := 0; q < r.nq; q++ {
		if r.heaps[q].Len() < r.topK {
			full = false
			out[q] = -math.MaxFloat32
		} else {
			out[q] = (*r.heaps[q])[0]
		}
	}
	return out, full
}

// float32MinHeap implements heap.Interface with root = smallest.
type float32MinHeap []float32

func (h float32MinHeap) Len() int            { return len(h) }
func (h float32MinHeap) Less(i, j int) bool  { return h[i] < h[j] }
func (h float32MinHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *float32MinHeap) Push(x interface{}) { *h = append(*h, x.(float32)) }
func (h *float32MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// extractBatchThresholds extracts the worst-of-top-K distance per nq from batch results.
// Used by the convergence checker.
func extractBatchThresholds(results []*internalpb.SearchResults, nq int64) []float32 {
	thresholds := make([]float32, nq)
	for i := range thresholds {
		thresholds[i] = -1e38 // sentinel: any real distance beats this
	}

	for _, r := range results {
		if r == nil || r.GetResultData() == nil {
			continue
		}
		data := r.GetResultData()
		scores := data.GetScores()
		topks := data.GetTopks()

		offset := int64(0)
		for q := int64(0); q < nq && q < int64(len(topks)); q++ {
			k := topks[q]
			if k <= 0 {
				offset += k
				continue
			}
			// Worst-of-top-K is the last score in this nq's slice.
			lastIdx := offset + k - 1
			if lastIdx < int64(len(scores)) {
				if scores[lastIdx] > thresholds[q] {
					thresholds[q] = scores[lastIdx]
				}
			}
			offset += k
		}
	}
	return thresholds
}

// getCollectionPropertyBool reads a boolean property from collection properties.
// Returns nil if not set, or a pointer to the boolean value.
func getCollectionPropertyBool(props []*commonpb.KeyValuePair, key string) *bool {
	for _, p := range props {
		if p.GetKey() == key {
			v := strings.ToLower(p.GetValue()) == "true"
			return &v
		}
	}
	return nil
}
