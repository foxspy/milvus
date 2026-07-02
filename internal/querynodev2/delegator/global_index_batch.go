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
	"context"
	"sort"
	"sync"

	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/internal/util/reduce"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// segQueryTask is one (segment, query) marking: a global query index plus its
// head-index entry-point hints in that segment (nil for passthrough).
type segQueryTask struct {
	globalQuery int64
	hints       []searchHint
}

// workerSearchPlan is the per-worker batched search plan: the set of queries whose
// target segments live on this worker, and, per segment, which of those queries must
// search it. It is the "worker <- queries" inversion of the per-query targets.
type workerSearchPlan struct {
	workerID   int64
	queries    []int64             // global query indices searched on this worker (ascending)
	localIndex map[int64]int32     // global query idx -> local idx in this worker's placeholder
	segTasks   map[int64][]segQueryTask
}

// buildWorkerSearchPlans inverts per-query targets (query -> segment -> hints) into
// per-worker plans (worker -> queries + segment x query marking), using the sealed
// snapshot's NodeID grouping as the segment -> worker map. A worker with no targeted
// segment gets no plan (and therefore no RPC): worker-level pruning is preserved.
func buildWorkerSearchPlans(perQueryTargets []queryPlan, sealed []SnapshotItem) map[int64]*workerSearchPlan {
	segWorker := make(map[int64]int64)
	for _, item := range sealed {
		for _, seg := range item.Segments {
			segWorker[seg.SegmentID] = item.NodeID
		}
	}

	plans := make(map[int64]*workerSearchPlan)
	for q, plan := range perQueryTargets {
		gq := int64(q)
		for segID, hints := range plan {
			worker, ok := segWorker[segID]
			if !ok {
				// Segment not in the current snapshot (e.g. went offline); skip it.
				continue
			}
			wp := plans[worker]
			if wp == nil {
				wp = &workerSearchPlan{
					workerID:   worker,
					localIndex: make(map[int64]int32),
					segTasks:   make(map[int64][]segQueryTask),
				}
				plans[worker] = wp
			}
			if _, seen := wp.localIndex[gq]; !seen {
				wp.localIndex[gq] = 0 // assigned below in ascending order
				wp.queries = append(wp.queries, gq)
			}
			wp.segTasks[segID] = append(wp.segTasks[segID], segQueryTask{globalQuery: gq, hints: hints})
		}
	}

	// Assign local placeholder indices in ascending global-query order for determinism.
	for _, wp := range plans {
		sort.Slice(wp.queries, func(i, j int) bool { return wp.queries[i] < wp.queries[j] })
		for li, gq := range wp.queries {
			wp.localIndex[gq] = int32(li)
		}
	}
	return plans
}

// buildWorkerBatchedRequest builds one batched nq'=M SearchRequest for a worker: its
// placeholder group carries the M queries of the plan (in ascending global order), and
// segment_query_tasks marks, per segment, which local queries search it plus their
// entry-point hints (gated by GlobalIndexSearchEntryPoints).
func buildWorkerBatchedRequest(
	req *querypb.SearchRequest,
	tag string,
	phType commonpb.PlaceholderType,
	queryValues [][]byte,
	wp *workerSearchPlan,
) *querypb.SearchRequest {
	values := make([][]byte, 0, len(wp.queries))
	for _, gq := range wp.queries {
		values = append(values, queryValues[gq])
	}
	phgBytes, _ := proto.Marshal(&commonpb.PlaceholderGroup{
		Placeholders: []*commonpb.PlaceholderValue{{
			Tag:    tag,
			Type:   phType,
			Values: values,
		}},
	})

	newReq := proto.Clone(req.GetReq()).(*internalpb.SearchRequest)
	newReq.Nq = int64(len(wp.queries))
	newReq.PlaceholderGroup = phgBytes

	withEntryPoints := paramtable.Get().QueryNodeCfg.GlobalIndexSearchEntryPoints.GetAsBool()
	tasks := make(map[int64]*internalpb.SegmentSearchTasks, len(wp.segTasks))
	for segID, sts := range wp.segTasks {
		list := &internalpb.SegmentSearchTasks{Tasks: make([]*internalpb.SegmentQueryTask, 0, len(sts))}
		for _, st := range sts {
			task := &internalpb.SegmentQueryTask{QueryIndex: wp.localIndex[st.globalQuery]}
			if withEntryPoints && len(st.hints) > 0 {
				hl := &internalpb.SearchHintList{Hints: make([]*internalpb.SearchHint, 0, len(st.hints))}
				for _, h := range st.hints {
					hl.Hints = append(hl.Hints, &internalpb.SearchHint{
						IdOffset:   h.idOffset,
						IdRange:    h.idRange,
						IdDistance: h.idDistance,
					})
				}
				task.Hints = hl
			}
			list.Tasks = append(list.Tasks, task)
		}
		tasks[segID] = list
	}
	newReq.SegmentQueryTasks = tasks

	return &querypb.SearchRequest{
		Req:             newReq,
		DmlChannels:     req.GetDmlChannels(),
		TotalChannelNum: req.GetTotalChannelNum(),
	}
}

// snapshotForWorker rebuilds a single-worker SnapshotItem keeping only that worker's
// targeted segments, so executeSearchSubTasks issues exactly one RPC to it.
func snapshotForWorker(sealed []SnapshotItem, wp *workerSearchPlan) []SnapshotItem {
	segs := make([]SegmentEntry, 0, len(wp.segTasks))
	for _, item := range sealed {
		if item.NodeID != wp.workerID {
			continue
		}
		for _, seg := range item.Segments {
			if _, ok := wp.segTasks[seg.SegmentID]; ok {
				segs = append(segs, seg)
			}
		}
	}
	if len(segs) == 0 {
		return nil
	}
	return []SnapshotItem{{NodeID: wp.workerID, Segments: segs}}
}

// globalIndexSearchBatched runs the worker-batched global index search: it inverts the
// per-query targets into one batched RPC per worker (marking per segment which queries
// search it), searches growing once with all queries, then reduces per global query
// across its worker/growing shards and concatenates into an nq=N result.
func (sd *shardDelegator) globalIndexSearchBatched(
	ctx context.Context,
	req *querypb.SearchRequest,
	sealed []SnapshotItem,
	growing []SegmentEntry,
	sealedRowCount map[int64]int64,
) ([]*internalpb.SearchResults, error) {
	log := sd.getLogger(ctx)
	r := req.GetReq()
	nq := r.GetNq()
	topk := r.GetTopk()
	metric := r.GetMetricType()

	perQueryTargets, applied, err := sd.headIndexTargetsPerQuery(ctx, req, sealed, sealedRowCount)
	if err != nil {
		return nil, err
	}
	if !applied {
		return nil, errGlobalIndexSearchInapplicable
	}

	tag, phType, queryValues, err := splitPlaceholderGroup(r.GetPlaceholderGroup())
	if err != nil {
		return nil, err
	}
	if int64(len(queryValues)) != nq {
		return nil, merr.WrapErrServiceInternalMsg(
			"global index batched search placeholder count mismatch, nq=%d, placeholders=%d", nq, len(queryValues))
	}

	plans := buildWorkerSearchPlans(perQueryTargets, sealed)

	// One batched RPC per worker (concurrently), plus one growing search over all queries.
	type workerOutput struct {
		wp      *workerSearchPlan
		results []*internalpb.SearchResults
	}
	workerOutputs := make([]workerOutput, 0, len(plans))
	var growingResults []*internalpb.SearchResults

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(globalIndexSearchConcurrency)
	var mu sync.Mutex
	for _, wp := range plans {
		wp := wp
		snapshot := snapshotForWorker(sealed, wp)
		if len(snapshot) == 0 {
			continue
		}
		subReq := buildWorkerBatchedRequest(req, tag, phType, queryValues, wp)
		g.Go(func() error {
			res, err := sd.executeSearchSubTasks(gctx, subReq, snapshot, []SegmentEntry{}, sealedRowCount)
			if err != nil {
				return err
			}
			mu.Lock()
			workerOutputs = append(workerOutputs, workerOutput{wp: wp, results: res})
			mu.Unlock()
			return nil
		})
	}
	if len(growing) > 0 {
		g.Go(func() error {
			res, err := sd.executeSearchSubTasks(gctx, req, []SnapshotItem{}, growing, sealedRowCount)
			if err != nil {
				return err
			}
			mu.Lock()
			growingResults = res
			mu.Unlock()
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}

	// Per global query, collect nq=1 shards from every worker that searched it and from
	// growing, reduce topk, then concatenate into the final nq=N result.
	buckets := make([][]*internalpb.SearchResults, nq)
	for _, out := range workerOutputs {
		reduced, err := segments.ReduceSearchOnQueryNode(ctx, out.results,
			reduce.NewReduceSearchResultInfo(int64(len(out.wp.queries)), topk).WithMetricType(metric))
		if err != nil {
			return nil, err
		}
		if reduced.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
			return nil, merr.Error(reduced.GetStatus())
		}
		slices, err := splitResultPerQuery(ctx, reduced, int64(len(out.wp.queries)), topk, metric)
		if err != nil {
			return nil, err
		}
		for li, gq := range out.wp.queries {
			buckets[gq] = append(buckets[gq], slices[li])
		}
	}
	if len(growingResults) > 0 {
		reduced, err := segments.ReduceSearchOnQueryNode(ctx, growingResults,
			reduce.NewReduceSearchResultInfo(nq, topk).WithMetricType(metric))
		if err != nil {
			return nil, err
		}
		if reduced.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
			return nil, merr.Error(reduced.GetStatus())
		}
		slices, err := splitResultPerQuery(ctx, reduced, nq, topk, metric)
		if err != nil {
			return nil, err
		}
		for q := int64(0); q < nq; q++ {
			buckets[q] = append(buckets[q], slices[q])
		}
	}

	perQueryReduced := make([]*internalpb.SearchResults, nq)
	for q := int64(0); q < nq; q++ {
		if len(buckets[q]) == 0 {
			perQueryReduced[q] = emptySingleQueryResult(topk)
			continue
		}
		reduced, err := segments.ReduceSearchOnQueryNode(ctx, buckets[q],
			reduce.NewReduceSearchResultInfo(1, topk).WithMetricType(metric))
		if err != nil {
			return nil, err
		}
		if reduced.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
			return nil, merr.Error(reduced.GetStatus())
		}
		perQueryReduced[q] = reduced
	}

	merged, err := concatPerQueryResults(ctx, perQueryReduced, nq, topk, metric)
	if err != nil {
		return nil, err
	}

	log.Info("[GIDBG-SEGPRUNE] global index worker-batched search done",
		zap.Int64("nq", nq),
		zap.Int("hitWorkers", len(plans)),
		zap.Int("sealedTotal", countSegments(sealed)),
		zap.Float64("avgSegmentsPerQuery", avgTargets(perQueryTargets)),
		zap.Float64("avgQueriesPerWorker", avgQueriesPerWorker(plans)))
	return []*internalpb.SearchResults{merged}, nil
}

// splitResultPerQuery decodes a reduced nq=count result and re-emits it as count nq=1
// results, so per-query shards from different workers can be regrouped and reduced.
func splitResultPerQuery(ctx context.Context, res *internalpb.SearchResults, count, topk int64, metric string) ([]*internalpb.SearchResults, error) {
	datas, err := segments.DecodeSearchResults(ctx, []*internalpb.SearchResults{res})
	if err != nil {
		return nil, err
	}
	var data *schemapb.SearchResultData
	if len(datas) > 0 {
		data = datas[0]
	}

	out := make([]*internalpb.SearchResults, count)
	offset := int64(0)
	for i := int64(0); i < count; i++ {
		var hit int64
		if data != nil && int(i) < len(data.GetTopks()) {
			hit = data.GetTopks()[i]
		}
		single := &schemapb.SearchResultData{
			NumQueries: 1,
			TopK:       topk,
			Scores:     make([]float32, 0, hit),
			Ids:        &schemapb.IDs{},
			Topks:      []int64{hit},
		}
		if data != nil {
			if len(data.GetFieldsData()) > 0 {
				single.FieldsData = typeutil.PrepareResultFieldData(data.GetFieldsData(), hit)
			}
			for row := int64(0); row < hit; row++ {
				typeutil.AppendIDs(single.Ids, data.GetIds(), int(offset+row))
				single.Scores = append(single.Scores, data.GetScores()[offset+row])
				if len(single.FieldsData) > 0 {
					typeutil.AppendFieldData(single.FieldsData, data.GetFieldsData(), offset+row)
				}
			}
			offset += hit
		}
		enc, err := segments.EncodeSearchResultData(ctx, single, 1, topk, metric)
		if err != nil {
			return nil, err
		}
		out[i] = enc
	}
	return out, nil
}

// emptySingleQueryResult builds an nq=1 result with zero hits (a query no worker searched).
func emptySingleQueryResult(topk int64) *internalpb.SearchResults {
	return &internalpb.SearchResults{ResultData: &schemapb.SearchResultData{
		NumQueries: 1,
		TopK:       topk,
		Topks:      []int64{0},
		Scores:     []float32{},
		Ids:        &schemapb.IDs{},
	}}
}

func avgQueriesPerWorker(plans map[int64]*workerSearchPlan) float64 {
	if len(plans) == 0 {
		return 0
	}
	total := 0
	for _, wp := range plans {
		total += len(wp.queries)
	}
	return float64(total) / float64(len(plans))
}
