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
	"errors"

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

// globalIndexSearchConcurrency caps the number of concurrent per-query sub-searches
// so a large nq does not overflow the querynode task scheduler queue.
const globalIndexSearchConcurrency = 256

// errGlobalIndexSearchInapplicable signals the dispatcher to fall back to the
// normal search path (e.g. head index returned unsupported for every stats index).
var errGlobalIndexSearchInapplicable = errors.New("global index search inapplicable")

// shouldUseGlobalIndexSearch decides whether the independent per-query head-index
// search path applies to this request. It is intentionally conservative: anything
// it cannot handle falls back to the normal search path.
func (sd *shardDelegator) shouldUseGlobalIndexSearch(req *querypb.SearchRequest) bool {
	if !paramtable.Get().QueryNodeCfg.GlobalIndexSearchEnabled.GetAsBool() {
		return false
	}
	r := req.GetReq()
	if r == nil || r.GetNq() <= 0 {
		return false
	}
	// Only plain ANN search; advanced/iterator/groupBy fall back.
	// Note: GroupByFieldId defaults to -1 (not 0) when there is no group-by.
	if r.GetIsAdvanced() || r.GetIsIterator() || r.GetGroupByFieldId() > 0 || len(r.GetGroupByFieldIds()) > 0 {
		return false
	}
	sd.globalStatsMut.RLock()
	defer sd.globalStatsMut.RUnlock()
	return len(sd.globalStatsIndexes) > 0
}

// globalIndexSearch runs an independent, per-query search path: each query uses the
// global head index to select its own target segments (no per-request union), then
// searches them independently. Per-query results are concatenated back into an nq=N
// result. This path does not touch the existing search / planGlobalStatsSearch flow.
func (sd *shardDelegator) globalIndexSearch(
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

	// 1. Per-query target segments from the head index.
	perQueryTargets, applied, err := sd.headIndexTargetsPerQuery(ctx, req, sealed, sealedRowCount)
	if err != nil {
		return nil, err
	}
	if !applied {
		return nil, errGlobalIndexSearchInapplicable
	}

	// 2. Parse the placeholder group into per-query slices for sub-requests.
	tag, phType, queryValues, err := splitPlaceholderGroup(r.GetPlaceholderGroup())
	if err != nil {
		return nil, err
	}
	if int64(len(queryValues)) != nq {
		return nil, merr.WrapErrServiceInternalMsg(
			"global index search placeholder count mismatch, nq=%d, placeholders=%d", nq, len(queryValues))
	}

	// 3. Execute each query independently and concurrently. Cap concurrency so a
	// large nq does not overwhelm the querynode task scheduler queue.
	perQueryResults := make([]*internalpb.SearchResults, nq)
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(globalIndexSearchConcurrency)
	for q := int64(0); q < nq; q++ {
		queryIdx := q
		g.Go(func() error {
			querySealed := buildSnapshotForSegments(sealed, perQueryTargets[queryIdx])
			subReq := sliceSearchRequestSingleQuery(req, tag, phType, queryValues[queryIdx])
			results, err := sd.executeSearchSubTasks(gctx, subReq, querySealed, growing, sealedRowCount)
			if err != nil {
				return err
			}
			res, err := segments.ReduceSearchOnQueryNode(gctx, results,
				reduce.NewReduceSearchResultInfo(1, topk).WithMetricType(metric))
			if err != nil {
				return err
			}
			if res.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
				return merr.Error(res.GetStatus())
			}
			perQueryResults[queryIdx] = res
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}

	// 4. Concatenate per-query (nq=1) results into a single nq=N result.
	merged, err := concatPerQueryResults(ctx, perQueryResults, nq, topk, metric)
	if err != nil {
		return nil, err
	}

	if log.Core().Enabled(zap.DebugLevel) {
		totalSearched := make(map[int64]struct{})
		for _, t := range perQueryTargets {
			for segID := range t {
				totalSearched[segID] = struct{}{}
			}
		}
		log.Debug("global index per-query search done",
			zap.Int64("nq", nq),
			zap.Int("sealedTotal", countSegments(sealed)),
			zap.Int("unionSearchedSegments", len(totalSearched)),
			zap.Float64("avgSegmentsPerQuery", avgTargets(perQueryTargets)))
	}
	return []*internalpb.SearchResults{merged}, nil
}

// headIndexTargetsPerQuery returns, for each query, the set of sealed segment IDs to
// search: the head-index-selected segments for that query, plus any sealed segment not
// covered by a global stats index (passthrough, to avoid missing data). applied is
// false when no stats index could serve the request (caller should fall back).
func (sd *shardDelegator) headIndexTargetsPerQuery(
	ctx context.Context,
	req *querypb.SearchRequest,
	sealed []SnapshotItem,
	sealedRowCount map[int64]int64,
) ([]map[int64]struct{}, bool, error) {
	nq := req.GetReq().GetNq()
	targets := make([]map[int64]struct{}, nq)
	for i := range targets {
		targets[i] = make(map[int64]struct{})
	}

	nprobe := paramtable.Get().QueryNodeCfg.GlobalIndexSearchNprobe.GetAsInt64()
	if nprobe <= 0 {
		nprobe = req.GetReq().GetTopk() + req.GetReq().GetOffset()
	}
	if nprobe <= 0 {
		nprobe = 1
	}

	sd.globalStatsMut.RLock()
	statsIndexes := make([]*loadedGlobalStatsIndex, 0, len(sd.globalStatsIndexes))
	for _, si := range sd.globalStatsIndexes {
		if si != nil && si.headIndexSearcher != nil {
			statsIndexes = append(statsIndexes, si)
		}
	}
	segmentCovered := make(map[int64]struct{}, len(sd.segmentGlobalStats))
	for segID, root := range sd.segmentGlobalStats {
		if root != "" {
			segmentCovered[segID] = struct{}{}
		}
	}
	sd.globalStatsMut.RUnlock()

	applied := false
	for _, si := range statsIndexes {
		centroidsPerQuery, err := si.headIndexSearcher.Search(ctx, req.GetReq(), nprobe)
		if errors.Is(err, errHeadIndexSearchUnsupported) {
			continue
		}
		if err != nil {
			return nil, false, merr.Wrap(err, "global index search head index")
		}
		if int64(len(centroidsPerQuery)) != nq {
			return nil, false, merr.WrapErrServiceInternalMsg(
				"head index returned %d query rows, expected nq=%d", len(centroidsPerQuery), nq)
		}
		applied = true
		for q, centroidIDs := range centroidsPerQuery {
			for _, centroidID := range centroidIDs {
				for _, chunk := range si.chunkMapping[centroidID] {
					if _, ok := sealedRowCount[chunk.SegmentID]; ok {
						targets[q][chunk.SegmentID] = struct{}{}
					}
				}
			}
		}
	}
	if !applied {
		return nil, false, nil
	}

	// Passthrough: sealed segments not covered by any global stats index must be
	// searched by every query, otherwise newly-flushed data would be missed.
	for _, item := range sealed {
		for _, seg := range item.Segments {
			if seg.Offline {
				continue
			}
			if _, ok := segmentCovered[seg.SegmentID]; ok {
				continue
			}
			for q := int64(0); q < nq; q++ {
				targets[q][seg.SegmentID] = struct{}{}
			}
		}
	}
	return targets, true, nil
}

// buildSnapshotForSegments rebuilds SnapshotItems keeping only the given segment IDs,
// preserving the original NodeID grouping.
func buildSnapshotForSegments(sealed []SnapshotItem, segIDs map[int64]struct{}) []SnapshotItem {
	out := make([]SnapshotItem, 0, len(sealed))
	for _, item := range sealed {
		segs := make([]SegmentEntry, 0, len(item.Segments))
		for _, seg := range item.Segments {
			if _, ok := segIDs[seg.SegmentID]; ok {
				segs = append(segs, seg)
			}
		}
		if len(segs) > 0 {
			out = append(out, SnapshotItem{NodeID: item.NodeID, Segments: segs})
		}
	}
	return out
}

// splitPlaceholderGroup decodes a serialized placeholder group and returns the tag,
// type and per-query raw values (one []byte per query vector).
func splitPlaceholderGroup(phgBytes []byte) (string, commonpb.PlaceholderType, [][]byte, error) {
	group := &commonpb.PlaceholderGroup{}
	if err := proto.Unmarshal(phgBytes, group); err != nil {
		return "", commonpb.PlaceholderType_None, nil, merr.WrapErrParameterInvalidMsg("invalid placeholder group: %v", err)
	}
	if len(group.GetPlaceholders()) == 0 {
		return "", commonpb.PlaceholderType_None, nil, merr.WrapErrParameterInvalidMsg("empty placeholder group")
	}
	ph := group.GetPlaceholders()[0]
	return ph.GetTag(), ph.GetType(), ph.GetValues(), nil
}

// sliceSearchRequestSingleQuery clones req into an nq=1 sub-request carrying only the
// given query vector value.
func sliceSearchRequestSingleQuery(req *querypb.SearchRequest, tag string, phType commonpb.PlaceholderType, value []byte) *querypb.SearchRequest {
	phgBytes, _ := proto.Marshal(&commonpb.PlaceholderGroup{
		Placeholders: []*commonpb.PlaceholderValue{{
			Tag:    tag,
			Type:   phType,
			Values: [][]byte{value},
		}},
	})
	newReq := proto.Clone(req.GetReq()).(*internalpb.SearchRequest)
	newReq.Nq = 1
	newReq.PlaceholderGroup = phgBytes
	return &querypb.SearchRequest{
		Req:             newReq,
		DmlChannels:     req.GetDmlChannels(),
		TotalChannelNum: req.GetTotalChannelNum(),
	}
}

// concatPerQueryResults stitches nq independent nq=1 results into a single nq=N result.
func concatPerQueryResults(ctx context.Context, perQuery []*internalpb.SearchResults, nq, topk int64, metric string) (*internalpb.SearchResults, error) {
	merged := &schemapb.SearchResultData{
		NumQueries: nq,
		TopK:       topk,
		Scores:     make([]float32, 0),
		Ids:        &schemapb.IDs{},
		Topks:      make([]int64, 0, nq),
	}
	var fieldSample []*schemapb.FieldData
	for _, res := range perQuery {
		datas, err := segments.DecodeSearchResults(ctx, []*internalpb.SearchResults{res})
		if err != nil {
			return nil, err
		}
		if len(datas) == 0 || datas[0] == nil {
			merged.Topks = append(merged.Topks, 0)
			continue
		}
		data := datas[0]
		if fieldSample == nil && len(data.GetFieldsData()) > 0 {
			fieldSample = data.GetFieldsData()
			merged.FieldsData = typeutil.PrepareResultFieldData(fieldSample, topk*nq)
		}
		var hit int64
		if len(data.GetTopks()) > 0 {
			hit = data.GetTopks()[0]
		}
		merged.Topks = append(merged.Topks, hit)
		merged.Scores = append(merged.Scores, data.GetScores()...)
		for row := int64(0); row < hit; row++ {
			typeutil.AppendIDs(merged.Ids, data.GetIds(), int(row))
			if len(merged.FieldsData) > 0 {
				typeutil.AppendFieldData(merged.FieldsData, data.GetFieldsData(), row)
			}
		}
	}
	return segments.EncodeSearchResultData(ctx, merged, nq, topk, metric)
}

func countSegments(sealed []SnapshotItem) int {
	n := 0
	for _, item := range sealed {
		n += len(item.Segments)
	}
	return n
}

func avgTargets(targets []map[int64]struct{}) float64 {
	if len(targets) == 0 {
		return 0
	}
	total := 0
	for _, t := range targets {
		total += len(t)
	}
	return float64(total) / float64(len(targets))
}
