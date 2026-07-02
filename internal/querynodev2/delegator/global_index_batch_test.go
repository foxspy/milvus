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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

// Two queries, two workers: the per-query targets invert into one plan per worker,
// each carrying only the queries whose segments live on it, with segment x query tasks.
func TestBuildWorkerSearchPlans_Inversion(t *testing.T) {
	perQuery := []queryPlan{
		{ // query 0
			10: []searchHint{{idOffset: 3, idRange: 5, idDistance: 1}},
			30: nil, // passthrough on worker 2
		},
		{ // query 1
			20: []searchHint{{idOffset: 7, idRange: 5, idDistance: 2}},
			30: nil, // passthrough on worker 2
		},
	}
	sealed := []SnapshotItem{
		{NodeID: 1, Segments: []SegmentEntry{{SegmentID: 10}, {SegmentID: 20}}},
		{NodeID: 2, Segments: []SegmentEntry{{SegmentID: 30}}},
	}

	plans := buildWorkerSearchPlans(perQuery, sealed)
	require.Len(t, plans, 2)

	w1 := plans[1]
	require.NotNil(t, w1)
	assert.Equal(t, []int64{0, 1}, w1.queries)
	assert.Equal(t, int32(0), w1.localIndex[0])
	assert.Equal(t, int32(1), w1.localIndex[1])
	require.Len(t, w1.segTasks[10], 1)
	assert.Equal(t, int64(0), w1.segTasks[10][0].globalQuery)
	assert.Equal(t, searchHint{idOffset: 3, idRange: 5, idDistance: 1}, w1.segTasks[10][0].hints[0])
	require.Len(t, w1.segTasks[20], 1)
	assert.Equal(t, int64(1), w1.segTasks[20][0].globalQuery)

	w2 := plans[2]
	require.NotNil(t, w2)
	assert.Equal(t, []int64{0, 1}, w2.queries)
	require.Len(t, w2.segTasks[30], 2) // both queries pass through seg 30
	assert.Nil(t, w2.segTasks[30][0].hints)
}

// A worker with no targeted segment gets no plan (and therefore no RPC).
func TestBuildWorkerSearchPlans_WorkerPruned(t *testing.T) {
	perQuery := []queryPlan{
		{10: []searchHint{{idOffset: 0, idRange: 1, idDistance: 1}}},
	}
	sealed := []SnapshotItem{
		{NodeID: 1, Segments: []SegmentEntry{{SegmentID: 10}}},
		{NodeID: 2, Segments: []SegmentEntry{{SegmentID: 30}}}, // never targeted
	}
	plans := buildWorkerSearchPlans(perQuery, sealed)
	require.Len(t, plans, 1)
	assert.NotNil(t, plans[1])
	assert.Nil(t, plans[2]) // pruned: no RPC to worker 2
}

// The batched request carries the worker's M queries in the placeholder group and marks,
// per segment, which local queries search it plus their entry-point hints.
func TestBuildWorkerBatchedRequest_Marking(t *testing.T) {
	paramtable.Init()
	paramtable.Get().Save(paramtable.Get().QueryNodeCfg.GlobalIndexSearchEntryPoints.Key, "true")
	defer paramtable.Get().Reset(paramtable.Get().QueryNodeCfg.GlobalIndexSearchEntryPoints.Key)

	req := &querypb.SearchRequest{Req: &internalpb.SearchRequest{Nq: 3, Topk: 10}}
	vec0, vec1, vec2 := []byte{0, 0, 0, 1}, []byte{0, 0, 0, 2}, []byte{0, 0, 0, 3}
	queryValues := [][]byte{vec0, vec1, vec2}

	wp := &workerSearchPlan{
		workerID:   1,
		queries:    []int64{0, 2}, // only queries 0 and 2 hit this worker
		localIndex: map[int64]int32{0: 0, 2: 1},
		segTasks: map[int64][]segQueryTask{
			10: {{globalQuery: 0, hints: []searchHint{{idOffset: 3, idRange: 5, idDistance: 1.5}}}},
			30: {{globalQuery: 0}, {globalQuery: 2}}, // passthrough, no hints
		},
	}

	sub := buildWorkerBatchedRequest(req, "$0", commonpb.PlaceholderType_FloatVector, queryValues, wp)
	assert.Equal(t, int64(2), sub.GetReq().GetNq())

	group := &commonpb.PlaceholderGroup{}
	require.NoError(t, proto.Unmarshal(sub.GetReq().GetPlaceholderGroup(), group))
	values := group.GetPlaceholders()[0].GetValues()
	require.Len(t, values, 2)
	assert.Equal(t, vec0, values[0]) // local 0 -> global 0
	assert.Equal(t, vec2, values[1]) // local 1 -> global 2

	tasks := sub.GetReq().GetSegmentQueryTasks()
	require.Contains(t, tasks, int64(10))
	require.Len(t, tasks[10].GetTasks(), 1)
	assert.Equal(t, int32(0), tasks[10].GetTasks()[0].GetQueryIndex()) // global 0 -> local 0
	require.Len(t, tasks[10].GetTasks()[0].GetHints().GetHints(), 1)
	assert.Equal(t, int64(3), tasks[10].GetTasks()[0].GetHints().GetHints()[0].GetIdOffset())

	require.Contains(t, tasks, int64(30))
	require.Len(t, tasks[30].GetTasks(), 2)
	// global 2 mapped to local index 1
	assert.Equal(t, int32(1), tasks[30].GetTasks()[1].GetQueryIndex())
	assert.Nil(t, tasks[30].GetTasks()[0].GetHints()) // passthrough
}

// splitResultPerQuery decodes an nq=M result and re-emits it as M independent nq=1
// results, preserving per-query ids/scores/topks.
func TestSplitResultPerQuery(t *testing.T) {
	paramtable.Init()
	intIDs := func(ids ...int64) *schemapb.IDs {
		return &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: ids}}}
	}
	src := &internalpb.SearchResults{ResultData: &schemapb.SearchResultData{
		NumQueries: 2, TopK: 10, Topks: []int64{2, 1},
		Scores: []float32{0.9, 0.8, 0.7}, Ids: intIDs(100, 101, 200),
	}}

	slices, err := splitResultPerQuery(context.Background(), src, 2, 10, "L2")
	require.NoError(t, err)
	require.Len(t, slices, 2)

	datas, err := segments.DecodeSearchResults(context.Background(), slices)
	require.NoError(t, err)
	require.Len(t, datas, 2)
	assert.Equal(t, []int64{2}, datas[0].GetTopks())
	assert.Equal(t, []int64{100, 101}, datas[0].GetIds().GetIntId().GetData())
	assert.Equal(t, []float32{0.9, 0.8}, datas[0].GetScores())
	assert.Equal(t, []int64{1}, datas[1].GetTopks())
	assert.Equal(t, []int64{200}, datas[1].GetIds().GetIntId().GetData())
}
