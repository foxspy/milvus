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
	"github.com/milvus-io/milvus/internal/util/globalindex"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

// unsupportedHeadIndexSearcher mimics a non-float-vector head index that cannot serve
// the request (the global_index_plan_test.go fakeHeadIndexSearcher always succeeds).
type unsupportedHeadIndexSearcher struct{}

func (unsupportedHeadIndexSearcher) Search(ctx context.Context, req *internalpb.SearchRequest, topK int64) ([][]centroidHit, error) {
	return nil, errHeadIndexSearchUnsupported
}

func newTestDelegatorWithGlobalStats() *shardDelegator {
	return &shardDelegator{
		globalStatsIndexes: map[string]*loadedGlobalStatsIndex{
			"root1": {
				root: "root1",
				chunkMapping: globalindex.ChunkMapping{
					1: {{SegmentID: 10, Offset: 3, Size: 5}},
					2: {{SegmentID: 20, Offset: 7, Size: 5}},
				},
				headIndexSearcher: fakeHeadIndexSearcher{centroidIDs: [][]int64{{1}, {2}}},
			},
		},
		segmentGlobalStats: map[int64]string{10: "root1", 20: "root1"},
	}
}

// Each query selects its own segments (no per-request union); uncovered segments
// (seg 30) pass through to every query.
func TestGlobalIndexSearch_TargetsPerQuery(t *testing.T) {
	paramtable.Init()
	sd := newTestDelegatorWithGlobalStats()
	req := &querypb.SearchRequest{Req: &internalpb.SearchRequest{Nq: 2, Topk: 10}}
	sealed := []SnapshotItem{{NodeID: 1, Segments: []SegmentEntry{
		{SegmentID: 10}, {SegmentID: 20}, {SegmentID: 30},
	}}}
	sealedRowCount := map[int64]int64{10: 5, 20: 5, 30: 5}

	targets, applied, err := sd.headIndexTargetsPerQuery(context.Background(), req, sealed, sealedRowCount)
	require.NoError(t, err)
	require.True(t, applied)
	require.Len(t, targets, 2)

	// query 0: head-index segment 10 (hint: offset 3, range 5, dist=centroid 1) + passthrough 30 (no hint)
	require.Len(t, targets[0], 2)
	require.Len(t, targets[0][10], 1)
	assert.Equal(t, searchHint{idOffset: 3, idRange: 5, idDistance: 1}, targets[0][10][0])
	_, has30 := targets[0][30]
	assert.True(t, has30)
	assert.Nil(t, targets[0][30])

	// query 1: head-index segment 20 (hint: offset 7, range 5, dist=centroid 2) + passthrough 30
	require.Len(t, targets[1], 2)
	require.Len(t, targets[1][20], 1)
	assert.Equal(t, searchHint{idOffset: 7, idRange: 5, idDistance: 2}, targets[1][20][0])

	// Critical: the two queries do NOT share a unioned segment set.
	assert.NotContains(t, targets[0], int64(20))
	assert.NotContains(t, targets[1], int64(10))
}

func TestGlobalIndexSearch_TargetsFallbackWhenUnsupported(t *testing.T) {
	paramtable.Init()
	sd := newTestDelegatorWithGlobalStats()
	sd.globalStatsIndexes["root1"].headIndexSearcher = unsupportedHeadIndexSearcher{}
	req := &querypb.SearchRequest{Req: &internalpb.SearchRequest{Nq: 2, Topk: 10}}
	sealed := []SnapshotItem{{NodeID: 1, Segments: []SegmentEntry{{SegmentID: 10}}}}

	_, applied, err := sd.headIndexTargetsPerQuery(context.Background(), req, sealed, map[int64]int64{10: 5})
	require.NoError(t, err)
	assert.False(t, applied) // caller should fall back to normal search
}

func TestGlobalIndexSearch_ShouldUse(t *testing.T) {
	paramtable.Init()
	sd := newTestDelegatorWithGlobalStats()
	req := &querypb.SearchRequest{Req: &internalpb.SearchRequest{Nq: 2, Topk: 10}}

	// flag off -> false
	paramtable.Get().Save(paramtable.Get().QueryNodeCfg.GlobalIndexSearchEnabled.Key, "false")
	assert.False(t, sd.shouldUseGlobalIndexSearch(req))

	// flag on -> true
	paramtable.Get().Save(paramtable.Get().QueryNodeCfg.GlobalIndexSearchEnabled.Key, "true")
	assert.True(t, sd.shouldUseGlobalIndexSearch(req))

	// groupByFieldId = -1 means "no group-by" (the real proxy default), must still use the path
	noGroupReq := &querypb.SearchRequest{Req: &internalpb.SearchRequest{Nq: 2, Topk: 10, GroupByFieldId: -1}}
	assert.True(t, sd.shouldUseGlobalIndexSearch(noGroupReq))

	// real group-by (positive field id) -> false
	groupReq := &querypb.SearchRequest{Req: &internalpb.SearchRequest{Nq: 2, Topk: 10, GroupByFieldId: 101}}
	assert.False(t, sd.shouldUseGlobalIndexSearch(groupReq))

	// advanced search -> false
	advReq := &querypb.SearchRequest{Req: &internalpb.SearchRequest{Nq: 2, Topk: 10, IsAdvanced: true}}
	assert.False(t, sd.shouldUseGlobalIndexSearch(advReq))

	// no stats loaded -> false
	empty := &shardDelegator{globalStatsIndexes: map[string]*loadedGlobalStatsIndex{}}
	assert.False(t, empty.shouldUseGlobalIndexSearch(req))
	paramtable.Get().Reset(paramtable.Get().QueryNodeCfg.GlobalIndexSearchEnabled.Key)
}

func TestGlobalIndexSearch_SlicePlaceholder(t *testing.T) {
	vec0 := []byte{1, 2, 3, 4}
	vec1 := []byte{5, 6, 7, 8}
	phgBytes, err := proto.Marshal(&commonpb.PlaceholderGroup{
		Placeholders: []*commonpb.PlaceholderValue{{
			Tag:    "$0",
			Type:   commonpb.PlaceholderType_FloatVector,
			Values: [][]byte{vec0, vec1},
		}},
	})
	require.NoError(t, err)

	tag, phType, values, err := splitPlaceholderGroup(phgBytes)
	require.NoError(t, err)
	require.Len(t, values, 2)
	assert.Equal(t, "$0", tag)
	assert.Equal(t, commonpb.PlaceholderType_FloatVector, phType)

	req := &querypb.SearchRequest{Req: &internalpb.SearchRequest{Nq: 2, PlaceholderGroup: phgBytes}}
	plan := queryPlan{ // seg 10 has hints, seg 20 passthrough
		10: []searchHint{{idOffset: 3, idRange: 5, idDistance: 1.5}, {idOffset: 8, idRange: 2, idDistance: 2.5}},
		20: nil,
	}
	sub := sliceSearchRequestSingleQuery(req, tag, phType, values[1], plan)
	assert.Equal(t, int64(1), sub.GetReq().GetNq())

	// search hints attached only for segments with hints
	hints := sub.GetReq().GetSegmentSearchHints()
	require.Contains(t, hints, int64(10))
	require.Len(t, hints[10].GetHints(), 2)
	assert.Equal(t, int64(3), hints[10].GetHints()[0].GetIdOffset())
	assert.Equal(t, int64(5), hints[10].GetHints()[0].GetIdRange())
	assert.Equal(t, float32(1.5), hints[10].GetHints()[0].GetIdDistance())
	assert.NotContains(t, hints, int64(20))

	subGroup := &commonpb.PlaceholderGroup{}
	require.NoError(t, proto.Unmarshal(sub.GetReq().GetPlaceholderGroup(), subGroup))
	require.Len(t, subGroup.GetPlaceholders(), 1)
	require.Len(t, subGroup.GetPlaceholders()[0].GetValues(), 1)
	assert.Equal(t, vec1, subGroup.GetPlaceholders()[0].GetValues()[0])
}

func TestGlobalIndexSearch_Concat(t *testing.T) {
	paramtable.Init()
	intIDs := func(ids ...int64) *schemapb.IDs {
		return &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: ids}}}
	}
	res0 := &internalpb.SearchResults{ResultData: &schemapb.SearchResultData{
		NumQueries: 1, TopK: 10, Topks: []int64{2}, Scores: []float32{0.9, 0.8}, Ids: intIDs(100, 101),
	}}
	res1 := &internalpb.SearchResults{ResultData: &schemapb.SearchResultData{
		NumQueries: 1, TopK: 10, Topks: []int64{1}, Scores: []float32{0.7}, Ids: intIDs(200),
	}}

	merged, err := concatPerQueryResults(context.Background(), []*internalpb.SearchResults{res0, res1}, 2, 10, "L2")
	require.NoError(t, err)

	datas, err := segments.DecodeSearchResults(context.Background(), []*internalpb.SearchResults{merged})
	require.NoError(t, err)
	require.Len(t, datas, 1)
	d := datas[0]
	assert.Equal(t, int64(2), d.GetNumQueries())
	assert.Equal(t, []int64{2, 1}, d.GetTopks())
	assert.Equal(t, []float32{0.9, 0.8, 0.7}, d.GetScores())
	assert.Equal(t, []int64{100, 101, 200}, d.GetIds().GetIntId().GetData())
}
