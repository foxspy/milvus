package delegator

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/globalindex"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

type fakeHeadIndexSearcher struct {
	centroidIDs [][]int64
}

func (s fakeHeadIndexSearcher) Search(ctx context.Context, req *internalpb.SearchRequest, topK int64) ([][]centroidHit, error) {
	result := make([][]centroidHit, len(s.centroidIDs))
	for q, ids := range s.centroidIDs {
		for _, id := range ids {
			result[q] = append(result[q], centroidHit{centroidID: id, distance: float32(id)})
		}
	}
	return result, nil
}

func TestParseHeadIndexSearchFloatVectors(t *testing.T) {
	vector := func(values ...float32) []byte {
		buf := make([]byte, len(values)*4)
		for i, value := range values {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(value))
		}
		return buf
	}
	group, err := proto.Marshal(&commonpb.PlaceholderGroup{
		Placeholders: []*commonpb.PlaceholderValue{{
			Tag:    "$0",
			Type:   commonpb.PlaceholderType_FloatVector,
			Values: [][]byte{vector(1, 2), vector(3, 4)},
		}},
	})
	require.NoError(t, err)

	vectors, dim, ok, err := parseHeadIndexSearchFloatVectors(&internalpb.SearchRequest{
		PlaceholderGroup: group,
	})
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, int64(2), dim)
	require.Equal(t, []float32{1, 2, 3, 4}, vectors)
}

func TestValidateGlobalIndexChunkPlan(t *testing.T) {
	sealed := []SnapshotItem{
		{
			NodeID: 1,
			Segments: []SegmentEntry{
				{NodeID: 1, SegmentID: 100},
				{NodeID: 1, SegmentID: 101, Offline: true},
			},
		},
	}
	rows := map[int64]int64{
		100: 10,
		101: 10,
	}

	require.NoError(t, ValidateGlobalIndexChunkPlan(globalindex.ChunkMapping{
		1: {{SegmentID: 100, Offset: 2, Size: 3}},
	}, sealed, rows))

	require.Error(t, ValidateGlobalIndexChunkPlan(globalindex.ChunkMapping{
		1: {{SegmentID: 102, Offset: 0, Size: 1}},
	}, sealed, rows))

	require.Error(t, ValidateGlobalIndexChunkPlan(globalindex.ChunkMapping{
		1: {{SegmentID: 101, Offset: 0, Size: 1}},
	}, sealed, rows))

	require.Error(t, ValidateGlobalIndexChunkPlan(globalindex.ChunkMapping{
		1: {{SegmentID: 100, Offset: 8, Size: 3}},
	}, sealed, rows))
}

func TestLoadGlobalStatsIndex(t *testing.T) {
	ctx := context.Background()
	rootPath := t.TempDir()
	chunkManagerFactory := storage.NewTestChunkManagerFactory(paramtable.Get(), rootPath)
	chunkManager, err := chunkManagerFactory.NewPersistentStorageChunkManager(ctx)
	require.NoError(t, err)

	root := "/root/global_stats_index/1/2/ch/3"
	headIndexFile := root + "/" + common.GlobalStatsHeadIndexPath
	chunkMappingFile := root + "/" + common.GlobalStatsChunkMapping
	require.NoError(t, chunkManager.Write(ctx, headIndexFile, []byte("head-index-bytes")))
	oldNewHeadIndexSearcherFromPath := newHeadIndexSearcherFromPath
	newHeadIndexSearcherFromPath = func(path string) (headIndexSearcher, error) {
		require.Equal(t, headIndexFile, path)
		return fakeHeadIndexSearcher{}, nil
	}
	defer func() {
		newHeadIndexSearcherFromPath = oldNewHeadIndexSearcherFromPath
	}()
	mappingBytes, err := json.Marshal(globalindex.ChunkMapping{
		10: {{SegmentID: 100, Offset: 0, Size: 5}},
	})
	require.NoError(t, err)
	require.NoError(t, chunkManager.Write(ctx, chunkMappingFile, mappingBytes))

	sd := &shardDelegator{
		collectionID:       1,
		vchannelName:       "ch",
		chunkManager:       chunkManager,
		globalStatsIndexes: make(map[string]*loadedGlobalStatsIndex),
		segmentGlobalStats: make(map[int64]string),
	}

	err = sd.loadGlobalStatsIndexes(ctx, []*querypb.SegmentLoadInfo{{
		SegmentID:            100,
		GlobalStatsIndexRoot: root,
		HeadIndexFile:        headIndexFile,
		ChunkMappingFile:     chunkMappingFile,
	}})
	require.NoError(t, err)

	loaded := sd.getLoadedGlobalStatsIndex(root, headIndexFile, chunkMappingFile)
	require.NotNil(t, loaded)
	require.Equal(t, headIndexFile, loaded.headIndexFile)
	require.NotNil(t, loaded.headIndexSearcher)
	require.Equal(t, root, sd.segmentGlobalStats[100])
	require.NoError(t, sd.validateGlobalStatsChunkMappings(
		[]string{root},
		[]SnapshotItem{{NodeID: 1, Segments: []SegmentEntry{{NodeID: 1, SegmentID: 100}}}},
		map[int64]int64{100: 5},
	))

	require.Error(t, sd.validateGlobalStatsChunkMappings(
		[]string{root},
		[]SnapshotItem{{NodeID: 1, Segments: []SegmentEntry{{NodeID: 1, SegmentID: 101}}}},
		map[int64]int64{101: 5},
	))
}

func TestPlanGlobalStatsSearchDispatchesHeadIndexCentroidsToSegments(t *testing.T) {
	root := "/root/global_stats_index/1/2/ch/3"
	sd := &shardDelegator{
		collectionID: 1,
		vchannelName: "ch",
		globalStatsIndexes: map[string]*loadedGlobalStatsIndex{
			root: {
				root: root,
				chunkMapping: globalindex.ChunkMapping{
					10: {{SegmentID: 100, Offset: 0, Size: 3}},
					20: {{SegmentID: 102, Offset: 5, Size: 4}},
					30: {{SegmentID: 103, Offset: 0, Size: 1}},
				},
				headIndexSearcher: fakeHeadIndexSearcher{centroidIDs: [][]int64{{10, 20}}},
			},
		},
		segmentGlobalStats: map[int64]string{
			100: root,
			101: root,
			102: root,
			103: root,
		},
	}

	sealed := []SnapshotItem{
		{NodeID: 1, Segments: []SegmentEntry{
			{NodeID: 1, SegmentID: 100},
			{NodeID: 1, SegmentID: 101},
		}},
		{NodeID: 2, Segments: []SegmentEntry{
			{NodeID: 2, SegmentID: 102},
			{NodeID: 2, SegmentID: 103},
		}},
	}

	planned, applied, err := sd.planGlobalStatsSearch(context.Background(), &querypb.SearchRequest{
		Req: &internalpb.SearchRequest{Topk: 2},
	}, sealed, map[int64]int64{100: 10, 101: 10, 102: 10, 103: 10})
	require.NoError(t, err)
	require.True(t, applied)
	require.Len(t, planned, 2)
	require.Equal(t, int64(1), planned[0].NodeID)
	require.Equal(t, []SegmentEntry{{NodeID: 1, SegmentID: 100}}, planned[0].Segments)
	require.Equal(t, int64(2), planned[1].NodeID)
	require.Equal(t, []SegmentEntry{{NodeID: 2, SegmentID: 102}}, planned[1].Segments)
}
