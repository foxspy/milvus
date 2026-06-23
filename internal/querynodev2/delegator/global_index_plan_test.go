package delegator

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/util/globalindex"
)

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
