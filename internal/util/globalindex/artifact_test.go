package globalindex

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBuildStatsIndexPaths(t *testing.T) {
	paths := BuildStatsIndexPaths("/root", 100, 200, "by-dev-rootcoord-dml_0_100v0", 300)

	require.Equal(t, "/root/global_stats_index/100/200/by-dev-rootcoord-dml_0_100v0/300", paths.Root)
	require.Equal(t, "/root/global_stats_index/100/200/by-dev-rootcoord-dml_0_100v0/300/manifest", paths.Manifest)
	require.Equal(t, "/root/global_stats_index/100/200/by-dev-rootcoord-dml_0_100v0/300/head_index", paths.HeadIndex)
	require.Equal(t, "/root/global_stats_index/100/200/by-dev-rootcoord-dml_0_100v0/300/chunk_mapping", paths.ChunkMapping)
}

func TestBuildAnalyzeArtifactPaths(t *testing.T) {
	paths := BuildAnalyzeArtifactPaths("/root", 10, 20, 100, 200, 300)

	require.Equal(t, "/root/analyze_stats/10/20/100/200/300", paths.FieldRoot)
	require.Equal(t, "/root/analyze_stats/10/20/100/200/300/compaction_pre_plan", paths.CompactionPrePlan)
	require.Equal(t, "/root/analyze_stats/10/20/100/200/300/400/offset_mapping", paths.SegmentOffsetMapping(400))
	require.Equal(t, "/root/analyze_stats/10/20/100/200/300/400/offset_distance_mapping", paths.SegmentOffsetDistanceMapping(400))
}

func TestBuildStatsIndexPathsRejectsSlashInVChannel(t *testing.T) {
	require.Panics(t, func() {
		BuildStatsIndexPaths("/root", 100, 200, "bad/channel", 300)
	})
}

func TestManifestRoundTrip(t *testing.T) {
	manifest := Manifest{
		CollectionID:     100,
		PartitionID:      200,
		VChannel:         "by-dev-rootcoord-dml_0_100v0",
		Version:          300,
		FieldID:          400,
		Dim:              128,
		MetricType:       "L2",
		IndexType:        "IVF_FLAT",
		HeadIndexPath:    "head_index",
		ChunkMappingPath: "chunk_mapping",
	}

	data, err := json.Marshal(manifest)
	require.NoError(t, err)

	var decoded Manifest
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, manifest, decoded)
}

func TestChunkMappingValidate(t *testing.T) {
	mapping := ChunkMapping{
		10: {
			{SegmentID: 1000, Offset: 0, Size: 10},
			{SegmentID: 1001, Offset: 5, Size: 20},
		},
	}

	require.NoError(t, mapping.Validate())

	mapping[10] = append(mapping[10], Chunk{SegmentID: 1002, Offset: -1, Size: 1})
	require.Error(t, mapping.Validate())
}
