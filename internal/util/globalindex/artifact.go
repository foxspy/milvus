package globalindex

import (
	"fmt"
	"path"
	"strconv"
	"strings"

	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

type StatsIndexPaths struct {
	Root         string
	Manifest     string
	HeadIndex    string
	ChunkMapping string
}

type AnalyzeArtifactPaths struct {
	Root              string
	FieldRoot         string
	CompactionPrePlan string
}

type Manifest struct {
	CollectionID     int64  `json:"collection_id"`
	PartitionID      int64  `json:"partition_id"`
	VChannel         string `json:"vchannel"`
	Version          int64  `json:"version"`
	FieldID          int64  `json:"field_id"`
	Dim              int64  `json:"dim"`
	MetricType       string `json:"metric_type"`
	IndexType        string `json:"index_type"`
	HeadIndexPath    string `json:"head_index_path"`
	ChunkMappingPath string `json:"chunk_mapping_path"`
}

type Chunk struct {
	SegmentID int64 `json:"segment_id"`
	Offset    int64 `json:"offset"`
	Size      int64 `json:"size"`
}

type ChunkMapping map[int64][]Chunk

func BuildStatsIndexPaths(rootPath string, collectionID, partitionID int64, vchannel string, version int64) StatsIndexPaths {
	if strings.Contains(vchannel, "/") {
		panic(fmt.Sprintf("vchannel must not contain slash: %s", vchannel))
	}
	root := path.Join(rootPath,
		common.GlobalStatsIndexPath,
		strconv.FormatInt(collectionID, 10),
		strconv.FormatInt(partitionID, 10),
		vchannel,
		strconv.FormatInt(version, 10),
	)
	return StatsIndexPaths{
		Root:         root,
		Manifest:     path.Join(root, common.GlobalStatsManifest),
		HeadIndex:    path.Join(root, common.GlobalStatsHeadIndexPath),
		ChunkMapping: path.Join(root, common.GlobalStatsChunkMapping),
	}
}

func BuildAnalyzeArtifactPaths(rootPath string, analyzeTaskID, version, collectionID, partitionID, fieldID int64) AnalyzeArtifactPaths {
	root := path.Join(rootPath,
		common.AnalyzeStatsPath,
		strconv.FormatInt(analyzeTaskID, 10),
		strconv.FormatInt(version, 10),
	)
	fieldRoot := path.Join(root,
		strconv.FormatInt(collectionID, 10),
		strconv.FormatInt(partitionID, 10),
		strconv.FormatInt(fieldID, 10),
	)
	return AnalyzeArtifactPaths{
		Root:              root,
		FieldRoot:         fieldRoot,
		CompactionPrePlan: path.Join(fieldRoot, common.CompactionPrePlan),
	}
}

func (p AnalyzeArtifactPaths) SegmentOffsetMapping(segmentID int64) string {
	return path.Join(p.FieldRoot, strconv.FormatInt(segmentID, 10), common.OffsetMapping)
}

func (p AnalyzeArtifactPaths) SegmentOffsetDistanceMapping(segmentID int64) string {
	return path.Join(p.FieldRoot, strconv.FormatInt(segmentID, 10), common.OffsetDistanceMapping)
}

func (m ChunkMapping) Validate() error {
	for centroidID, chunks := range m {
		if centroidID < 0 {
			return merr.WrapErrServiceInternalMsg("centroid id must be non-negative: %d", centroidID)
		}
		for _, chunk := range chunks {
			if chunk.SegmentID <= 0 {
				return merr.WrapErrServiceInternalMsg("segment id must be positive: %d", chunk.SegmentID)
			}
			if chunk.Offset < 0 {
				return merr.WrapErrServiceInternalMsg("chunk offset must be non-negative: %d", chunk.Offset)
			}
			if chunk.Size <= 0 {
				return merr.WrapErrServiceInternalMsg("chunk size must be positive: %d", chunk.Size)
			}
		}
	}
	return nil
}
