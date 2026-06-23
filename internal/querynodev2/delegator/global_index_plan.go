package delegator

import (
	"github.com/milvus-io/milvus/internal/util/globalindex"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func ValidateGlobalIndexChunkPlan(mapping globalindex.ChunkMapping, sealedSegments []SnapshotItem, segmentRows map[int64]int64) error {
	if err := mapping.Validate(); err != nil {
		return err
	}

	visibleSegments := make(map[int64]struct{})
	for _, item := range sealedSegments {
		for _, segment := range item.Segments {
			if segment.Offline {
				continue
			}
			visibleSegments[segment.SegmentID] = struct{}{}
		}
	}

	for centroidID, chunks := range mapping {
		for _, chunk := range chunks {
			if _, ok := visibleSegments[chunk.SegmentID]; !ok {
				return merr.WrapErrServiceInternalMsg(
					"global index chunk references non-visible segment, centroidID=%d, segmentID=%d",
					centroidID,
					chunk.SegmentID,
				)
			}
			rowCount, ok := segmentRows[chunk.SegmentID]
			if !ok {
				return merr.WrapErrServiceInternalMsg(
					"global index chunk references segment without row count, centroidID=%d, segmentID=%d",
					centroidID,
					chunk.SegmentID,
				)
			}
			if chunk.Offset+chunk.Size > rowCount {
				return merr.WrapErrServiceInternalMsg(
					"global index chunk range exceeds segment rows, centroidID=%d, segmentID=%d, offset=%d, size=%d, rows=%d",
					centroidID,
					chunk.SegmentID,
					chunk.Offset,
					chunk.Size,
					rowCount,
				)
			}
		}
	}
	return nil
}
