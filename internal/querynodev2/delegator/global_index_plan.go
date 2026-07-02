package delegator

import (
	"context"
	"errors"

	"github.com/milvus-io/milvus/internal/util/globalindex"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
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

func (sd *shardDelegator) planGlobalStatsSearch(
	ctx context.Context,
	req *querypb.SearchRequest,
	sealed []SnapshotItem,
	sealedRowCount map[int64]int64,
) ([]SnapshotItem, bool, error) {
	roots := sd.globalStatsRootsForSealedSegments(sealed)
	if len(roots) == 0 {
		return sealed, false, nil
	}

	statsIndexes := make([]*loadedGlobalStatsIndex, 0, len(roots))
	sd.globalStatsMut.RLock()
	for _, root := range roots {
		statsIndex := sd.globalStatsIndexes[root]
		if statsIndex != nil {
			statsIndexes = append(statsIndexes, statsIndex)
		}
	}
	sd.globalStatsMut.RUnlock()
	if len(statsIndexes) == 0 {
		return sealed, false, nil
	}

	// nprobe (number of nearest centroids to probe in the head index) is decoupled
	// from the query topk: pruning recall is governed by how many centroids we probe,
	// not by how many results the caller asked for. When unset (<=0), fall back to the
	// legacy topk+offset behavior for backward compatibility.
	topK := paramtable.Get().QueryNodeCfg.GlobalIndexSearchNprobe.GetAsInt64()
	if topK <= 0 {
		topK = req.GetReq().GetTopk() + req.GetReq().GetOffset()
	}
	if topK <= 0 {
		topK = 1
	}

	targetSegments := make(map[int64]struct{})
	applied := false
	for _, statsIndex := range statsIndexes {
		if statsIndex.headIndexSearcher == nil {
			continue
		}
		centroidIDsPerQuery, err := statsIndex.headIndexSearcher.Search(ctx, req.GetReq(), topK)
		if errors.Is(err, errHeadIndexSearchUnsupported) {
			continue
		}
		if err != nil {
			return nil, false, merr.Wrap(err, "search global head index")
		}
		applied = true
		for _, hits := range centroidIDsPerQuery {
			for _, hit := range hits {
				for _, chunk := range statsIndex.chunkMapping[hit.centroidID] {
					if _, ok := sealedRowCount[chunk.SegmentID]; ok {
						targetSegments[chunk.SegmentID] = struct{}{}
					}
				}
			}
		}
	}
	if !applied {
		return sealed, false, nil
	}

	planned := make([]SnapshotItem, 0, len(sealed))
	for _, item := range sealed {
		segments := make([]SegmentEntry, 0, len(item.Segments))
		for _, segment := range item.Segments {
			if _, ok := targetSegments[segment.SegmentID]; ok {
				segments = append(segments, segment)
			}
		}
		if len(segments) > 0 {
			planned = append(planned, SnapshotItem{
				NodeID:   item.NodeID,
				Segments: segments,
			})
		}
	}
	return planned, true, nil
}
