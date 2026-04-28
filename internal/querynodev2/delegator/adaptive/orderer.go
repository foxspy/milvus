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

package adaptive

import (
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/segcore"
)

// Ordered pairs a segment ID with its distance to a query vector.
type Ordered struct {
	SegmentID int64
	Distance  float32
}

// Split partitions segment IDs into those present in stats (orderable)
// and those absent (pre-batch: growing or newly-flushed without stats).
func Split(stats *storage.PartitionStatsSnapshot, segIDs []int64) (orderable, noStats []int64) {
	if stats == nil {
		return nil, segIDs
	}
	for _, id := range segIDs {
		if _, ok := stats.SegmentStats[id]; ok {
			orderable = append(orderable, id)
		} else {
			noStats = append(noStats, id)
		}
	}
	return
}

// OrderByCentroidSearcher uses the C++ CentroidSearcher (knowhere BruteForce)
// to order segments by distance to queryVec. Returns ordered segment IDs.
// The searcher must have been pre-loaded with centroids via Update().
func OrderByCentroidSearcher(searcher *segcore.CentroidSearcher, queryVec []float32) ([]int64, error) {
	if searcher == nil || searcher.Count() == 0 {
		return nil, nil
	}
	return searcher.Order(queryVec, 1) // nq=1: order by first query vector
}

// Order sorts orderable segments by distance(queryVec, segment.centroid).
// This is the pure-Go fallback when CentroidSearcher is not available.
// Prefer OrderByCentroidSearcher for SIMD-accelerated performance.
func Order(queryVec []float32, segIDs []int64,
	stats *storage.PartitionStatsSnapshot, fieldID int64, metric string) []int64 {
	// Fallback: extract centroids from stats and delegate to CentroidSearcher.
	// For now, keep a simple Go implementation for cases where the C++ searcher
	// is not pre-built (e.g., tests with mock stats).
	if stats == nil || len(segIDs) == 0 || len(queryVec) == 0 {
		return segIDs
	}
	dim := int64(len(queryVec))

	// Build a temporary CentroidSearcher.
	searcher, err := segcore.NewCentroidSearcher(dim, metric)
	if err != nil {
		return segIDs // fallback: unordered
	}
	defer searcher.Close()

	var ids []int64
	var centroids []float32
	for _, id := range segIDs {
		segStat, ok := stats.SegmentStats[id]
		if !ok {
			continue
		}
		for _, fs := range segStat.FieldStats {
			if fs.FieldID == fieldID && len(fs.Centroids) > 0 {
				if c, ok := fs.Centroids[0].GetValue().([]float32); ok && int64(len(c)) == dim {
					ids = append(ids, id)
					centroids = append(centroids, c...)
				}
				break
			}
		}
	}
	if len(ids) == 0 {
		return segIDs
	}

	if err := searcher.Update(ids, centroids); err != nil {
		return segIDs
	}
	ordered, err := searcher.Order(queryVec, 1)
	if err != nil {
		return segIDs
	}
	return ordered
}

