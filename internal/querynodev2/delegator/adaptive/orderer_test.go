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
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/v2/util/distance"
)

func TestSplit(t *testing.T) {
	t.Run("nil stats returns all as noStats", func(t *testing.T) {
		segIDs := []int64{1, 2, 3}
		orderable, noStats := Split(nil, segIDs)
		assert.Nil(t, orderable)
		assert.Equal(t, segIDs, noStats)
	})

	t.Run("empty input", func(t *testing.T) {
		stats := storage.NewPartitionStatsSnapshot()
		orderable, noStats := Split(stats, []int64{})
		assert.Nil(t, orderable)
		assert.Nil(t, noStats)
	})

	t.Run("splits segments by presence in stats", func(t *testing.T) {
		stats := storage.NewPartitionStatsSnapshot()
		stats.UpdateSegmentStats(1, storage.SegmentStats{FieldStats: []storage.FieldStats{}})
		stats.UpdateSegmentStats(3, storage.SegmentStats{FieldStats: []storage.FieldStats{}})

		segIDs := []int64{1, 2, 3, 4}
		orderable, noStats := Split(stats, segIDs)
		assert.Equal(t, []int64{1, 3}, orderable)
		assert.Equal(t, []int64{2, 4}, noStats)
	})
}

func TestOrder(t *testing.T) {
	queryVec := []float32{1, 0, 0}
	fieldID := int64(100)

	t.Run("empty input", func(t *testing.T) {
		stats := storage.NewPartitionStatsSnapshot()
		result := Order(queryVec, []int64{}, stats, fieldID, distance.L2)
		assert.Empty(t, result)
	})

	t.Run("L2 ascending order", func(t *testing.T) {
		stats := storage.NewPartitionStatsSnapshot()
		// seg1: centroid [2,0,0] → L2 dist to [1,0,0] = 1
		stats.UpdateSegmentStats(1, makeSegStats(fieldID, []float32{2, 0, 0}))
		// seg2: centroid [3,0,0] → L2 dist = 4
		stats.UpdateSegmentStats(2, makeSegStats(fieldID, []float32{3, 0, 0}))
		// seg3: centroid [0,0,0] → L2 dist = 1
		stats.UpdateSegmentStats(3, makeSegStats(fieldID, []float32{0, 0, 0}))

		result := Order(queryVec, []int64{1, 2, 3}, stats, fieldID, distance.L2)
		assert.Len(t, result, 3)
		// Last must be seg2 (farthest). First two are seg1/seg3 (tied).
		assert.Equal(t, int64(2), result[2])
		assert.Contains(t, result[:2], int64(1))
		assert.Contains(t, result[:2], int64(3))
	})

	t.Run("IP descending order", func(t *testing.T) {
		stats := storage.NewPartitionStatsSnapshot()
		// seg1: centroid [1,0,0] → IP = 1
		stats.UpdateSegmentStats(1, makeSegStats(fieldID, []float32{1, 0, 0}))
		// seg2: centroid [0.5,0,0] → IP = 0.5
		stats.UpdateSegmentStats(2, makeSegStats(fieldID, []float32{0.5, 0, 0}))

		result := Order(queryVec, []int64{1, 2}, stats, fieldID, distance.IP)
		assert.Len(t, result, 2)
		assert.Equal(t, int64(1), result[0]) // higher IP first
		assert.Equal(t, int64(2), result[1])
	})

	t.Run("COSINE descending order", func(t *testing.T) {
		stats := storage.NewPartitionStatsSnapshot()
		// seg1: centroid [1,1,0] → cosine with [1,0,0] ≈ 0.707
		stats.UpdateSegmentStats(1, makeSegStats(fieldID, []float32{1, 1, 0}))
		// seg2: centroid [0,0,1] → cosine with [1,0,0] = 0
		stats.UpdateSegmentStats(2, makeSegStats(fieldID, []float32{0, 0, 1}))

		result := Order(queryVec, []int64{1, 2}, stats, fieldID, distance.COSINE)
		assert.Len(t, result, 2)
		assert.Equal(t, int64(1), result[0]) // higher cosine first
		assert.Equal(t, int64(2), result[1])
	})

	t.Run("skips segments without centroid stats", func(t *testing.T) {
		stats := storage.NewPartitionStatsSnapshot()
		stats.UpdateSegmentStats(1, makeSegStats(fieldID, []float32{1, 0, 0}))
		stats.UpdateSegmentStats(2, storage.SegmentStats{
			FieldStats: []storage.FieldStats{{FieldID: fieldID, Centroids: []storage.VectorFieldValue{}}},
		})

		result := Order(queryVec, []int64{1, 2}, stats, fieldID, distance.L2)
		assert.Len(t, result, 1)
		assert.Equal(t, int64(1), result[0])
	})

	t.Run("skips segments with wrong field ID", func(t *testing.T) {
		stats := storage.NewPartitionStatsSnapshot()
		stats.UpdateSegmentStats(1, makeSegStats(999, []float32{1, 0, 0}))

		result := Order(queryVec, []int64{1}, stats, fieldID, distance.L2)
		// seg1's centroid has fieldID=999, not 100 → skipped → fallback returns [1] unordered
		assert.Len(t, result, 1)
	})

	t.Run("skips centroids with non-float32 values", func(t *testing.T) {
		stats := storage.NewPartitionStatsSnapshot()
		mockCentroid := &mockVectorFieldValue{}
		stats.UpdateSegmentStats(1, storage.SegmentStats{
			FieldStats: []storage.FieldStats{{FieldID: fieldID, Centroids: []storage.VectorFieldValue{mockCentroid}}},
		})

		result := Order(queryVec, []int64{1}, stats, fieldID, distance.L2)
		// non-float32 centroid → skipped → fallback returns [1] unordered
		assert.Len(t, result, 1)
	})
}

func makeSegStats(fieldID int64, centroid []float32) storage.SegmentStats {
	return storage.SegmentStats{
		FieldStats: []storage.FieldStats{{
			FieldID:   fieldID,
			Centroids: []storage.VectorFieldValue{storage.NewFloatVectorFieldValue(centroid)},
		}},
	}
}

type mockVectorFieldValue struct{}

func (m *mockVectorFieldValue) MarshalJSON() ([]byte, error) { return nil, nil }
func (m *mockVectorFieldValue) UnmarshalJSON([]byte) error   { return nil }
func (m *mockVectorFieldValue) SetValue(interface{}) error   { return nil }
func (m *mockVectorFieldValue) GetValue() interface{}        { return "not-a-vector" }
func (m *mockVectorFieldValue) Type() schemapb.DataType      { return schemapb.DataType_FloatVector }
func (m *mockVectorFieldValue) Size() int64                  { return 0 }
