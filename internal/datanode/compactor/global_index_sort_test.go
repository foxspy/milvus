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

package compactor

import (
	"container/heap"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/globalindex"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func TestGlobalIndexSortKeyLess(t *testing.T) {
	// lower centroid first
	assert.True(t, globalIndexSortKey{centroidID: 1, distance: 9}.less(globalIndexSortKey{centroidID: 2, distance: 0}))
	// same centroid, lower distance first
	assert.True(t, globalIndexSortKey{centroidID: 2, distance: 0.5}.less(globalIndexSortKey{centroidID: 2, distance: 0.6}))
	// equal keys are not strictly less
	assert.False(t, globalIndexSortKey{centroidID: 2, distance: 0.5}.less(globalIndexSortKey{centroidID: 2, distance: 0.5}))
}

// fakeRangeWriter simulates a MultiSegmentWriter that rotates to a new output
// segment once the current one reaches segCap rows.
type fakeRangeWriter struct {
	segCap   int
	segIDs   []int64
	curSeg   int
	curCount int
}

func (w *fakeRangeWriter) WriteValue(v *storage.Value) error {
	if w.curCount >= w.segCap {
		w.curSeg++
		w.curCount = 0
	}
	w.curCount++
	return nil
}

func (w *fakeRangeWriter) CurrentSegmentID() typeutil.UniqueID {
	return w.segIDs[w.curSeg]
}

func TestRangeTrackerContiguousAndSplit(t *testing.T) {
	mapping := make(globalindex.ChunkMapping)
	// segment capacity 4 rows; two output segments available.
	w := &fakeRangeWriter{segCap: 4, segIDs: []int64{100, 200}}
	rt := newRangeTracker(w, mapping)

	// Globally sorted by centroid: c0 x2, c1 x3, c2 x3 => 8 rows.
	// Segment rotation at 4 rows splits centroid 1 across seg 100 and seg 200.
	stream := []int32{0, 0, 1, 1, 1, 2, 2, 2}
	for _, c := range stream {
		assert.NoError(t, rt.write(&storage.Value{}, c))
	}
	rt.finish()

	// centroid 0: rows 0..1 in seg 100
	assert.Equal(t, []globalindex.Chunk{{SegmentID: 100, Offset: 0, Size: 2}}, mapping[0])
	// centroid 1: row 2..3 in seg 100, then row 0 in seg 200 (split by rotation)
	assert.Equal(t, []globalindex.Chunk{
		{SegmentID: 100, Offset: 2, Size: 2},
		{SegmentID: 200, Offset: 0, Size: 1},
	}, mapping[1])
	// centroid 2: rows 1..3 in seg 200
	assert.Equal(t, []globalindex.Chunk{{SegmentID: 200, Offset: 1, Size: 3}}, mapping[2])

	// Every row is covered exactly once.
	var total int64
	for _, chunks := range mapping {
		for _, c := range chunks {
			total += c.Size
		}
	}
	assert.Equal(t, int64(len(stream)), total)
	assert.NoError(t, mapping.Validate())
}

func TestRangeTrackerSingleCentroid(t *testing.T) {
	mapping := make(globalindex.ChunkMapping)
	w := &fakeRangeWriter{segCap: 100, segIDs: []int64{7}}
	rt := newRangeTracker(w, mapping)
	for i := 0; i < 5; i++ {
		assert.NoError(t, rt.write(&storage.Value{}, 3))
	}
	rt.finish()
	assert.Equal(t, []globalindex.Chunk{{SegmentID: 7, Offset: 0, Size: 5}}, mapping[3])
}

func TestSpillMergeHeapOrdering(t *testing.T) {
	h := &spillMergeHeap{}
	heap.Init(h)
	heap.Push(h, &spillRunReader{curKey: globalIndexSortKey{centroidID: 2, distance: 0.1}})
	heap.Push(h, &spillRunReader{curKey: globalIndexSortKey{centroidID: 0, distance: 0.9}})
	heap.Push(h, &spillRunReader{curKey: globalIndexSortKey{centroidID: 0, distance: 0.2}})
	heap.Push(h, &spillRunReader{curKey: globalIndexSortKey{centroidID: 1, distance: 0.5}})

	got := make([]globalIndexSortKey, 0, 4)
	for h.Len() > 0 {
		got = append(got, heap.Pop(h).(*spillRunReader).curKey)
	}
	assert.Equal(t, []globalIndexSortKey{
		{centroidID: 0, distance: 0.2},
		{centroidID: 0, distance: 0.9},
		{centroidID: 1, distance: 0.5},
		{centroidID: 2, distance: 0.1},
	}, got)
}
