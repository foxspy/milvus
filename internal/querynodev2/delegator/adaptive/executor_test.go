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
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/storage"
)

// --- Mocks ---

type mockResults struct{ segIDs []int64 }

type mockDispatcher struct {
	calls [][]int64 // record which segment IDs were dispatched
}

func (d *mockDispatcher) Dispatch(ctx context.Context, segIDs []int64) ([]SearchResults, error) {
	d.calls = append(d.calls, segIDs)
	return []SearchResults{&mockResults{segIDs}}, nil
}

type mockHeap struct {
	mergeCount int
	thresholds []float32
	finalized  bool
	closed     bool
}

func (h *mockHeap) MergeBatch(results []SearchResults) error {
	h.mergeCount++
	return nil
}
func (h *mockHeap) Thresholds() ([]float32, error) { return h.thresholds, nil }
func (h *mockHeap) Finalize() (interface{}, error)  { h.finalized = true; return "blobs", nil }
func (h *mockHeap) Close()                          { h.closed = true }

type mockHeapFactory struct {
	heap *mockHeap
}

func (f *mockHeapFactory) New(nq, topK int64, metric string) (Heap, error) {
	return f.heap, nil
}

// alwaysConverger converges after `after` batches.
type alwaysConverger struct {
	after int
}

func (c *alwaysConverger) Check(thresh []float32, newEntries, batchIdx int) (bool, string) {
	if batchIdx+1 >= c.after {
		return true, "stable"
	}
	return false, ""
}

// neverConverger never converges.
type neverConverger struct{}

func (c *neverConverger) Check(thresh []float32, newEntries, batchIdx int) (bool, string) {
	return false, ""
}

// Build a simple PartitionStatsSnapshot with N segments (IDs 0..N-1).
func makeStats(n int, fieldID int64) map[int64]*storage.PartitionStatsSnapshot {
	segStats := make(map[int64]storage.SegmentStats, n)
	for i := 0; i < n; i++ {
		segStats[int64(i)] = storage.SegmentStats{
			FieldStats: []storage.FieldStats{{
				FieldID:   fieldID,
				Centroids: []storage.VectorFieldValue{storage.NewFloatVectorFieldValue([]float32{float32(i), 0})},
			}},
		}
	}
	return map[int64]*storage.PartitionStatsSnapshot{
		1: {SegmentStats: segStats},
	}
}

func baseInput(n int) Input {
	ids := make([]int64, n)
	for i := range ids {
		ids[i] = int64(i)
	}
	return Input{
		Ctx:          context.Background(),
		Nq:           1,
		TopK:         5,
		Metric:       "L2",
		CollectionID: 1,
		QueryVectors: []float32{0, 0},
		FieldID:      100,
		PartStats:    makeStats(n, 100),
		SealedIDs:    ids,
		GrowingIDs:   nil,
	}
}

// --- Tests ---

func TestExecutor_SingleBatch(t *testing.T) {
	heap := &mockHeap{thresholds: []float32{1.0}}
	disp := &mockDispatcher{}
	exec := &BatchedExecutor{
		Dispatcher:  disp,
		HeapFactory: &mockHeapFactory{heap: heap},
		Converger:   &neverConverger{},
		BatchSizer:  func(total int) int { return total }, // one batch
	}
	out, err := exec.Run(baseInput(8))
	assert.NoError(t, err)
	assert.Equal(t, 1, out.Batches)
	assert.Equal(t, 0, out.Pruned)
	assert.Equal(t, "exhausted", out.Reason)
	assert.True(t, heap.finalized)
}

func TestExecutor_EarlyTermination(t *testing.T) {
	heap := &mockHeap{thresholds: []float32{1.0}}
	disp := &mockDispatcher{}
	exec := &BatchedExecutor{
		Dispatcher:  disp,
		HeapFactory: &mockHeapFactory{heap: heap},
		Converger:   &alwaysConverger{after: 3},
		BatchSizer:  func(total int) int { return 10 }, // 10 per batch
	}
	out, err := exec.Run(baseInput(100))
	assert.NoError(t, err)
	assert.Equal(t, 3, out.Batches)
	assert.Equal(t, 70, out.Pruned)
	assert.Equal(t, "stable", out.Reason)
}

func TestExecutor_PreBatchRunsGrowingAndNoStats(t *testing.T) {
	heap := &mockHeap{thresholds: []float32{1.0}}
	disp := &mockDispatcher{}
	exec := &BatchedExecutor{
		Dispatcher:  disp,
		HeapFactory: &mockHeapFactory{heap: heap},
		Converger:   &neverConverger{},
		BatchSizer:  func(total int) int { return 100 },
	}
	in := baseInput(5) // 5 orderable segments (IDs 0-4)
	in.SealedIDs = append(in.SealedIDs, 99, 100) // 99, 100 not in stats → pre-batch
	in.GrowingIDs = []int64{200, 201}             // growing → pre-batch

	out, err := exec.Run(in)
	assert.NoError(t, err)

	// First dispatch call should be pre-batch (noStats + growing).
	assert.GreaterOrEqual(t, len(disp.calls), 2) // at least pre-batch + 1 ordered batch
	preBatchIDs := disp.calls[0]
	assert.Contains(t, preBatchIDs, int64(99))
	assert.Contains(t, preBatchIDs, int64(100))
	assert.Contains(t, preBatchIDs, int64(200))
	assert.Contains(t, preBatchIDs, int64(201))
	assert.Equal(t, 1, out.Batches) // ordered batch count
}

func TestExecutor_ContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	heap := &mockHeap{thresholds: []float32{1.0}}
	disp := &mockDispatcher{}
	exec := &BatchedExecutor{
		Dispatcher:  disp,
		HeapFactory: &mockHeapFactory{heap: heap},
		Converger:   &neverConverger{},
		BatchSizer:  func(total int) int { return 10 },
	}
	in := baseInput(100)
	in.Ctx = ctx

	_, err := exec.Run(in)
	assert.Error(t, err)
	assert.True(t, heap.closed)
}

func TestExecutor_DispatchErrorPropagates(t *testing.T) {
	errDisp := &errorDispatcher{err: fmt.Errorf("rpc failed")}
	heap := &mockHeap{thresholds: []float32{1.0}}
	exec := &BatchedExecutor{
		Dispatcher:  errDisp,
		HeapFactory: &mockHeapFactory{heap: heap},
		Converger:   &neverConverger{},
		BatchSizer:  func(total int) int { return 10 },
	}
	_, err := exec.Run(baseInput(20))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "rpc failed")
	assert.True(t, heap.closed)
}

func TestExecutor_HeapFinalizeCalledOnSuccess(t *testing.T) {
	heap := &mockHeap{thresholds: []float32{1.0}}
	exec := &BatchedExecutor{
		Dispatcher:  &mockDispatcher{},
		HeapFactory: &mockHeapFactory{heap: heap},
		Converger:   &neverConverger{},
		BatchSizer:  func(total int) int { return 100 },
	}
	out, err := exec.Run(baseInput(10))
	assert.NoError(t, err)
	assert.True(t, heap.finalized)
	assert.Equal(t, "blobs", out.Blobs)
}

func TestResolveBatchSize_Auto(t *testing.T) {
	assert.Equal(t, 10, ResolveBatchSize("auto", 100))
	assert.Equal(t, 4, ResolveBatchSize("auto", 4))     // min 4
	assert.Equal(t, 64, ResolveBatchSize("auto", 10000)) // max 64
}

func TestResolveBatchSize_Explicit(t *testing.T) {
	assert.Equal(t, 20, ResolveBatchSize("20", 100))
	assert.Equal(t, 50, ResolveBatchSize("100", 50)) // capped at total
}

func TestResolveBatchSize_Invalid(t *testing.T) {
	assert.Equal(t, ResolveBatchSize("auto", 100), ResolveBatchSize("garbage", 100))
}

// errorDispatcher always returns an error.
type errorDispatcher struct{ err error }

func (d *errorDispatcher) Dispatch(ctx context.Context, segIDs []int64) ([]SearchResults, error) {
	return nil, d.err
}
