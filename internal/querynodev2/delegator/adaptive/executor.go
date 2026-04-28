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
	"math"
	"strconv"

	"github.com/milvus-io/milvus/internal/storage"
)

// SearchResults is an opaque batch of per-segment results. The executor
// passes it between Dispatcher and Heap without inspecting contents.
type SearchResults interface{}

// Dispatcher sends a search/query to a set of segments and returns results.
type Dispatcher interface {
	Dispatch(ctx context.Context, segmentIDs []int64) ([]SearchResults, error)
}

// Heap abstracts the cross-batch top-K state (backed by C++ ReduceHeap in production).
type Heap interface {
	MergeBatch(results []SearchResults) error
	Thresholds() ([]float32, error)
	Finalize() (interface{}, error) // returns opaque blobs
	Close()
}

// HeapFactory creates a Heap per query.
type HeapFactory interface {
	New(nq, topK int64, metric string) (Heap, error)
}

// Input contains everything the executor needs to run one adaptive search.
type Input struct {
	Ctx          context.Context
	Nq           int64
	TopK         int64
	Metric       string
	CollectionID int64
	QueryVectors []float32 // first query vector (for ordering)
	FieldID      int64     // clustering key field ID
	PartStats    map[int64]*storage.PartitionStatsSnapshot
	SealedIDs    []int64
	GrowingIDs   []int64
}

// Output is the result of an adaptive search.
type Output struct {
	Blobs   interface{} // opaque; caller type-asserts
	Batches int
	Pruned  int
	Reason  string // convergence reason or "exhausted"
}

// BatchedExecutor coordinates adaptive search execution.
type BatchedExecutor struct {
	Dispatcher  Dispatcher
	HeapFactory HeapFactory
	Converger   Converger
	BatchSizer  func(total int) int
}

func (e *BatchedExecutor) Run(in Input) (*Output, error) {
	heap, err := e.HeapFactory.New(in.Nq, in.TopK, in.Metric)
	if err != nil {
		return nil, err
	}
	closed := false
	defer func() {
		if !closed {
			heap.Close()
		}
	}()

	// Collect all orderable segments across all partitions.
	var allOrderable []int64
	var allNoStats []int64
	for _, ps := range in.PartStats {
		o, ns := Split(ps, in.SealedIDs)
		allOrderable = append(allOrderable, o...)
		allNoStats = append(allNoStats, ns...)
	}
	// Deduplicate (a segment may appear in multiple partition stats).
	allOrderable = dedup(allOrderable)
	allNoStats = dedup(allNoStats)

	// Order by centroid distance using first query vector.
	ordered := Order(in.QueryVectors, allOrderable, firstStats(in.PartStats), in.FieldID, in.Metric)
	// Order now returns []int64 (segment IDs already sorted by centroid distance)

	// Pre-batch: growing + no-stats sealed.
	preBatch := append([]int64{}, allNoStats...)
	preBatch = append(preBatch, in.GrowingIDs...)
	if len(preBatch) > 0 {
		results, err := e.Dispatcher.Dispatch(in.Ctx, preBatch)
		if err != nil {
			return nil, err
		}
		if err := heap.MergeBatch(results); err != nil {
			return nil, err
		}
	}

	batchSize := e.BatchSizer(len(ordered))
	batches := 0
	consumed := 0
	reason := "exhausted"

	for consumed < len(ordered) {
		if err := in.Ctx.Err(); err != nil {
			return nil, err
		}

		end := consumed + batchSize
		if end > len(ordered) {
			end = len(ordered)
		}
		batch := make([]int64, 0, end-consumed)
		for i := consumed; i < end; i++ {
			batch = append(batch, ordered[i])
		}

		results, err := e.Dispatcher.Dispatch(in.Ctx, batch)
		if err != nil {
			return nil, err
		}
		if err := heap.MergeBatch(results); err != nil {
			return nil, err
		}

		batches++
		consumed = end

		thresh, err := heap.Thresholds()
		if err != nil {
			return nil, err
		}
		converged, r := e.Converger.Check(thresh, len(results), batches-1)
		if converged {
			reason = r
			break
		}
	}

	pruned := len(ordered) - consumed

	blobs, err := heap.Finalize()
	if err != nil {
		return nil, err
	}
	closed = true

	return &Output{
		Blobs:   blobs,
		Batches: batches,
		Pruned:  pruned,
		Reason:  reason,
	}, nil
}

// ResolveBatchSize converts the config string to an int.
func ResolveBatchSize(cfg string, total int) int {
	if cfg == "auto" || cfg == "" {
		n := int(math.Sqrt(float64(total)))
		if n < 4 {
			n = 4
		}
		if n > 64 {
			n = 64
		}
		return n
	}
	if v, err := strconv.Atoi(cfg); err == nil && v > 0 {
		if v > total {
			return total
		}
		return v
	}
	return ResolveBatchSize("auto", total)
}

func firstStats(m map[int64]*storage.PartitionStatsSnapshot) *storage.PartitionStatsSnapshot {
	for _, v := range m {
		return v
	}
	return nil
}

func dedup(ids []int64) []int64 {
	seen := make(map[int64]struct{}, len(ids))
	out := make([]int64, 0, len(ids))
	for _, id := range ids {
		if _, ok := seen[id]; !ok {
			seen[id] = struct{}{}
			out = append(out, id)
		}
	}
	return out
}
