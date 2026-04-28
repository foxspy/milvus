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

package segcore

/*
#cgo pkg-config: milvus_core

#include "segcore/reduce_incremental_c.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/cockroachdb/errors"
)

// ReduceHeap wraps the C++ IncrementalReduceHeap for cross-batch top-K state.
type ReduceHeap struct {
	ptr    C.CReduceHeap
	nq     int64
	topK   int64
	closed bool
}

// NewReduceHeap creates a new incremental reduce heap.
func NewReduceHeap(plan *SearchPlan, nq, topK int64) (*ReduceHeap, error) {
	var out C.CReduceHeap
	status := C.NewReduceHeap(plan.cSearchPlan, C.int64_t(nq), C.int64_t(topK), &out)
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return nil, errors.Wrap(err, "NewReduceHeap failed")
	}
	return &ReduceHeap{ptr: out, nq: nq, topK: topK}, nil
}

// MergeBatch merges a batch of per-segment search results into the heap.
func (h *ReduceHeap) MergeBatch(results []*SearchResult) error {
	if h.closed {
		return fmt.Errorf("ReduceHeap already closed")
	}
	if len(results) == 0 {
		return nil
	}
	cResults := make([]C.CSearchResult, len(results))
	for i, r := range results {
		cResults[i] = r.cSearchResult
	}
	status := C.ReduceHeap_MergeBatch(h.ptr, &cResults[0], C.int(len(results)))
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return errors.Wrap(err, "ReduceHeap_MergeBatch failed")
	}
	return nil
}

// Thresholds reads the current top-K threshold per nq.
func (h *ReduceHeap) Thresholds() ([]float32, error) {
	if h.closed {
		return nil, fmt.Errorf("ReduceHeap already closed")
	}
	out := make([]float32, h.nq)
	status := C.ReduceHeap_GetThresholds(h.ptr, (*C.float)(unsafe.Pointer(&out[0])))
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return nil, errors.Wrap(err, "ReduceHeap_GetThresholds failed")
	}
	return out, nil
}

// Finalize consumes the heap and returns the final search result blobs.
// After Finalize the heap is not reusable.
func (h *ReduceHeap) Finalize() (SearchResultDataBlobs, error) {
	if h.closed {
		return nil, fmt.Errorf("ReduceHeap already closed")
	}
	var blobs C.CSearchResultDataBlobs
	status := C.ReduceHeap_Finalize(h.ptr, &blobs)
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return nil, errors.Wrap(err, "ReduceHeap_Finalize failed")
	}
	h.closed = true
	return SearchResultDataBlobs(blobs), nil
}

// Close releases the C++ heap. Safe to call multiple times.
func (h *ReduceHeap) Close() {
	if h.closed {
		return
	}
	C.DeleteReduceHeap(h.ptr)
	h.closed = true
}
