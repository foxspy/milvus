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

#include <stdlib.h>
#include "segcore/centroid_searcher_c.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/cockroachdb/errors"
)

// CentroidSearcher wraps the C++ CentroidSearcher for SIMD-accelerated
// segment ordering by centroid distance.
type CentroidSearcher struct {
	ptr    C.CCentroidSearcher
	closed bool
}

// NewCentroidSearcher creates a new searcher for the given dimension and metric.
func NewCentroidSearcher(dim int64, metricType string) (*CentroidSearcher, error) {
	cMetric := C.CString(metricType)
	defer C.free(unsafe.Pointer(cMetric))

	var out C.CCentroidSearcher
	status := C.NewCentroidSearcher(C.int64_t(dim), cMetric, &out)
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return nil, errors.Wrap(err, "NewCentroidSearcher failed")
	}
	return &CentroidSearcher{ptr: out}, nil
}

// Update replaces all centroids. segIDs and centroids are parallel;
// centroids is flat [len(segIDs) × dim].
func (s *CentroidSearcher) Update(segIDs []int64, centroids []float32) error {
	if s.closed {
		return fmt.Errorf("CentroidSearcher closed")
	}
	if len(segIDs) == 0 {
		return nil
	}
	status := C.CentroidSearcher_Update(
		s.ptr,
		(*C.int64_t)(unsafe.Pointer(&segIDs[0])),
		(*C.float)(unsafe.Pointer(&centroids[0])),
		C.int64_t(len(segIDs)),
	)
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return errors.Wrap(err, "CentroidSearcher_Update failed")
	}
	return nil
}

// Order returns segment IDs sorted by distance to queryVec.
// queryVec is [nq × dim] flat. Returns [nq × count] ordered segment IDs.
func (s *CentroidSearcher) Order(queryVec []float32, nq int64) ([]int64, error) {
	if s.closed {
		return nil, fmt.Errorf("CentroidSearcher closed")
	}
	count := int64(C.CentroidSearcher_Count(s.ptr))
	if count == 0 || nq == 0 {
		return nil, nil
	}
	outSize := nq * count
	result := make([]int64, outSize)
	var outCount C.int64_t
	status := C.CentroidSearcher_Order(
		s.ptr,
		(*C.float)(unsafe.Pointer(&queryVec[0])),
		C.int64_t(nq),
		(*C.int64_t)(unsafe.Pointer(&result[0])),
		&outCount,
	)
	if err := ConsumeCStatusIntoError(&status); err != nil {
		return nil, errors.Wrap(err, "CentroidSearcher_Order failed")
	}
	return result[:nq*int64(outCount)], nil
}

// Count returns the number of centroids currently held.
func (s *CentroidSearcher) Count() int64 {
	if s.closed {
		return 0
	}
	return int64(C.CentroidSearcher_Count(s.ptr))
}

// Close releases the C++ object. Safe to call multiple times.
func (s *CentroidSearcher) Close() {
	if s.closed {
		return
	}
	C.DeleteCentroidSearcher(s.ptr)
	s.closed = true
}
