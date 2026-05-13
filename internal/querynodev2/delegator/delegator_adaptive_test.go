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

package delegator

import (
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/proto/internalpb"
)

func TestDecodeAdaptiveSearchResultsFeedsRunningTopK(t *testing.T) {
	resultData := &schemapb.SearchResultData{
		NumQueries: 1,
		TopK:       2,
		Topks:      []int64{2},
		Scores:     []float32{-0.11, -0.22},
	}
	blob, err := proto.Marshal(resultData)
	require.NoError(t, err)

	results := []*internalpb.SearchResults{{SlicedBlob: blob}}

	runHeap := newRunningTopK(1, 2)
	runHeap.merge(results)
	_, full := runHeap.thresholds()
	require.False(t, full)
	require.Zero(t, searchResultsScoreCount(results))

	require.NoError(t, decodeAdaptiveSearchResults(results))
	require.NotNil(t, results[0].GetResultData())
	require.Empty(t, results[0].GetSlicedBlob())
	require.Equal(t, 2, searchResultsScoreCount(results))

	runHeap.merge(results)
	thresholds, full := runHeap.thresholds()
	require.True(t, full)
	require.Equal(t, []float32{-0.22}, thresholds)
}
