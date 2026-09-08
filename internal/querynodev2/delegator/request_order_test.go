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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/internalpb"
)

func TestShallowCopySearchRequestPreservesProxyOrder(t *testing.T) {
	sd := &shardDelegator{}
	original := &internalpb.SearchRequest{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_Search,
			MsgID:     102,
			Timestamp: 101,
			SourceID:  100,
			TargetID:  99,
		},
	}

	copied := sd.shallowCopySearchRequest(original, 200)

	require.NotNil(t, copied.GetBase())
	assert.NotSame(t, original.GetBase(), copied.GetBase())
	assert.Equal(t, original.GetBase().GetMsgType(), copied.GetBase().GetMsgType())
	assert.Equal(t, original.GetBase().GetMsgID(), copied.GetBase().GetMsgID())
	assert.Equal(t, original.GetBase().GetTimestamp(), copied.GetBase().GetTimestamp())
	assert.Equal(t, original.GetBase().GetSourceID(), copied.GetBase().GetSourceID())
	assert.Equal(t, int64(200), copied.GetBase().GetTargetID())
}

func TestShallowCopyRetrieveRequestPreservesProxyOrder(t *testing.T) {
	sd := &shardDelegator{}
	original := &internalpb.RetrieveRequest{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_Retrieve,
			MsgID:     202,
			Timestamp: 201,
			SourceID:  100,
			TargetID:  99,
		},
	}

	copied := sd.shallowCopyRetrieveRequest(original, 200)

	require.NotNil(t, copied.GetBase())
	assert.NotSame(t, original.GetBase(), copied.GetBase())
	assert.Equal(t, original.GetBase().GetMsgType(), copied.GetBase().GetMsgType())
	assert.Equal(t, original.GetBase().GetMsgID(), copied.GetBase().GetMsgID())
	assert.Equal(t, original.GetBase().GetTimestamp(), copied.GetBase().GetTimestamp())
	assert.Equal(t, original.GetBase().GetSourceID(), copied.GetBase().GetSourceID())
	assert.Equal(t, int64(200), copied.GetBase().GetTargetID())
}
