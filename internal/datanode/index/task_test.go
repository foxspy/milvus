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

package index

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/analyzecgowrapper"
	"github.com/milvus-io/milvus/internal/util/dependency"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/clusteringpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/etcdpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/workerpb"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/metric"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/timerecord"
)

type IndexBuildTaskSuite struct {
	suite.Suite
	schema       *schemapb.CollectionSchema
	collectionID int64
	partitionID  int64
	segmentID    int64
	dataPath     string
	rootPath     string

	numRows int
	dim     int
}

func (suite *IndexBuildTaskSuite) SetupSuite() {
	paramtable.Init()
	suite.collectionID = 1000
	suite.partitionID = 1001
	suite.segmentID = 1002
	suite.rootPath = suite.T().TempDir() + "/data"
	suite.dataPath = suite.rootPath + "/1000/1001/1002/3/1"
	suite.numRows = 100
	suite.dim = 128
}

func (suite *IndexBuildTaskSuite) SetupTest() {
	suite.schema = &schemapb.CollectionSchema{
		Name:        "test",
		Description: "test",
		AutoID:      false,
		Fields: []*schemapb.FieldSchema{
			{FieldID: common.RowIDField, Name: common.RowIDFieldName, DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: common.TimeStampField, Name: common.TimeStampFieldName, DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "ts", DataType: schemapb.DataType_Int64},
			{FieldID: 102, Name: "vec", DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{{Key: "dim", Value: "128"}}},
		},
	}
}

func (suite *IndexBuildTaskSuite) serializeData() ([]*storage.Blob, error) {
	insertCodec := storage.NewInsertCodecWithSchema(&etcdpb.CollectionMeta{
		Schema: suite.schema,
	})
	return insertCodec.Serialize(suite.partitionID, suite.segmentID, &storage.InsertData{
		Data: map[storage.FieldID]storage.FieldData{
			0:   &storage.Int64FieldData{Data: generateLongs(suite.numRows)},
			1:   &storage.Int64FieldData{Data: generateLongs(suite.numRows)},
			100: &storage.Int64FieldData{Data: generateLongs(suite.numRows)},
			101: &storage.Int64FieldData{Data: generateLongs(suite.numRows)},
			102: &storage.FloatVectorFieldData{Data: generateFloats(suite.numRows * suite.dim), Dim: suite.dim},
		},
		Infos: []storage.BlobInfo{{Length: suite.numRows}},
	})
}

func (suite *IndexBuildTaskSuite) TestBuildMemoryIndex() {
	ctx, cancel := context.WithCancel(context.Background())
	req := &workerpb.CreateJobRequest{
		BuildID:      1,
		IndexVersion: 1,
		DataPaths:    []string{suite.dataPath},
		IndexID:      0,
		IndexName:    "",
		IndexParams:  []*commonpb.KeyValuePair{{Key: common.IndexTypeKey, Value: "FLAT"}, {Key: common.MetricTypeKey, Value: metric.L2}},
		TypeParams:   []*commonpb.KeyValuePair{{Key: "dim", Value: "128"}},
		NumRows:      int64(suite.numRows),
		StorageConfig: &indexpb.StorageConfig{
			RootPath:    suite.rootPath,
			StorageType: "local",
		},
		CollectionID: 1,
		PartitionID:  2,
		SegmentID:    3,
		FieldID:      102,
		FieldName:    "vec",
		FieldType:    schemapb.DataType_FloatVector,
	}

	cm, err := dependency.NewDefaultFactory(true).NewPersistentStorageChunkManager(ctx)
	suite.NoError(err)
	blobs, err := suite.serializeData()
	suite.NoError(err)
	err = cm.Write(ctx, suite.dataPath, blobs[0].Value)
	suite.NoError(err)

	t := NewIndexBuildTask(ctx, cancel, req, cm, NewTaskManager(context.Background()), nil)

	err = t.PreExecute(context.Background())
	suite.NoError(err)
	err = t.Execute(context.Background())
	suite.NoError(err)
	err = t.PostExecute(context.Background())
	suite.NoError(err)
}

func TestIndexBuildTask(t *testing.T) {
	suite.Run(t, new(IndexBuildTaskSuite))
}

type AnalyzeTaskSuite struct {
	suite.Suite
	schema       *schemapb.CollectionSchema
	collectionID int64
	partitionID  int64
	segmentID    int64
	fieldID      int64
	taskID       int64
}

func (suite *AnalyzeTaskSuite) SetupSuite() {
	paramtable.Init()
	suite.collectionID = 1000
	suite.partitionID = 1001
	suite.segmentID = 1002
	suite.fieldID = 102
	suite.taskID = 1004
}

func (suite *AnalyzeTaskSuite) SetupTest() {
	suite.schema = &schemapb.CollectionSchema{
		Name:        "test",
		Description: "test",
		AutoID:      false,
		Fields: []*schemapb.FieldSchema{
			{FieldID: common.RowIDField, Name: common.RowIDFieldName, DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: common.TimeStampField, Name: common.TimeStampFieldName, DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "ts", DataType: schemapb.DataType_Int64},
			{FieldID: 102, Name: "vec", DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{{Key: "dim", Value: "1"}}},
		},
	}
}

func (suite *AnalyzeTaskSuite) serializeData() ([]*storage.Blob, error) {
	insertCodec := storage.NewInsertCodecWithSchema(&etcdpb.CollectionMeta{
		Schema: suite.schema,
	})
	return insertCodec.Serialize(suite.partitionID, suite.segmentID, &storage.InsertData{
		Data: map[storage.FieldID]storage.FieldData{
			0:   &storage.Int64FieldData{Data: []int64{0, 1, 2}},
			1:   &storage.Int64FieldData{Data: []int64{1, 2, 3}},
			100: &storage.Int64FieldData{Data: []int64{0, 1, 2}},
			101: &storage.Int64FieldData{Data: []int64{0, 1, 2}},
			102: &storage.FloatVectorFieldData{Data: []float32{1, 2, 3}, Dim: 1},
		},
		Infos: []storage.BlobInfo{{Length: 3}},
	})
}

func (suite *AnalyzeTaskSuite) TestAnalyze() {
	ctx, cancel := context.WithCancel(context.Background()) //nolint:gosec // cancel is deferred below
	defer cancel()
	req := &workerpb.AnalyzeRequest{
		ClusterID:    "test",
		TaskID:       1,
		CollectionID: suite.collectionID,
		PartitionID:  suite.partitionID,
		FieldID:      suite.fieldID,
		FieldName:    "vec",
		FieldType:    schemapb.DataType_FloatVector,
		SegmentStats: map[int64]*indexpb.SegmentStats{
			suite.segmentID: {
				ID:      suite.segmentID,
				NumRows: 1024,
				LogIDs:  []int64{1},
			},
		},
		Version: 1,
		StorageConfig: &indexpb.StorageConfig{
			RootPath:    suite.T().TempDir() + "/data",
			StorageType: "local",
		},
		Dim: 1,
	}

	cm, err := dependency.NewDefaultFactory(true).NewPersistentStorageChunkManager(ctx)
	suite.NoError(err)
	blobs, err := suite.serializeData()
	suite.NoError(err)
	dataPath := metautil.BuildInsertLogPath(cm.RootPath(), suite.collectionID, suite.partitionID, suite.segmentID,
		suite.fieldID, 1)

	err = cm.Write(ctx, dataPath, blobs[0].Value)
	suite.NoError(err)

	t := &analyzeTask{
		ident:    "",
		cancel:   cancel,
		ctx:      ctx,
		req:      req,
		tr:       timerecord.NewTimeRecorder("test-indexBuildTask"),
		queueDur: 0,
		manager:  NewTaskManager(context.Background()),
	}

	err = t.PreExecute(context.Background())
	suite.NoError(err)
}

type fakeAnalyze struct{}

func (f *fakeAnalyze) Delete() error {
	return nil
}

func (f *fakeAnalyze) GetResult(size int) (string, int64, []string, []int64, error) {
	return "centroids", 1, nil, nil, nil
}

func TestAnalyzeTaskExecuteUsesInjectedRunner(t *testing.T) {
	ctx := context.Background()
	req := &workerpb.AnalyzeRequest{
		ClusterID:    "cluster",
		TaskID:       10,
		CollectionID: 100,
		PartitionID:  200,
		FieldID:      300,
		FieldName:    "vec",
		FieldType:    schemapb.DataType_FloatVector,
		SegmentStats: map[int64]*indexpb.SegmentStats{
			400: {
				ID:      400,
				NumRows: 1000,
				LogIDs:  []int64{500, 501},
			},
		},
		Version: 2,
		StorageConfig: &indexpb.StorageConfig{
			RootPath:    "/root",
			StorageType: "local",
		},
		Dim:                 128,
		NumClusters:         16,
		MaxTrainSizeRatio:   0.2,
		MinClusterSizeRatio: 0.1,
		MaxClusterSizeRatio: 10,
		MaxClusterSize:      1024,
	}

	var called bool
	runner := func(ctx context.Context, info *clusteringpb.AnalyzeInfo) (analyzecgowrapper.CodecAnalyze, error) {
		called = true
		require.Equal(t, req.GetClusterID(), info.GetClusterID())
		require.Equal(t, req.GetTaskID(), info.GetBuildID())
		require.Equal(t, req.GetCollectionID(), info.GetCollectionID())
		require.Equal(t, req.GetPartitionID(), info.GetPartitionID())
		require.Equal(t, req.GetFieldID(), info.GetFieldSchema().GetFieldID())
		require.Equal(t, req.GetFieldType(), info.GetFieldSchema().GetDataType())
		require.Equal(t, req.GetNumClusters(), info.GetNumClusters())
		require.Equal(t, req.GetDim(), info.GetDim())
		require.Equal(t, int64(1000), info.GetNumRows()[400])
		require.Equal(t, []string{
			metautil.BuildInsertLogPath("/root", 100, 200, 400, 300, 500),
			metautil.BuildInsertLogPath("/root", 100, 200, 400, 300, 501),
		}, info.GetInsertFiles()[400].GetInsertFiles())
		return &fakeAnalyze{}, nil
	}

	task := NewAnalyzeTask(ctx, func() {}, req, NewTaskManager(ctx))
	task.analyzeRunner = runner
	require.NoError(t, task.Execute(ctx))
	require.True(t, called)
	require.IsType(t, &fakeAnalyze{}, task.analyze)
}

func TestAnalyzeTaskExecuteUsesAnalyzeV2RunnerForGlobalIndex(t *testing.T) {
	ctx := context.Background()
	req := &workerpb.AnalyzeRequest{
		ClusterID:         "cluster",
		TaskID:            10,
		CollectionID:      100,
		PartitionID:       200,
		FieldID:           300,
		FieldName:         "vec",
		FieldType:         schemapb.DataType_FloatVector,
		InsertChannel:     "by-dev-rootcoord-dml_0_100v0",
		EnableGlobalIndex: true,
		SegmentStats: map[int64]*indexpb.SegmentStats{
			400: {
				ID:      400,
				NumRows: 1000,
				LogIDs:  []int64{500},
			},
		},
		Version: 2,
		StorageConfig: &indexpb.StorageConfig{
			RootPath:    "/root",
			StorageType: "local",
		},
		Dim:                 128,
		NumClusters:         16,
		MaxTrainSizeRatio:   0.2,
		MinClusterSizeRatio: 0.1,
		MaxClusterSizeRatio: 10,
		MaxClusterSize:      1024,
	}

	var called bool
	runner := func(ctx context.Context, info *clusteringpb.AnalyzeInfo) (analyzecgowrapper.CodecAnalyze, error) {
		called = true
		require.True(t, info.GetEnableGlobalIndex())
		require.Equal(t, req.GetInsertChannel(), info.GetInsertChannel())
		require.Contains(t, info.GetGlobalStatsIndexRoot(), "global_stats_index/100/200/by-dev-rootcoord-dml_0_100v0/2")
		require.Contains(t, info.GetHeadIndexPath(), "global_stats_index/100/200/by-dev-rootcoord-dml_0_100v0/2/head_index")
		require.Contains(t, info.GetChunkMappingPath(), "global_stats_index/100/200/by-dev-rootcoord-dml_0_100v0/2/chunk_mapping")
		require.Contains(t, info.GetCompactionPlanPath(), "analyze_stats/10/2/100/200/300/compaction_pre_plan")
		return &fakeAnalyze{}, nil
	}

	task := NewAnalyzeTask(ctx, func() {}, req, NewTaskManager(ctx))
	task.analyzeV2Runner = runner
	require.NoError(t, task.Execute(ctx))
	require.True(t, called)
	require.IsType(t, &fakeAnalyze{}, task.analyze)
}

func TestAnalyzeTaskSuite(t *testing.T) {
	suite.Run(t, new(AnalyzeTaskSuite))
}
