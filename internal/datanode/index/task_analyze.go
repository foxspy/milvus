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
	"fmt"
	"time"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/analyzecgowrapper"
	"github.com/milvus-io/milvus/internal/util/globalindex"
	"github.com/milvus-io/milvus/pkg/v3/log"
	"github.com/milvus-io/milvus/pkg/v3/proto/clusteringpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/workerpb"
	"github.com/milvus-io/milvus/pkg/v3/util/hardware"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/timerecord"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

var _ Task = (*analyzeTask)(nil)

type analyzeTask struct {
	ident  string
	ctx    context.Context
	cancel context.CancelFunc
	req    *workerpb.AnalyzeRequest

	tr       *timerecord.TimeRecorder
	queueDur time.Duration
	manager  *TaskManager
	analyze  analyzecgowrapper.CodecAnalyze

	analyzeRunner   func(context.Context, *clusteringpb.AnalyzeInfo) (analyzecgowrapper.CodecAnalyze, error)
	analyzeV2Runner func(context.Context, *clusteringpb.AnalyzeInfo) (analyzecgowrapper.CodecAnalyze, error)
}

func NewAnalyzeTask(ctx context.Context,
	cancel context.CancelFunc,
	req *workerpb.AnalyzeRequest,
	manager *TaskManager,
) *analyzeTask {
	return &analyzeTask{
		ident:   fmt.Sprintf("%s/%d", req.GetClusterID(), req.GetTaskID()),
		ctx:     ctx,
		cancel:  cancel,
		req:     req,
		manager: manager,
		tr:      timerecord.NewTimeRecorder(fmt.Sprintf("ClusterID: %s, TaskID: %d", req.GetClusterID(), req.GetTaskID())),

		analyzeRunner:   analyzecgowrapper.Analyze,
		analyzeV2Runner: analyzecgowrapper.AnalyzeV2,
	}
}

func (at *analyzeTask) Ctx() context.Context {
	return at.ctx
}

func (at *analyzeTask) Name() string {
	return at.ident
}

func (at *analyzeTask) GetSlot() int64 {
	return at.req.GetTaskSlot()
}

func (at *analyzeTask) IsVectorIndex() bool {
	return false
}

func (at *analyzeTask) PreExecute(ctx context.Context) error {
	at.queueDur = at.tr.RecordSpan()
	log := log.Ctx(ctx).With(zap.String("clusterID", at.req.GetClusterID()),
		zap.Int64("TaskID", at.req.GetTaskID()), zap.Int64("Collection", at.req.GetCollectionID()),
		zap.Int64("partitionID", at.req.GetPartitionID()), zap.Int64("fieldID", at.req.GetFieldID()))
	log.Info("Begin to prepare analyze task")

	log.Info("Successfully prepare analyze task, nothing to do...")
	return nil
}

func (at *analyzeTask) Execute(ctx context.Context) error {
	var err error

	log := log.Ctx(ctx).With(zap.String("clusterID", at.req.GetClusterID()),
		zap.Int64("TaskID", at.req.GetTaskID()), zap.Int64("Collection", at.req.GetCollectionID()),
		zap.Int64("partitionID", at.req.GetPartitionID()), zap.Int64("fieldID", at.req.GetFieldID()))

	log.Info("Begin to build analyze task")

	storageConfig := &clusteringpb.StorageConfig{
		Address:           at.req.GetStorageConfig().GetAddress(),
		AccessKeyID:       at.req.GetStorageConfig().GetAccessKeyID(),
		SecretAccessKey:   at.req.GetStorageConfig().GetSecretAccessKey(),
		UseSSL:            at.req.GetStorageConfig().GetUseSSL(),
		BucketName:        at.req.GetStorageConfig().GetBucketName(),
		RootPath:          at.req.GetStorageConfig().GetRootPath(),
		UseIAM:            at.req.GetStorageConfig().GetUseIAM(),
		IAMEndpoint:       at.req.GetStorageConfig().GetIAMEndpoint(),
		StorageType:       at.req.GetStorageConfig().GetStorageType(),
		UseVirtualHost:    at.req.GetStorageConfig().GetUseVirtualHost(),
		Region:            at.req.GetStorageConfig().GetRegion(),
		CloudProvider:     at.req.GetStorageConfig().GetCloudProvider(),
		RequestTimeoutMs:  at.req.GetStorageConfig().GetRequestTimeoutMs(),
		SslCACert:         at.req.GetStorageConfig().GetSslCACert(),
		GcpCredentialJSON: at.req.GetStorageConfig().GetGcpCredentialJSON(),
		SslTlsMinVersion:  at.req.GetStorageConfig().GetSslTlsMinVersion(),
		UseCrc32CChecksum: at.req.GetStorageConfig().GetUseCrc32CChecksum(),
	}

	numRowsMap := make(map[int64]int64)
	segmentInsertFilesMap := make(map[int64]*clusteringpb.InsertFiles)
	segmentStorageInfos := make(map[int64]*clusteringpb.SegmentStorageInfo)

	for segID, stats := range at.req.GetSegmentStats() {
		numRows := stats.GetNumRows()
		numRowsMap[segID] = numRows
		log.Info("append segment rows", zap.Int64("segment id", segID), zap.Int64("rows", numRows))
		insertFiles := make([]string, 0, len(stats.GetLogIDs()))
		for _, id := range stats.GetLogIDs() {
			path := metautil.BuildInsertLogPath(at.req.GetStorageConfig().RootPath,
				at.req.GetCollectionID(), at.req.GetPartitionID(), segID, at.req.GetFieldID(), id)
			insertFiles = append(insertFiles, path)
		}
		segmentInsertFilesMap[segID] = &clusteringpb.InsertFiles{InsertFiles: insertFiles}

		if storageInfo := at.req.GetSegmentStorageInfos()[segID]; storageInfo != nil {
			segmentInsertFiles := buildClusteringSegmentInsertFiles(storageInfo.GetInsertLogs(), at.req.GetStorageConfig(), at.req.GetCollectionID(), at.req.GetPartitionID(), segID)
			segmentStorageInfos[segID] = &clusteringpb.SegmentStorageInfo{
				StorageVersion:     storageInfo.GetStorageVersion(),
				SegmentInsertFiles: segmentInsertFiles,
				Manifest:           storageInfo.GetManifest(),
			}
			log.Info("append segment storage info",
				zap.Int64("segmentID", segID),
				zap.Int64("storageVersion", storageInfo.GetStorageVersion()),
				zap.Int("insertLogFields", len(storageInfo.GetInsertLogs())),
				zap.Int("segmentInsertFileFields", len(segmentInsertFiles.GetFieldInsertFiles())),
				zap.String("manifest", storageInfo.GetManifest()))
		} else {
			log.Info("missing segment storage info", zap.Int64("segmentID", segID))
		}
	}
	log.Info("prepared analyze info storage infos",
		zap.Int("requestSegmentStats", len(at.req.GetSegmentStats())),
		zap.Int("requestSegmentStorageInfos", len(at.req.GetSegmentStorageInfos())),
		zap.Int("analyzeSegmentStorageInfos", len(segmentStorageInfos)))

	field := at.req.GetField()
	if field == nil || field.GetDataType() == schemapb.DataType_None {
		field = &schemapb.FieldSchema{
			FieldID:  at.req.GetFieldID(),
			Name:     at.req.GetFieldName(),
			DataType: at.req.GetFieldType(),
		}
	}

	analyzeInfo := &clusteringpb.AnalyzeInfo{
		ClusterID:           at.req.GetClusterID(),
		BuildID:             at.req.GetTaskID(),
		CollectionID:        at.req.GetCollectionID(),
		PartitionID:         at.req.GetPartitionID(),
		Version:             at.req.GetVersion(),
		Dim:                 at.req.GetDim(),
		StorageConfig:       storageConfig,
		NumClusters:         at.req.GetNumClusters(),
		TrainSize:           int64(float64(hardware.GetMemoryCount()) * at.req.GetMaxTrainSizeRatio()),
		MinClusterRatio:     at.req.GetMinClusterSizeRatio(),
		MaxClusterRatio:     at.req.GetMaxClusterSizeRatio(),
		MaxClusterSize:      at.req.GetMaxClusterSize(),
		NumRows:             numRowsMap,
		InsertFiles:         segmentInsertFilesMap,
		FieldSchema:         field,
		SegmentStorageInfos: segmentStorageInfos,
	}

	if at.req.GetEnableGlobalIndex() {
		statsPaths := globalindex.BuildStatsIndexPaths(at.req.GetStorageConfig().GetRootPath(),
			at.req.GetCollectionID(),
			at.req.GetPartitionID(),
			at.req.GetInsertChannel(),
			at.req.GetVersion(),
		)
		analyzePaths := globalindex.BuildAnalyzeArtifactPaths(at.req.GetStorageConfig().GetRootPath(),
			at.req.GetTaskID(),
			at.req.GetVersion(),
			at.req.GetCollectionID(),
			at.req.GetPartitionID(),
			at.req.GetFieldID(),
		)
		analyzeInfo.EnableGlobalIndex = true
		analyzeInfo.InsertChannel = at.req.GetInsertChannel()
		analyzeInfo.GlobalStatsIndexRoot = statsPaths.Root
		analyzeInfo.HeadIndexPath = statsPaths.HeadIndex
		analyzeInfo.ChunkMappingPath = statsPaths.ChunkMapping
		analyzeInfo.CompactionPlanPath = analyzePaths.CompactionPrePlan
		// knowhere.cluster.* pass-through; knowhere applies defaults for absent keys.
		analyzeInfo.ClusterParams = paramtable.Get().KnowhereConfig.GetClusterParams()
		analyzeInfo.CompactionMaxRows = paramtable.Get().DataCoordCfg.ClusteringCompactionGlobalCompactionMaxRows.GetAsInt64()
		if analyzeInfo.CompactionMaxRows <= 0 && analyzeInfo.GetDim() > 0 {
			vectorSize := typeutil.VectorTypeSize(analyzeInfo.GetFieldSchema().GetDataType())
			if vectorSize > 0 {
				maxBytes := paramtable.Get().DataCoordCfg.SegmentMaxSize.GetAsFloat() * 1024 * 1024 *
					paramtable.Get().DataCoordCfg.ClusteringCompactionMaxSegmentSizeRatio.GetAsFloat()
				analyzeInfo.CompactionMaxRows = int64(maxBytes / float64(vectorSize) / float64(analyzeInfo.GetDim()))
			}
		}
		analyzeInfo.CompactionMinRows = paramtable.Get().DataCoordCfg.ClusteringCompactionGlobalCompactionMinRows.GetAsInt64()
	}

	runner := at.selectAnalyzeRunner()
	at.analyze, err = runner(ctx, analyzeInfo)
	if err != nil {
		log.Error("failed to analyze data", zap.Error(err))
		return err
	}

	analyzeLatency := at.tr.RecordSpan()
	log.Info("analyze done", zap.Int64("analyze cost", analyzeLatency.Milliseconds()))
	return nil
}

func buildClusteringSegmentInsertFiles(fieldBinlogs []*datapb.FieldBinlog, storageConfig *indexpb.StorageConfig, collectionID int64, partitionID int64, segmentID int64) *clusteringpb.SegmentInsertFiles {
	insertLogs := make([]*clusteringpb.FieldInsertFiles, 0, len(fieldBinlogs))
	for _, insertLog := range fieldBinlogs {
		filePaths := make([]string, 0, len(insertLog.GetBinlogs()))
		columnGroupID := insertLog.GetFieldID()
		for _, binlog := range insertLog.GetBinlogs() {
			filePath := metautil.BuildInsertLogPath(storageConfig.GetRootPath(), collectionID, partitionID, segmentID, columnGroupID, binlog.GetLogID())
			filePaths = append(filePaths, filePath)
		}
		insertLogs = append(insertLogs, &clusteringpb.FieldInsertFiles{
			FilePaths: filePaths,
		})
	}
	return &clusteringpb.SegmentInsertFiles{
		FieldInsertFiles: insertLogs,
	}
}

func (at *analyzeTask) selectAnalyzeRunner() func(context.Context, *clusteringpb.AnalyzeInfo) (analyzecgowrapper.CodecAnalyze, error) {
	if at.req.GetEnableGlobalIndex() {
		if at.analyzeV2Runner != nil {
			return at.analyzeV2Runner
		}
		return analyzecgowrapper.AnalyzeV2
	}
	if at.analyzeRunner != nil {
		return at.analyzeRunner
	}
	return analyzecgowrapper.Analyze
}

func (at *analyzeTask) PostExecute(ctx context.Context) error {
	log := log.Ctx(ctx).With(zap.String("clusterID", at.req.GetClusterID()),
		zap.Int64("TaskID", at.req.GetTaskID()), zap.Int64("Collection", at.req.GetCollectionID()),
		zap.Int64("partitionID", at.req.GetPartitionID()), zap.Int64("fieldID", at.req.GetFieldID()))
	gc := func() {
		if err := at.analyze.Delete(); err != nil {
			log.Error("indexBuildTask Execute CIndexDelete failed", zap.Error(err))
		}
	}
	defer gc()

	centroidsFile, _, _, _, err := at.analyze.GetResult(len(at.req.GetSegmentStats()))
	if err != nil {
		log.Error("failed to upload index", zap.Error(err))
		return err
	}
	log.Info("analyze result", zap.String("centroidsFile", centroidsFile))

	var globalStatsIndexRoot, headIndexFile, compactionPlanFile string
	if at.req.GetEnableGlobalIndex() {
		statsPaths := globalindex.BuildStatsIndexPaths(at.req.GetStorageConfig().GetRootPath(),
			at.req.GetCollectionID(),
			at.req.GetPartitionID(),
			at.req.GetInsertChannel(),
			at.req.GetVersion(),
		)
		analyzePaths := globalindex.BuildAnalyzeArtifactPaths(at.req.GetStorageConfig().GetRootPath(),
			at.req.GetTaskID(),
			at.req.GetVersion(),
			at.req.GetCollectionID(),
			at.req.GetPartitionID(),
			at.req.GetFieldID(),
		)
		globalStatsIndexRoot = statsPaths.Root
		headIndexFile = statsPaths.HeadIndex
		compactionPlanFile = analyzePaths.CompactionPrePlan
	}

	at.manager.StoreAnalyzeFilesAndStatistic(at.req.GetClusterID(),
		at.req.GetTaskID(),
		centroidsFile,
		globalStatsIndexRoot,
		headIndexFile,
		compactionPlanFile)
	at.tr.Elapse("index building all done")
	log.Info("Successfully save analyze files")
	return nil
}

func (at *analyzeTask) OnEnqueue(ctx context.Context) error {
	at.queueDur = 0
	at.tr.RecordSpan()

	log.Ctx(ctx).Info("analyzeTask enqueued", zap.String("clusterID", at.req.GetClusterID()),
		zap.Int64("TaskID", at.req.GetTaskID()))
	return nil
}

func (at *analyzeTask) SetState(state indexpb.JobState, failReason string) {
	at.manager.StoreAnalyzeTaskState(at.req.GetClusterID(), at.req.GetTaskID(), state, failReason)
}

func (at *analyzeTask) GetState() indexpb.JobState {
	return at.manager.LoadAnalyzeTaskState(at.req.GetClusterID(), at.req.GetTaskID())
}

func (at *analyzeTask) Reset() {
	at.ident = ""
	at.ctx = nil
	at.cancel = nil
	at.req = nil
	at.tr = nil
	at.queueDur = 0
	at.manager = nil
}
