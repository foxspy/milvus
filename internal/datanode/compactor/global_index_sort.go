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
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	sio "io"
	"math"
	"os"
	"path"
	"sort"
	"sync"

	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/compaction"
	"github.com/milvus-io/milvus/internal/flushcommon/io"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/globalindex"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/log"
	"github.com/milvus-io/milvus/pkg/v3/objectstorage"
	"github.com/milvus-io/milvus/pkg/v3/proto/clusteringpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// useGlobalIndexSort reports whether this clustering compaction must lay out the
// output segments strictly sorted by (centroid_id, distance_to_centroid) so the
// global head-index can range-scan a whole centroid block. Only the vector
// clustering key + global index path needs it.
func (t *clusteringCompactionTask) useGlobalIndexSort() bool {
	return t.isVectorClusteringKey &&
		t.plan.GetEnableGlobalIndex() &&
		t.plan.GetGlobalStatsIndexRoot() != ""
}

// globalIndexSortKey is the per-row merge key carried alongside the spilled rows.
type globalIndexSortKey struct {
	centroidID int32
	distance   float32
}

func (k globalIndexSortKey) less(o globalIndexSortKey) bool {
	if k.centroidID != o.centroidID {
		return k.centroidID < o.centroidID
	}
	return k.distance < o.distance
}

const globalIndexSortKeySize = 8 // int32 centroidID + float32 distance

// globalIndexSpillRun is one local-disk sorted run: the rows of a single input
// segment that belong to a single group, sorted by (centroid_id, distance). The
// row data is stored as standard binlog(s) on local disk; the per-row merge keys
// are stored in a parallel sidecar file in the exact same row order.
type globalIndexSpillRun struct {
	inputSegmentID int64
	localSegments  []*datapb.CompactionSegment
	keysPath       string
	rowCount       int64
}

// globalIndexSpiller owns the local-disk staging area for the two-phase external
// merge sort. Phase 1 (concurrent, per input segment) appends sorted runs; Phase 2
// merges runs per group.
type globalIndexSpiller struct {
	task         *clusteringCompactionTask
	localCM      storage.ChunkManager
	localIO      io.BinlogIO
	root         string
	spillParams  compaction.Params
	spillStorage *indexpb.StorageConfig
	segAlloc     allocator.Interface
	logAlloc     allocator.Interface

	// subRunRowBudget bounds how many rows one mapper holds in memory before it sorts
	// and spills them as a sub-run, so a single large input segment never has to fit
	// in memory all at once (Phase 2 merges all sub-runs of a group regardless).
	subRunRowBudget int64
	// readBufferSize is the (small) per-run read buffer used in Phase 2, where many
	// run readers may be open at once.
	readBufferSize int64

	mu   sync.Mutex
	runs map[int][]*globalIndexSpillRun // groupID -> runs
}

// globalIndexSortOverheadFactor accounts for the in-memory Go representation of a row
// (a map[fieldID]interface{} plus boxed values) being several times larger than its
// serialized binlog size, which is what EstimateSizePerRecord returns.
const globalIndexSortOverheadFactor = 4

func newGlobalIndexSpiller(t *clusteringCompactionTask) *globalIndexSpiller {
	root := path.Join(localStorageRoot(), "global_index_compaction", fmt.Sprintf("%d", t.GetPlanID()))
	localCM := storage.NewLocalChunkManager(objectstorage.RootPath(root))
	// Spill runs are an internal temporary format: always write them as the legacy
	// storage v1 (Composite) binlog so reads/writes honor the local chunk manager
	// (the storage v2 packed/manifest reader bypasses the local downloader).
	spillParams := t.compactionParams
	spillParams.StorageVersion = storage.StorageV1
	// The local chunk manager writes/reads keys as literal filesystem paths, so the
	// spill binlog keys must already point under the dedicated spill dir. Rooting a
	// cloned storage config at `root` makes the binlog writer generate keys there and
	// keeps all spill artifacts isolated and removable in one shot.
	spillStorage := proto.Clone(t.compactionParams.StorageConfig).(*indexpb.StorageConfig)
	spillStorage.RootPath = root

	// Bound Phase 1 memory: split the compaction memory budget across concurrent
	// mappers, then convert bytes to a row count using the estimated in-memory row size.
	poolSize := int64(t.getWorkerPoolSize())
	if poolSize < 1 {
		poolSize = 1
	}
	rowBytes, err := typeutil.EstimateSizePerRecord(t.plan.GetSchema())
	if err != nil || rowBytes <= 0 {
		rowBytes = 1024
	}
	effRowBytes := int64(rowBytes) * globalIndexSortOverheadFactor
	budgetBytes := int64(float64(t.memoryLimit) * 0.7 / float64(poolSize))
	subRunRowBudget := budgetBytes / effRowBytes
	if subRunRowBudget < 4096 {
		subRunRowBudget = 4096
	}

	return &globalIndexSpiller{
		task:         t,
		localCM:      localCM,
		localIO:      io.NewBinlogIO(localCM),
		root:         root,
		spillParams:  spillParams,
		spillStorage: spillStorage,
		// Synthetic local-only id ranges: spill files never leave the local FS, so
		// they need not draw from the plan's pre-allocated segment/log id budget.
		segAlloc:        allocator.NewLocalAllocator(1, math.MaxInt64),
		logAlloc:        allocator.NewLocalAllocator(1, math.MaxInt64),
		subRunRowBudget: subRunRowBudget,
		readBufferSize:  4 << 20,
		runs:            make(map[int][]*globalIndexSpillRun),
	}
}

func (s *globalIndexSpiller) addRun(groupID int, run *globalIndexSpillRun) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.runs[groupID] = append(s.runs[groupID], run)
}

func (s *globalIndexSpiller) cleanup(ctx context.Context) {
	// The local chunk manager keys are literal filesystem paths under s.root, so
	// remove the directory tree directly rather than via the (root-relative) CM.
	if err := os.RemoveAll(s.root); err != nil {
		log.Ctx(ctx).Warn("failed to clean up global index spill dir",
			zap.String("root", s.root), zap.Error(err))
	}
}

// mappingGlobalIndexSorted is the global-index replacement for mapping(): it spills
// each input segment as per-group sorted runs (Phase 1), then merges each group into
// strictly (centroid, distance)-ordered output segments while recording the exact
// per-centroid row range, and uploads the chunk mapping (Phase 2).
func (t *clusteringCompactionTask) mappingGlobalIndexSorted(ctx context.Context,
) ([]*datapb.CompactionSegment, *storage.PartitionStatsSnapshot, error) {
	log := log.Ctx(ctx)
	inputSegments := t.plan.GetSegmentBinlogs()

	spiller := newGlobalIndexSpiller(t)
	defer spiller.cleanup(ctx)

	// Phase 1: concurrent per-segment sort + partition by group + local spill.
	futures := make([]*conc.Future[any], 0, len(inputSegments))
	for _, segment := range inputSegments {
		segmentClone := &datapb.CompactionSegmentBinlogs{
			SegmentID:      segment.SegmentID,
			Deltalogs:      segment.Deltalogs,
			FieldBinlogs:   segment.FieldBinlogs,
			StorageVersion: segment.StorageVersion,
			Manifest:       segment.GetManifest(),
		}
		future := t.mappingPool.Submit(func() (any, error) {
			return struct{}{}, t.spillSegment(ctx, spiller, segmentClone)
		})
		futures = append(futures, future)
	}
	if err := conc.AwaitAll(futures...); err != nil {
		return nil, nil, err
	}

	// Phase 2: per-group k-way merge into each group's final writer, concurrent across
	// groups. Each group writes its own output segments (independent MultiSegmentWriter,
	// mutex-guarded shared ID allocator) and records ranges into a private mapping
	// (centroids are disjoint across groups); the private mappings are merged afterwards.
	groupMappings := make([]globalindex.ChunkMapping, len(t.clusterBuffers))
	mergeFutures := make([]*conc.Future[any], 0, len(t.clusterBuffers))
	for groupID, buffer := range t.clusterBuffers {
		groupID, buffer := groupID, buffer
		local := make(globalindex.ChunkMapping)
		groupMappings[groupID] = local
		mergeFutures = append(mergeFutures, t.mappingPool.Submit(func() (any, error) {
			return struct{}{}, t.mergeGroup(ctx, spiller, groupID, buffer, local)
		}))
	}
	if err := conc.AwaitAll(mergeFutures...); err != nil {
		return nil, nil, err
	}
	mapping := make(globalindex.ChunkMapping)
	for _, groupMapping := range groupMappings {
		for centroidID, chunks := range groupMapping {
			mapping[centroidID] = chunks
		}
	}

	if err := t.flushAll(); err != nil {
		return nil, nil, err
	}
	if err := t.uploadSortedChunkMapping(ctx, mapping); err != nil {
		return nil, nil, err
	}

	resultSegments, resultPartitionStats := t.collectClusterBufferResults(ctx)
	log.Info("global index sorted mapping end",
		zap.Int("segmentFrom", len(inputSegments)),
		zap.Int("segmentTo", len(resultSegments)),
		zap.Int("centroidCount", len(mapping)))
	return resultSegments, resultPartitionStats, nil
}

// spillSegment reads one input segment, filters deleted/expired rows, partitions
// the survivors by group, sorts each partition by (centroid, distance), and spills
// each partition as a local sorted run plus its merge-key sidecar.
func (t *clusteringCompactionTask) spillSegment(ctx context.Context, spiller *globalIndexSpiller,
	segment *datapb.CompactionSegmentBinlogs,
) error {
	log := log.Ctx(ctx).With(zap.Int64("planID", t.GetPlanID()), zap.Int64("segmentID", segment.GetSegmentID()))

	delta, err := compaction.ComposeDeleteFromDeltalogs(ctx, t.primaryKeyField.DataType, segment,
		storage.WithDownloader(t.binlogIO.Download),
		storage.WithStorageConfig(t.compactionParams.StorageConfig),
	)
	if err != nil {
		return err
	}
	entityFilter := compaction.NewEntityFilter(delta, t.plan.GetCollectionTtl(), t.currentTime, segment.GetCommitTimestamp())

	mappingStats := &clusteringpb.ClusteringCentroidIdMappingStats{}
	offSetPath := t.segmentIDOffsetMapping[segment.SegmentID]
	offsetBytes, err := t.binlogIO.Download(ctx, []string{offSetPath})
	if err != nil {
		return err
	}
	if err = proto.Unmarshal(offsetBytes[0], mappingStats); err != nil {
		return err
	}
	idMapping := mappingStats.GetCentroidIdMapping()
	distances := mappingStats.GetDistanceToCentroid()

	var binlogNum int
	for _, b := range segment.GetFieldBinlogs() {
		if b != nil {
			binlogNum = len(b.GetBinlogs())
			break
		}
	}
	if binlogNum == 0 {
		log.Warn("compact wrong, all segments' binlogs are empty")
		return merr.WrapErrIllegalCompactionPlan()
	}

	rr, existingFields, err := newCompactionSegmentRecordReader(ctx, segment, t.plan.Schema, t.compactionParams.StorageConfig,
		storage.WithDownloader(t.binlogIO.Download),
		storage.WithCollectionID(t.GetCollection()),
		storage.WithVersion(segment.StorageVersion),
		storage.WithBufferSize(t.bufferSize),
		storage.WithStorageConfig(t.compactionParams.StorageConfig),
	)
	if err != nil {
		log.Warn("new binlog record reader wrong", zap.Error(err))
		return err
	}
	materializer, err := NewRecordMaterializer(t.plan.Schema, t.plan.Schema.GetFunctions(), existingFields)
	if err != nil {
		rr.Close()
		log.Warn("new record materializer wrong", zap.Error(err))
		return err
	}
	rr = newMaterializedRecordReader(rr, materializer)
	defer rr.Close()

	hasTTLField := t.ttlFieldID >= common.StartOfUserFieldID
	// partition rows by group; index = groupID. Rows accumulate up to a memory budget
	// and are then sorted+spilled as a sub-run so a large input segment never has to
	// fit in memory all at once.
	partitions := make(map[int][]globalIndexSortRow)
	var accumRows int64
	subRunIdx := 0
	flushSubRuns := func() error {
		for groupID, rows := range partitions {
			run, err := spiller.writeRun(ctx, segment.GetSegmentID(), groupID, subRunIdx, rows)
			if err != nil {
				return err
			}
			spiller.addRun(groupID, run)
		}
		partitions = make(map[int][]globalIndexSortRow)
		accumRows = 0
		subRunIdx++
		return nil
	}

	offset := int64(-1)
	for {
		r, err := rr.Next()
		if err != nil {
			if err == sio.EOF {
				break
			}
			log.Warn("compact wrong, failed to iter through data", zap.Error(err))
			return err
		}
		vs := make([]*storage.Value, r.Len())
		if err = storage.ValueDeserializerWithSchema(r, vs, t.plan.Schema, true); err != nil {
			log.Warn("compact wrong, failed to deserialize data", zap.Error(err))
			return err
		}
		for _, v := range vs {
			offset++
			row, ok := v.Value.(map[typeutil.UniqueID]interface{})
			if !ok {
				return merr.WrapErrServiceInternalMsg("unexpected error: row is not a map")
			}
			expireTs := int64(-1)
			if hasTTLField {
				if val, exists := row[t.ttlFieldID]; exists {
					if ts, ok := val.(int64); ok {
						expireTs = ts
					}
				}
			}
			if entityFilter.Filtered(v.PK.GetValue(), uint64(v.Timestamp), expireTs) {
				continue
			}
			if commitTs := segment.GetCommitTimestamp(); commitTs != 0 {
				v.Timestamp = int64(commitTs)
			}
			if offset < 0 || offset >= int64(len(idMapping)) {
				return merr.WrapErrServiceInternalMsg(
					"row offset out of centroid mapping range, segmentID=%d, offset=%d, mapping=%d",
					segment.GetSegmentID(), offset, len(idMapping))
			}
			centroidID := int32(idMapping[offset])
			groupID, ok := t.centroidGroupIndex[int(centroidID)]
			if !ok {
				return merr.WrapErrServiceInternalMsg(
					"centroid not covered by compaction plan, segmentID=%d, centroidID=%d",
					segment.GetSegmentID(), centroidID)
			}
			var distance float32
			if int(offset) < len(distances) {
				distance = distances[offset]
			}
			partitions[groupID] = append(partitions[groupID], globalIndexSortRow{
				value: v,
				key:   globalIndexSortKey{centroidID: centroidID, distance: distance},
			})
			accumRows++
			if accumRows >= spiller.subRunRowBudget {
				if err := flushSubRuns(); err != nil {
					return err
				}
			}
		}
	}

	if len(partitions) > 0 {
		if err := flushSubRuns(); err != nil {
			return err
		}
	}
	log.Info("spill segment end", zap.Int("subRuns", subRunIdx), zap.Int64("rowBudgetPerSubRun", spiller.subRunRowBudget))
	return nil
}

// globalIndexSortRow pairs a row with its merge key.
type globalIndexSortRow struct {
	value *storage.Value
	key   globalIndexSortKey
}

// writeRun sorts one (segment, group) partition by (centroid, distance) and spills
// it as a local binlog run plus a merge-key sidecar in the same row order.
func (s *globalIndexSpiller) writeRun(ctx context.Context, inputSegmentID int64, groupID, subRunIdx int,
	rows []globalIndexSortRow,
) (*globalIndexSpillRun, error) {
	sort.Slice(rows, func(i, j int) bool {
		return rows[i].key.less(rows[j].key)
	})

	t := s.task
	alloc := NewCompactionAllocator(s.segAlloc, s.logAlloc)
	// MaxInt64 segment size: never rotate, exactly one local segment per run.
	w, err := NewMultiSegmentWriter(ctx, s.localIO, alloc, math.MaxInt64, t.plan.GetSchema(),
		s.spillParams, t.plan.MaxSegmentRows, t.partitionID, t.collectionID, t.plan.Channel, 100,
		storage.WithBufferSize(t.bufferSize),
		storage.WithStorageConfig(s.spillStorage),
	)
	if err != nil {
		return nil, err
	}

	keyBuf := make([]byte, len(rows)*globalIndexSortKeySize)
	for i, row := range rows {
		if err := w.WriteValue(row.value); err != nil {
			return nil, err
		}
		binary.LittleEndian.PutUint32(keyBuf[i*globalIndexSortKeySize:], uint32(row.key.centroidID))
		binary.LittleEndian.PutUint32(keyBuf[i*globalIndexSortKeySize+4:], math.Float32bits(row.key.distance))
	}
	if err := w.Close(); err != nil {
		return nil, err
	}

	// Absolute path under the spill root (the local chunk manager writes literal paths).
	keysPath := path.Join(s.root, "keys", fmt.Sprintf("seg_%d_group_%d_sub_%d.keys", inputSegmentID, groupID, subRunIdx))
	if err := s.localCM.Write(ctx, keysPath, keyBuf); err != nil {
		return nil, merr.Wrapf(err, "write spill keys %q", keysPath)
	}
	t.writtenRowNum.Add(int64(len(rows)))
	return &globalIndexSpillRun{
		inputSegmentID: inputSegmentID,
		localSegments:  w.GetCompactionSegments(),
		keysPath:       keysPath,
		rowCount:       int64(len(rows)),
	}, nil
}

// mergeGroup k-way merges all runs of a group by (centroid, distance) and writes the
// fully sorted stream into the group's final writer, recording each centroid's exact
// contiguous row range per output segment into mapping.
func (t *clusteringCompactionTask) mergeGroup(ctx context.Context, spiller *globalIndexSpiller,
	groupID int, buffer *ClusterBuffer, mapping globalindex.ChunkMapping,
) error {
	runs := spiller.runs[groupID]
	if len(runs) == 0 {
		return nil
	}

	h := &spillMergeHeap{}
	heap.Init(h)
	readers := make([]*spillRunReader, 0, len(runs))
	defer func() {
		for _, rd := range readers {
			rd.close()
		}
	}()
	for _, run := range runs {
		rd, err := newSpillRunReader(ctx, spiller, t, run)
		if err != nil {
			return err
		}
		readers = append(readers, rd)
		ok, err := rd.advance()
		if err != nil {
			return err
		}
		if ok {
			heap.Push(h, rd)
		}
	}

	tracker := newRangeTracker(buffer.writer, mapping)
	for h.Len() > 0 {
		rd := heap.Pop(h).(*spillRunReader)
		if err := tracker.write(rd.curValue, rd.curKey.centroidID); err != nil {
			return err
		}
		ok, err := rd.advance()
		if err != nil {
			return err
		}
		if ok {
			heap.Push(h, rd)
		}
	}
	tracker.finish()
	return nil
}

// rangeTracker writes the merged, sorted rows into a group's MultiSegmentWriter and
// records the exact [offset, size) row range of each centroid in each output segment.
// Because the stream is globally sorted by centroid, every centroid forms a single
// contiguous range per segment (or two ranges if a segment rotation splits it).
// rangeWriter is the minimal writer behavior rangeTracker needs: it writes a value
// and reports which output segment the value landed in. *MultiSegmentWriter satisfies it.
type rangeWriter interface {
	WriteValue(v *storage.Value) error
	CurrentSegmentID() typeutil.UniqueID
}

type rangeTracker struct {
	writer  rangeWriter
	mapping globalindex.ChunkMapping

	segOffsets  map[int64]int64
	curSeg      int64
	curCentroid int32
	rangeStart  int64
	rangeEnd    int64
	open        bool
}

func newRangeTracker(writer rangeWriter, mapping globalindex.ChunkMapping) *rangeTracker {
	return &rangeTracker{
		writer:     writer,
		mapping:    mapping,
		segOffsets: make(map[int64]int64),
		curSeg:     -1,
	}
}

func (rt *rangeTracker) write(v *storage.Value, centroidID int32) error {
	if err := rt.writer.WriteValue(v); err != nil {
		return err
	}
	seg := rt.writer.CurrentSegmentID()
	off := rt.segOffsets[seg]
	rt.segOffsets[seg] = off + 1

	if rt.open && (seg != rt.curSeg || centroidID != rt.curCentroid) {
		rt.flush()
	}
	if !rt.open {
		rt.curSeg = seg
		rt.curCentroid = centroidID
		rt.rangeStart = off
		rt.open = true
	}
	rt.rangeEnd = off
	return nil
}

func (rt *rangeTracker) flush() {
	if !rt.open {
		return
	}
	rt.mapping[int64(rt.curCentroid)] = append(rt.mapping[int64(rt.curCentroid)], globalindex.Chunk{
		SegmentID: rt.curSeg,
		Offset:    rt.rangeStart,
		Size:      rt.rangeEnd - rt.rangeStart + 1,
	})
	rt.open = false
}

func (rt *rangeTracker) finish() {
	rt.flush()
}

// spillRunReader is a single-row cursor over one sorted run: it streams rows from the
// run's local binlog segment(s) in write order, zipped with the run's merge-key sidecar.
type spillRunReader struct {
	ctx     context.Context
	spiller *globalIndexSpiller
	task    *clusteringCompactionTask
	run     *globalIndexSpillRun

	keys     []globalIndexSortKey
	keyIdx   int
	segIdx   int
	rr       storage.RecordReader
	batch    []*storage.Value
	batchPos int

	curValue *storage.Value
	curKey   globalIndexSortKey
}

func newSpillRunReader(ctx context.Context, spiller *globalIndexSpiller, t *clusteringCompactionTask,
	run *globalIndexSpillRun,
) (*spillRunReader, error) {
	keyBytes, err := spiller.localCM.Read(ctx, run.keysPath)
	if err != nil {
		return nil, merr.Wrapf(err, "read spill keys %q", run.keysPath)
	}
	if int64(len(keyBytes)) != run.rowCount*globalIndexSortKeySize {
		return nil, merr.WrapErrServiceInternalMsg(
			"spill keys size mismatch, path=%s, want=%d, got=%d",
			run.keysPath, run.rowCount*globalIndexSortKeySize, len(keyBytes))
	}
	keys := make([]globalIndexSortKey, run.rowCount)
	for i := int64(0); i < run.rowCount; i++ {
		base := i * globalIndexSortKeySize
		keys[i] = globalIndexSortKey{
			centroidID: int32(binary.LittleEndian.Uint32(keyBytes[base:])),
			distance:   math.Float32frombits(binary.LittleEndian.Uint32(keyBytes[base+4:])),
		}
	}
	return &spillRunReader{
		ctx:     ctx,
		spiller: spiller,
		task:    t,
		run:     run,
		keys:    keys,
	}, nil
}

// advance loads the next row into curValue/curKey. Returns false at end of run.
func (rd *spillRunReader) advance() (bool, error) {
	for rd.batchPos >= len(rd.batch) {
		ok, err := rd.nextBatch()
		if err != nil {
			return false, err
		}
		if !ok {
			return false, nil
		}
	}
	rd.curValue = rd.batch[rd.batchPos]
	rd.batchPos++
	if rd.keyIdx >= len(rd.keys) {
		return false, merr.WrapErrServiceInternalMsg("spill run produced more rows than keys, path=%s", rd.run.keysPath)
	}
	rd.curKey = rd.keys[rd.keyIdx]
	rd.keyIdx++
	return true, nil
}

// nextBatch advances to the next record batch, opening the next local segment when
// the current reader is drained. Returns false when all local segments are exhausted.
func (rd *spillRunReader) nextBatch() (bool, error) {
	for {
		if rd.rr == nil {
			if rd.segIdx >= len(rd.run.localSegments) {
				return false, nil
			}
			seg := rd.run.localSegments[rd.segIdx]
			rd.segIdx++
			segBinlogs := &datapb.CompactionSegmentBinlogs{
				SegmentID:      seg.GetSegmentID(),
				FieldBinlogs:   seg.GetInsertLogs(),
				StorageVersion: seg.GetStorageVersion(),
				Manifest:       seg.GetManifest(),
			}
			rr, _, err := newCompactionSegmentRecordReader(rd.ctx, segBinlogs, rd.task.plan.Schema,
				rd.spiller.spillStorage,
				storage.WithDownloader(rd.spiller.localIO.Download),
				storage.WithCollectionID(rd.task.GetCollection()),
				storage.WithVersion(seg.GetStorageVersion()),
				storage.WithBufferSize(rd.spiller.readBufferSize),
				storage.WithStorageConfig(rd.spiller.spillStorage),
			)
			if err != nil {
				return false, err
			}
			rd.rr = rr
		}
		r, err := rd.rr.Next()
		if err != nil {
			if err == sio.EOF {
				rd.rr.Close()
				rd.rr = nil
				continue
			}
			return false, err
		}
		vs := make([]*storage.Value, r.Len())
		if err = storage.ValueDeserializerWithSchema(r, vs, rd.task.plan.Schema, true); err != nil {
			return false, err
		}
		if len(vs) == 0 {
			continue
		}
		rd.batch = vs
		rd.batchPos = 0
		return true, nil
	}
}

func (rd *spillRunReader) close() {
	if rd.rr != nil {
		rd.rr.Close()
		rd.rr = nil
	}
}

// spillMergeHeap is a min-heap of run readers ordered by their current head key.
type spillMergeHeap []*spillRunReader

func (h spillMergeHeap) Len() int            { return len(h) }
func (h spillMergeHeap) Less(i, j int) bool  { return h[i].curKey.less(h[j].curKey) }
func (h spillMergeHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *spillMergeHeap) Push(x interface{}) { *h = append(*h, x.(*spillRunReader)) }
func (h *spillMergeHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	*h = old[:n-1]
	return item
}

// uploadSortedChunkMapping validates and uploads the exact per-centroid chunk mapping
// to the remote global stats index root.
func (t *clusteringCompactionTask) uploadSortedChunkMapping(ctx context.Context, mapping globalindex.ChunkMapping) error {
	root := t.plan.GetGlobalStatsIndexRoot()
	if len(mapping) == 0 {
		if t.writtenRowNum.Load() == 0 {
			return nil
		}
		return merr.WrapErrServiceInternalMsg("global stats chunk mapping is empty, root=%s", root)
	}
	if err := mapping.Validate(); err != nil {
		return err
	}
	mappingBytes, err := json.Marshal(mapping)
	if err != nil {
		return merr.Wrap(err, "marshal global stats chunk mapping")
	}
	chunkMappingPath := path.Join(root, common.GlobalStatsChunkMapping)
	if err := t.binlogIO.Upload(ctx, map[string][]byte{chunkMappingPath: mappingBytes}); err != nil {
		return merr.Wrapf(err, "upload global stats chunk mapping %q", chunkMappingPath)
	}
	log.Ctx(ctx).Info("uploaded sorted global stats chunk mapping",
		zap.String("path", chunkMappingPath),
		zap.Int("centroidCount", len(mapping)),
		zap.Int("bytes", len(mappingBytes)))
	return nil
}

// collectClusterBufferResults gathers the finalized output segments and their
// partition stats from all cluster buffers (shared with the non-sorted path).
func (t *clusteringCompactionTask) collectClusterBufferResults(ctx context.Context,
) ([]*datapb.CompactionSegment, *storage.PartitionStatsSnapshot) {
	resultSegments := make([]*datapb.CompactionSegment, 0)
	resultPartitionStats := &storage.PartitionStatsSnapshot{
		SegmentStats: make(map[typeutil.UniqueID]storage.SegmentStats),
	}
	for _, buffer := range t.clusterBuffers {
		segments := buffer.GetCompactionSegments()
		resultSegments = append(resultSegments, segments...)
		for _, segment := range segments {
			resultPartitionStats.SegmentStats[segment.SegmentID] = storage.SegmentStats{
				FieldStats: []storage.FieldStats{buffer.clusteringKeyFieldStats.Clone()},
				NumRows:    int(segment.NumOfRows),
			}
		}
	}
	return resultSegments, resultPartitionStats
}

// localStorageRoot returns the node's local storage path used as the spill prefix.
func localStorageRoot() string {
	return paramtable.Get().LocalStorageCfg.Path.GetValue()
}
