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
	"context"
	"encoding/json"
	"sort"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/util/globalindex"
	"github.com/milvus-io/milvus/pkg/v3/log"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

type headIndexSearcher interface {
	Search(ctx context.Context, req *internalpb.SearchRequest, topK int64) ([][]int64, error)
}

type loadedGlobalStatsIndex struct {
	root              string
	headIndexFile     string
	chunkMappingFile  string
	chunkMapping      globalindex.ChunkMapping
	headIndexSearcher headIndexSearcher
}

func globalStatsInfoFromLoadInfo(info *querypb.SegmentLoadInfo) (root string, headIndexFile string, chunkMappingFile string, ok bool) {
	root = info.GetGlobalStatsIndexRoot()
	headIndexFile = info.GetHeadIndexFile()
	chunkMappingFile = info.GetChunkMappingFile()
	ok = root != "" || headIndexFile != "" || chunkMappingFile != ""
	return
}

func (sd *shardDelegator) loadGlobalStatsIndexes(ctx context.Context, infos []*querypb.SegmentLoadInfo) error {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", sd.collectionID),
		zap.String("channel", sd.vchannelName),
	)

	type loadTarget struct {
		root             string
		headIndexFile    string
		chunkMappingFile string
	}
	targets := make(map[string]loadTarget)
	segmentRoots := make(map[int64]string)

	for _, info := range infos {
		root, headIndexFile, chunkMappingFile, ok := globalStatsInfoFromLoadInfo(info)
		if !ok {
			continue
		}
		if root == "" || headIndexFile == "" || chunkMappingFile == "" {
			return merr.WrapErrServiceInternalMsg(
				"global stats index metadata is incomplete, segmentID=%d, root=%q, headIndexFile=%q, chunkMappingFile=%q",
				info.GetSegmentID(),
				root,
				headIndexFile,
				chunkMappingFile)
		}
		targets[root] = loadTarget{
			root:             root,
			headIndexFile:    headIndexFile,
			chunkMappingFile: chunkMappingFile,
		}
		segmentRoots[info.GetSegmentID()] = root
	}
	if len(targets) == 0 {
		return nil
	}

	loaded := make(map[string]*loadedGlobalStatsIndex, len(targets))
	for root, target := range targets {
		if cached := sd.getLoadedGlobalStatsIndex(root, target.headIndexFile, target.chunkMappingFile); cached != nil {
			loaded[root] = cached
			continue
		}

		chunkMappingBytes, err := sd.chunkManager.Read(ctx, target.chunkMappingFile)
		if err != nil {
			return merr.Wrapf(err, "read global chunk mapping %q", target.chunkMappingFile)
		}
		var mapping globalindex.ChunkMapping
		if err := json.Unmarshal(chunkMappingBytes, &mapping); err != nil {
			return merr.WrapErrServiceInternalErr(err, "parse global chunk mapping %q", target.chunkMappingFile)
		}
		if err := mapping.Validate(); err != nil {
			return merr.Wrap(err, "validate global chunk mapping")
		}
		headIndexSearcher, err := newHeadIndexSearcherFromPath(target.headIndexFile)
		if err != nil {
			return merr.Wrap(err, "load global head index")
		}

		loaded[root] = &loadedGlobalStatsIndex{
			root:              target.root,
			headIndexFile:     target.headIndexFile,
			chunkMappingFile:  target.chunkMappingFile,
			chunkMapping:      mapping,
			headIndexSearcher: headIndexSearcher,
		}
	}

	sd.globalStatsMut.Lock()
	defer sd.globalStatsMut.Unlock()
	for root, statsIndex := range loaded {
		sd.globalStatsIndexes[root] = statsIndex
		log.Info("loaded global stats index",
			zap.String("root", root),
			zap.String("headIndexFile", statsIndex.headIndexFile),
			zap.String("chunkMappingFile", statsIndex.chunkMappingFile),
			zap.Int("centroidCount", len(statsIndex.chunkMapping)))
	}
	for segmentID, root := range segmentRoots {
		sd.segmentGlobalStats[segmentID] = root
	}
	return nil
}

func (sd *shardDelegator) getLoadedGlobalStatsIndex(root string, headIndexFile string, chunkMappingFile string) *loadedGlobalStatsIndex {
	sd.globalStatsMut.RLock()
	defer sd.globalStatsMut.RUnlock()
	cached := sd.globalStatsIndexes[root]
	if cached == nil {
		return nil
	}
	if cached.headIndexFile != headIndexFile || cached.chunkMappingFile != chunkMappingFile {
		return nil
	}
	return cached
}

func (sd *shardDelegator) releaseGlobalStatsSegments(segmentIDs ...int64) {
	if len(segmentIDs) == 0 {
		return
	}
	sd.globalStatsMut.Lock()
	defer sd.globalStatsMut.Unlock()
	for _, segmentID := range segmentIDs {
		delete(sd.segmentGlobalStats, segmentID)
	}
}

func (sd *shardDelegator) globalStatsRootsForSealedSegments(sealed []SnapshotItem) []string {
	roots := make(map[string]struct{})
	sd.globalStatsMut.RLock()
	defer sd.globalStatsMut.RUnlock()
	for _, item := range sealed {
		for _, segment := range item.Segments {
			root := sd.segmentGlobalStats[segment.SegmentID]
			if root == "" {
				continue
			}
			if sd.globalStatsIndexes[root] == nil {
				continue
			}
			roots[root] = struct{}{}
		}
	}
	result := make([]string, 0, len(roots))
	for root := range roots {
		result = append(result, root)
	}
	sort.Strings(result)
	return result
}

func (sd *shardDelegator) validateGlobalStatsChunkMappings(roots []string, sealed []SnapshotItem, sealedRowCount map[int64]int64) error {
	if len(roots) == 0 {
		return nil
	}
	sd.globalStatsMut.RLock()
	defer sd.globalStatsMut.RUnlock()
	for _, root := range roots {
		statsIndex := sd.globalStatsIndexes[root]
		if statsIndex == nil {
			return merr.WrapErrServiceInternalMsg("global stats index %q is not loaded", root)
		}
		if err := ValidateGlobalIndexChunkPlan(statsIndex.chunkMapping, sealed, sealedRowCount); err != nil {
			return merr.Wrap(err, "validate global stats chunk mapping")
		}
	}
	return nil
}
