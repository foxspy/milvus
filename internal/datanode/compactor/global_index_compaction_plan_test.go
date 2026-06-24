package compactor

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/flushcommon/io"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
)

type fakeGlobalPlanBinlogIO struct {
	io.BinlogIO
	files map[string][]byte
}

func (f fakeGlobalPlanBinlogIO) Download(ctx context.Context, paths []string) ([][]byte, error) {
	result := make([][]byte, 0, len(paths))
	for _, path := range paths {
		result = append(result, f.files[path])
	}
	return result, nil
}

func (f fakeGlobalPlanBinlogIO) AsyncDownload(ctx context.Context, paths []string) []*conc.Future[any] {
	return nil
}

func (f fakeGlobalPlanBinlogIO) Upload(ctx context.Context, kvs map[string][]byte) error {
	return nil
}

func (f fakeGlobalPlanBinlogIO) AsyncUpload(ctx context.Context, kvs map[string][]byte) []*conc.Future[any] {
	return nil
}

func TestLoadGlobalIVFCompactionPlan(t *testing.T) {
	ctx := context.Background()
	task := &clusteringCompactionTask{
		binlogIO: fakeGlobalPlanBinlogIO{files: map[string][]byte{
			"plan": []byte(`{"format":"cardinal_global_ivf_compaction_plan_v1","centroid_count":4,"groups":[{"group_id":0,"centroids":[0,2],"rows":10},{"group_id":1,"centroids":[1,3],"rows":5}]}`),
		}},
		plan: &datapb.CompactionPlan{
			CompactionPlanFile: "plan",
		},
	}

	groups, err := task.loadGlobalIVFCompactionPlan(ctx, 4)
	require.NoError(t, err)
	require.Equal(t, [][]int{{0, 2}, {1, 3}}, groups)
}

func TestLoadGlobalIVFCompactionPlanRejectsInvalidPlan(t *testing.T) {
	ctx := context.Background()
	for name, plan := range map[string]string{
		"count mismatch": `{"centroid_count":3,"groups":[{"group_id":0,"centroids":[0]}]}`,
		"out of range":   `{"centroid_count":2,"groups":[{"group_id":0,"centroids":[2]}]}`,
		"duplicated":     `{"centroid_count":2,"groups":[{"group_id":0,"centroids":[0]},{"group_id":1,"centroids":[0]}]}`,
		"missing":        `{"centroid_count":2,"groups":[{"group_id":0,"centroids":[0]}]}`,
		"empty groups":   `{"centroid_count":2,"groups":[]}`,
	} {
		t.Run(name, func(t *testing.T) {
			task := &clusteringCompactionTask{
				binlogIO: fakeGlobalPlanBinlogIO{files: map[string][]byte{
					"plan": []byte(plan),
				}},
				plan: &datapb.CompactionPlan{
					CompactionPlanFile: "plan",
				},
			}

			_, err := task.loadGlobalIVFCompactionPlan(ctx, 2)
			require.Error(t, err)
		})
	}
}
