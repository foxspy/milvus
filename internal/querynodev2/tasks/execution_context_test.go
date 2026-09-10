package tasks

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/util/searchutil/scheduler"
)

func TestTaskExecutionContextCancellation(t *testing.T) {
	parent := context.Background()
	execution, cancel := context.WithCancel(parent)
	cancel()
	for _, tc := range []struct {
		name     string
		task     scheduler.Task
		isSearch bool
	}{
		{"search", &SearchTask{ctx: parent}, true},
		{"streaming search", &StreamingSearchTask{SearchTask: SearchTask{ctx: parent}}, true},
		{"query", &QueryTask{ctx: parent}, false},
		{"streaming query", &QueryStreamTask{ctx: parent}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.isSearch, tc.task.IsSearch())
			// The canceled execution context must be observed before touching
			// collection/segment state, although the original context is live.
			require.ErrorIs(t, tc.task.Execute(execution), context.Canceled)
			require.NoError(t, tc.task.Context().Err())
		})
	}
}
