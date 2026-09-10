package scheduler

import (
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

type orderingTestTask struct {
	Task
	beforeExecute func()
}

func (t *orderingTestTask) PreExecute() error {
	t.beforeExecute()
	return t.Task.PreExecute()
}

func TestSchedulerTimestampOrderingHotUpdate(t *testing.T) {
	paramtable.Init()
	params := paramtable.Get()
	key := params.QueryNodeCfg.EnableTimestampOrdering.Key
	old := params.QueryNodeCfg.EnableTimestampOrdering.GetValue()
	t.Cleanup(func() { require.NoError(t, params.Save(key, old)) })
	for _, tc := range []struct {
		name        string
		beforeStart bool
		updates     []bool
		userPolling bool
		expected    []int
	}{
		{name: "default FIFO", expected: []int{0, 1, 2, 3}},
		{name: "enable with pending tasks", updates: []bool{true}, expected: []int{1, 3, 2, 0}},
		{name: "disable with pending tasks", beforeStart: true, updates: []bool{false}, expected: []int{0, 1, 2, 3}},
		{name: "repeated transitions", updates: []bool{true, false, true}, expected: []int{1, 3, 2, 0}},
		{name: "change before Start", beforeStart: true, expected: []int{1, 3, 2, 0}},
		{name: "user polling unaffected", updates: []bool{true}, userPolling: true, expected: []int{0, 1, 2, 3}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			require.NoError(t, params.Save(key, "false"))
			policy := newFIFOPolicy()
			if tc.userPolling {
				policy = newUserTaskPollingPolicy()
			}
			s := newScheduler(policy)
			// Construct with the switch off, then change it before starting.
			require.NoError(t, params.Save(key, strconv.FormatBool(tc.beforeStart)))
			s.Start()
			t.Cleanup(s.Stop)
			started := make(chan struct{})
			release := make(chan struct{})
			unblock := sync.OnceFunc(func() { close(release) })
			t.Cleanup(unblock)
			blocker := &orderingTestTask{
				Task: newMockTask(mockTaskConfig{executeCost: time.Nanosecond}),
				beforeExecute: func() {
					close(started)
					<-release
				},
			}
			require.NoError(t, s.Add(blocker))
			select {
			case <-started:
			case <-time.After(5 * time.Second):
				t.Fatal("executor did not accept the blocker")
			}

			// The executor has already accepted the blocker; all subsequent tasks
			// remain in the scheduler while its current Peek can be displaced.
			dispatched := make(chan int, 4)
			tasks := make([]Task, 0, 4)
			for i, timestamp := range []uint64{30, 10, 20, 10} {
				task := &orderingTestTask{
					Task: newMockTask(mockTaskConfig{order: TaskOrder{Timestamp: timestamp}, executeCost: time.Nanosecond}),
					beforeExecute: func() {
						dispatched <- i
					},
				}
				tasks = append(tasks, task)
				require.NoError(t, s.Add(task))
			}
			require.Equal(t, int64(4), s.GetWaitingTaskTotal())
			for _, enabled := range tc.updates {
				require.NoError(t, params.Save(key, strconv.FormatBool(enabled)))
			}
			unblock()
			for _, expected := range tc.expected {
				select {
				case actual := <-dispatched:
					require.Equal(t, expected, actual)
				case <-time.After(5 * time.Second):
					t.Fatal("task was not dispatched after the ordering update")
				}
			}
			for _, task := range tasks {
				require.NoError(t, task.Wait())
			}
			require.NoError(t, blocker.Wait())
			require.Eventually(t, func() bool { return s.GetWaitingTaskTotal() == 0 }, time.Second, time.Millisecond)
		})
	}
}

func TestSchedulerTimestampOrderingConcurrentUpdatesAndStop(t *testing.T) {
	paramtable.Init()
	params := paramtable.Get()
	key := params.QueryNodeCfg.EnableTimestampOrdering.Key
	old := params.QueryNodeCfg.EnableTimestampOrdering.GetValue()
	t.Cleanup(func() { require.NoError(t, params.Save(key, old)) })
	s := newScheduler(newFIFOPolicy())
	s.Start()
	stop := sync.OnceFunc(s.Stop)
	t.Cleanup(stop)
	const count = 100
	dispatched := make(chan int, count)
	updatesDone := make(chan struct{})
	go func() {
		defer close(updatesDone)
		for i := 0; i < count; i++ {
			assert.NoError(t, params.Save(key, strconv.FormatBool(i%2 == 0)))
		}
	}()
	tasks := make([]Task, 0, count)
	for i := 0; i < count; i++ {
		task := &orderingTestTask{
			Task:          newMockTask(mockTaskConfig{order: TaskOrder{Timestamp: uint64(count - i)}, executeCost: time.Nanosecond}),
			beforeExecute: func() { dispatched <- i },
		}
		tasks = append(tasks, task)
		require.NoError(t, s.Add(task))
	}
	stop()
	select {
	case <-updatesDone:
	case <-time.After(5 * time.Second):
		t.Fatal("ordering update did not finish during shutdown")
	}
	seen := make(map[int]bool, count)
	for i := 0; i < count; i++ {
		select {
		case id := <-dispatched:
			require.False(t, seen[id], "task dispatched more than once")
			seen[id] = true
		case <-time.After(5 * time.Second):
			t.Fatal("task lost during ordering update")
		}
	}
	for _, task := range tasks {
		require.NoError(t, task.Wait())
	}
	require.Zero(t, s.GetWaitingTaskTotal())
	require.Zero(t, s.GetWaitingTaskTotalNQ())
}
