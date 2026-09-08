package scheduler

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

func TestUserTaskPollingPolicy(t *testing.T) {
	paramtable.Init()
	testCommonPolicyOperation(t, newUserTaskPollingPolicy())
	testCrossUserMerge(t, newUserTaskPollingPolicy())
}

func TestFIFOPolicy(t *testing.T) {
	paramtable.Init()
	testCommonPolicyOperation(t, newFIFOPolicy())
}

func TestFIFOPolicyOrdersByProxySequence(t *testing.T) {
	paramtable.Init()
	orders := []TaskOrder{
		{Timestamp: 30, MessageID: 300, SourceID: 3},
		{Timestamp: 10, MessageID: 100, SourceID: 1},
		{Timestamp: 20, MessageID: 200, SourceID: 2},
	}

	policyA := newFIFOPolicy()
	policyB := newFIFOPolicy()
	for _, i := range []int{0, 1, 2} {
		_, err := policyA.Push(newQueuedTask(newMockTask(mockTaskConfig{order: orders[i]}), time.Now()))
		assert.NoError(t, err)
	}
	for _, i := range []int{2, 0, 1} {
		_, err := policyB.Push(newQueuedTask(newMockTask(mockTaskConfig{order: orders[i]}), time.Now()))
		assert.NoError(t, err)
	}

	expected := []TaskOrder{orders[1], orders[2], orders[0]}
	for _, expectedOrder := range expected {
		assert.Equal(t, expectedOrder, policyA.Pop(time.Now()).Order())
		assert.Equal(t, expectedOrder, policyB.Pop(time.Now()).Order())
	}
}

func TestFIFOPolicyUsesMessageAndSourceAsStableTieBreakers(t *testing.T) {
	paramtable.Init()
	policy := newFIFOPolicy()
	orders := []TaskOrder{
		{Timestamp: 10, MessageID: 2, SourceID: 2},
		{Timestamp: 10, MessageID: 1, SourceID: 2},
		{Timestamp: 10, MessageID: 1, SourceID: 1},
	}
	for _, order := range orders {
		_, err := policy.Push(newQueuedTask(newMockTask(mockTaskConfig{order: order}), time.Now()))
		assert.NoError(t, err)
	}

	assert.Equal(t, TaskOrder{Timestamp: 10, MessageID: 1, SourceID: 1}, policy.Pop(time.Now()).Order())
	assert.Equal(t, TaskOrder{Timestamp: 10, MessageID: 1, SourceID: 2}, policy.Pop(time.Now()).Order())
	assert.Equal(t, TaskOrder{Timestamp: 10, MessageID: 2, SourceID: 2}, policy.Pop(time.Now()).Order())
}

func TestFIFOPolicyDoesNotMergeEarlierOrderIntoLaterTask(t *testing.T) {
	paramtable.Init()
	policy := newFIFOPolicy()
	later := newMockTask(mockTaskConfig{
		order:     TaskOrder{Timestamp: 20, MessageID: 20, SourceID: 1},
		mergeAble: true,
		nq:        1,
	})
	earlier := newMockTask(mockTaskConfig{
		order:     TaskOrder{Timestamp: 10, MessageID: 10, SourceID: 1},
		mergeAble: true,
		nq:        1,
	})

	added, err := policy.Push(newQueuedTask(later, time.Now()))
	assert.NoError(t, err)
	assert.Equal(t, 1, added)
	added, err = policy.Push(newQueuedTask(earlier, time.Now()))
	assert.NoError(t, err)
	assert.Equal(t, 1, added)
	assert.Equal(t, 2, policy.Len())
	assert.Equal(t, earlier.Order(), policy.Pop(time.Now()).Order())
}

func TestPolicyCleanupExpiredTasks(t *testing.T) {
	paramtable.Init()
	for name, policy := range map[string]schedulePolicy{
		"fifo":              newFIFOPolicy(),
		"user-task-polling": newUserTaskPollingPolicy(),
	} {
		t.Run(name, func(t *testing.T) {
			base := time.Now()
			ctx, cancel := context.WithDeadline(context.Background(), base.Add(10*time.Millisecond))
			defer cancel()

			added, err := policy.Push(newQueuedTask(newMockTask(mockTaskConfig{ctx: ctx, nq: 1}), base))
			assert.NoError(t, err)
			assert.Equal(t, 1, added)
			assert.Equal(t, 1, policy.Len())

			expired := policy.Cleanup(base.Add(20 * time.Millisecond))
			assert.Len(t, expired, 1)
			assert.Equal(t, 0, policy.Len())
			assert.False(t, policy.Pop(base.Add(20*time.Millisecond)).valid())
		})
	}
}

func TestPolicyCleanupCanceledTasks(t *testing.T) {
	paramtable.Init()
	for name, policy := range map[string]schedulePolicy{
		"fifo":              newFIFOPolicy(),
		"user-task-polling": newUserTaskPollingPolicy(),
	} {
		t.Run(name, func(t *testing.T) {
			base := time.Now()
			ctx, cancel := context.WithCancel(context.Background())

			added, err := policy.Push(newQueuedTask(newMockTask(mockTaskConfig{ctx: ctx, nq: 1}), base))
			assert.NoError(t, err)
			assert.Equal(t, 1, added)

			cancel()
			removed := policy.Cleanup(base)

			assert.Len(t, removed, 1)
			assert.Equal(t, 0, policy.Len())
			assert.False(t, policy.Pop(base).valid())
		})
	}
}

func testCrossUserMerge(t *testing.T, policy schedulePolicy) {
	userN := 10
	maxNQ := paramtable.Get().QueryNodeCfg.MaxGroupNQ.GetAsInt64()
	// Do not open cross user merge.
	n := userN * 4
	for i := 1; i <= n; i++ {
		username := fmt.Sprintf("user_%d", (i-1)%userN)
		task := newMockTask(mockTaskConfig{
			username:  username,
			nq:        maxNQ / 2,
			mergeAble: true,
		})
		policy.Push(newQueuedTask(task, time.Now()))
	}
	nAfterMerge := n / 2
	assert.Equal(t, nAfterMerge, policy.Len())
	for i := 1; i <= nAfterMerge; i++ {
		assert.True(t, policy.Pop(time.Now()).valid())
		assert.Equal(t, nAfterMerge-i, policy.Len())
	}

	// Open cross user grouping
	oldGrouping := paramtable.Get().QueryNodeCfg.SchedulePolicyEnableCrossUserGrouping.SwapTempValue("true")
	defer paramtable.Get().QueryNodeCfg.SchedulePolicyEnableCrossUserGrouping.SwapTempValue(oldGrouping)
	oldNQMergeRatio := paramtable.Get().QueryNodeCfg.NQMergeRatio.SwapTempValue("0")
	defer paramtable.Get().QueryNodeCfg.NQMergeRatio.SwapTempValue(oldNQMergeRatio)
	for i := 1; i <= n; i++ {
		username := fmt.Sprintf("user_%d", (i-1)%userN)
		task := newMockTask(mockTaskConfig{
			username:  username,
			nq:        maxNQ / 4,
			mergeAble: true,
		})
		policy.Push(newQueuedTask(task, time.Now()))
	}
	nAfterMerge = n / 4
	assert.Equal(t, nAfterMerge, policy.Len())
	for i := 1; i <= nAfterMerge; i++ {
		assert.True(t, policy.Pop(time.Now()).valid())
		assert.Equal(t, nAfterMerge-i, policy.Len())
	}
}

// testCommonPolicyOperation
func testCommonPolicyOperation(t *testing.T, policy schedulePolicy) {
	// Empty policy assertion.
	assert.Equal(t, 0, policy.Len())
	assert.False(t, policy.Pop(time.Now()).valid())
	assert.Equal(t, 0, policy.Len())

	// Test no merge push pop.
	n := 50
	userN := 10
	// Test Push
	for i := 1; i <= n; i++ {
		username := fmt.Sprintf("user_%d", (i-1)%userN)
		task := newMockTask(mockTaskConfig{
			username: username,
		})
		policy.Push(newQueuedTask(task, time.Now()))
		assert.Equal(t, i, policy.Len())
	}
	// Test Pop
	for i := 1; i <= n; i++ {
		assert.True(t, policy.Pop(time.Now()).valid())
		assert.Equal(t, n-i, policy.Len())
	}

	// Test with merge
	maxNQ := paramtable.Get().QueryNodeCfg.MaxGroupNQ.GetAsInt64()
	// cannot merge if the nq is gte than maxNQ
	for i := 1; i <= n; i++ {
		username := fmt.Sprintf("user_%d", (i-1)%userN)
		task := newMockTask(mockTaskConfig{
			username:  username,
			nq:        maxNQ,
			mergeAble: true,
		})
		policy.Push(newQueuedTask(task, time.Now()))
	}
	assert.Equal(t, n, policy.Len())
	for i := 1; i <= n; i++ {
		assert.True(t, policy.Pop(time.Now()).valid())
		assert.Equal(t, n-i, policy.Len())
	}

	// Merge half MaxNQ
	n = userN * 2
	for i := 1; i <= n; i++ {
		username := fmt.Sprintf("user_%d", (i-1)%userN)
		task := newMockTask(mockTaskConfig{
			username:  username,
			nq:        maxNQ / 2,
			mergeAble: true,
		})
		policy.Push(newQueuedTask(task, time.Now()))
	}
	nAfterMerge := n / 2
	assert.Equal(t, nAfterMerge, policy.Len())
	for i := 1; i <= nAfterMerge; i++ {
		assert.True(t, policy.Pop(time.Now()).valid())
		assert.Equal(t, nAfterMerge-i, policy.Len())
	}
}
