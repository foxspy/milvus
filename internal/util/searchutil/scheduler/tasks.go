package scheduler

import (
	"context"
	"time"

	"github.com/milvus-io/milvus/pkg/v2/proto/internalpb"
)

const (
	schedulePolicyNameFIFO            = "fifo"
	schedulePolicyNameUserTaskPolling = "user-task-polling"
)

// TaskOrder is the global order assigned by Proxy before a read request is
// fanned out to QueryNodes. Every sub-task of one request carries the same
// order, so independently receiving QueryNodes can make the same scheduling
// decision for tasks that are still waiting in their queues.
type TaskOrder struct {
	Timestamp uint64
	MessageID int64
	SourceID  int64
}

// Before reports whether o must be scheduled before other.
func (o TaskOrder) Before(other TaskOrder) bool {
	if o.Timestamp != other.Timestamp {
		return o.Timestamp < other.Timestamp
	}
	if o.MessageID != other.MessageID {
		return o.MessageID < other.MessageID
	}
	return o.SourceID < other.SourceID
}

// NewScheduler create a scheduler by policyName.
func NewScheduler(policyName string) Scheduler {
	switch policyName {
	case "":
		fallthrough
	case schedulePolicyNameFIFO:
		return newScheduler(
			newFIFOPolicy(),
		)
	case schedulePolicyNameUserTaskPolling:
		return newScheduler(
			newUserTaskPollingPolicy(),
		)
	default:
		panic("invalid schedule task policy")
	}
}

// tryIntoMergeTask convert inner task into MergeTask,
// Return nil if inner task is not a MergeTask.
func tryIntoMergeTask(t Task) MergeTask {
	if mt, ok := t.(MergeTask); ok {
		return mt
	}
	return nil
}

type Scheduler interface {
	// Add a new task into scheduler, follow some constraints.
	// 1. Error will be returned if scheduler reaches some limit.
	// 2. Error will be returned if task context is canceled while waiting to be accepted.
	// 3. Concurrent safe.
	Add(task Task) error

	// Start schedule the owned task asynchronously and continuously.
	// Shall be called only once
	Start()

	// Stop make scheduler deny all incoming tasks
	// and cleans up all related resources
	Stop()

	// GetWaitingTaskTotalNQ
	GetWaitingTaskTotalNQ() int64

	// GetWaitingTaskTotal
	GetWaitingTaskTotal() int64
}

// schedulePolicy is the policy of scheduler.
type schedulePolicy interface {
	// Cleanup removes queued tasks whose context deadline has been reached.
	// Removed tasks are returned to scheduler for error notification.
	Cleanup(now time.Time) []*queuedTask

	// Push add a new task into scheduler.
	// Return the count of new task added (task may be chunked or merged)
	// 0 and an error will be returned if scheduler reaches some limit.
	Push(task *queuedTask) (int, error)

	// Pop get the task next ready to run.
	Pop(now time.Time) *queuedTask

	// Peek gets the next ready task without removing it. This lets the policy
	// reorder the task while the executor is not yet ready to receive it.
	Peek(now time.Time) *queuedTask

	Len() int
}

type queuedTask struct {
	Task

	enqueueTime time.Time
}

func newQueuedTask(task Task, enqueueTime time.Time) *queuedTask {
	return &queuedTask{
		Task:        task,
		enqueueTime: enqueueTime,
	}
}

func (t *queuedTask) queueDuration(now time.Time) time.Duration {
	if !t.valid() || t.enqueueTime.IsZero() {
		return 0
	}
	return now.Sub(t.enqueueTime)
}

func (t *queuedTask) valid() bool {
	return t != nil && t.Task != nil
}

func (t *queuedTask) cleanupReady(now time.Time) bool {
	if !t.valid() {
		return false
	}
	if t.Context().Err() != nil {
		return true
	}
	deadline, ok := t.Context().Deadline()
	return ok && !now.Before(deadline)
}

func cleanupTaskError(task *queuedTask) error {
	if err := task.Context().Err(); err != nil {
		return err
	}
	return context.DeadlineExceeded
}

// MergeTask is a Task which can be merged with other task
type MergeTask interface {
	Task

	// MergeWith other task, return true if merge success.
	// After success, the task merged should be dropped.
	MergeWith(Task) bool

	// MinNQ returns the minimum NQ among the original tasks in this merged task.
	MinNQ() int64
}

// A task is execute unit of scheduler.
type Task interface {
	Context() context.Context

	// Order returns the Proxy-assigned global order of the original request.
	Order() TaskOrder

	// Return the username which task is belong to.
	// Return "" if the task do not contain any user info.
	Username() string

	// Return whether the task would be running on GPU.
	IsGpuIndex() bool

	// PreExecute the task, only call once.
	PreExecute() error

	// Execute the task, only call once.
	Execute() error

	// Done notify the task finished.
	Done(err error)

	// Wait for task finish.
	// Concurrent safe.
	Wait() error

	// Return the NQ of task.
	NQ() int64

	SearchResult() *internalpb.SearchResults
}
