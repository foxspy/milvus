package scheduler

import (
	"sort"
	"time"

	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

var _ schedulePolicy = &fifoPolicy{}

// newFIFOPolicy create a new fifo schedule policy.
func newFIFOPolicy() schedulePolicy {
	p := &fifoPolicy{
		queue: newMergeTaskQueue(""),
	}
	p.setTimestampOrdering(paramtable.Get().QueryNodeCfg.EnableTimestampOrdering.GetAsBool())
	return p
}

// fifoPolicy is a fifo policy with merge queue.
type fifoPolicy struct {
	queue        *mergeTaskQueue
	nextSequence uint64
}

// After initialization, setTimestampOrdering must run on the scheduler goroutine,
// between dispatches.
// Keeping arrival sequences lets disabling the switch restore FIFO even when
// several tasks were enqueued with the same batch timestamp.
func (p *fifoPolicy) setTimestampOrdering(enabled bool) {
	if enabled == (p.queue.less != nil) {
		return
	}
	p.queue.compactRemoved()
	if enabled {
		p.queue.less = func(left, right *queuedTask) bool {
			return left.Order().Before(right.Order())
		}
		sort.SliceStable(p.queue.tasks, func(i, j int) bool {
			return p.queue.less(p.queue.tasks[i], p.queue.tasks[j])
		})
	} else {
		p.queue.less = nil
		sort.Slice(p.queue.tasks, func(i, j int) bool {
			return p.queue.tasks[i].arrivalSequence < p.queue.tasks[j].arrivalSequence
		})
	}
}

func (p *fifoPolicy) Cleanup(now time.Time) []*queuedTask {
	return p.queue.cleanup(now)
}

// Push add a new task into scheduler, an error will be returned if scheduler reaches some limit.
func (p *fifoPolicy) Push(task *queuedTask) (int, error) {
	pt := paramtable.Get()

	// Try to merge task if task can merge.
	if t := tryIntoMergeTask(task.Task); t != nil {
		maxNQ := pt.QueryNodeCfg.MaxGroupNQ.GetAsInt64()
		nqMergeRatio := pt.QueryNodeCfg.NQMergeRatio.GetAsFloat()
		maxDeadlineMergeGap := pt.QueryNodeCfg.MaxDeadlineMergeGap.GetAsDurationByParse()
		if p.queue.tryMerge(task, maxNQ, nqMergeRatio, maxDeadlineMergeGap) {
			return 0, nil
		}
	}

	// Add a new task into queue.
	task.arrivalSequence = p.nextSequence
	p.nextSequence++
	p.queue.push(task)
	return 1, nil
}

// Pop get the task next ready to run.
func (p *fifoPolicy) Pop(now time.Time) *queuedTask {
	return p.queue.pop()
}

func (p *fifoPolicy) Peek(now time.Time) *queuedTask {
	return p.queue.front()
}

// Len get ready task counts.
func (p *fifoPolicy) Len() int {
	return p.queue.len()
}
