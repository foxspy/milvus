package scheduler

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/cockroachdb/errors"
	"go.uber.org/atomic"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/querynodev2/collector"
	"github.com/milvus-io/milvus/pkg/v2/config"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/util/conc"
	"github.com/milvus-io/milvus/pkg/v2/util/lifetime"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/metricsinfo"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

const (
	maxReceiveChanBatchConsumeNum = 100

	readTaskQueueOutcomeScheduled = "scheduled"
	readTaskQueueOutcomeExpired   = "expired"
)

// newScheduler create a scheduler with given schedule policy.
func newScheduler(policy schedulePolicy) Scheduler {
	maxReadConcurrency := paramtable.Get().QueryNodeCfg.MaxReadConcurrency.GetAsInt()
	log.Info("query node use concurrent safe scheduler", zap.Int("max_concurrency", maxReadConcurrency))
	s := &scheduler{
		policy:           policy,
		receiveChan:      make(chan addTaskReq),
		execChan:         make(chan Task),
		pool:             conc.NewPool[any](maxReadConcurrency, conc.WithPreAlloc(true)),
		gpuPool:          conc.NewPool[any](paramtable.Get().QueryNodeCfg.MaxGpuReadConcurrency.GetAsInt(), conc.WithPreAlloc(true)),
		schedulerCounter: schedulerCounter{},
		lifetime:         lifetime.NewLifetime(lifetime.Initializing),
	}
	s.watchDynamicDeadline()
	return s
}

type addTaskReq struct {
	task Task
	err  chan<- error
}

type timestampOrderingUpdate struct {
	done chan struct{}
}

// scheduler is a general concurrent safe scheduler implementation by wrapping a schedule policy.
type scheduler struct {
	policy      schedulePolicy
	receiveChan chan addTaskReq
	execChan    chan Task
	pool        *conc.Pool[any]
	gpuPool     *conc.Pool[any]

	orderingUpdates chan timestampOrderingUpdate
	orderingHandler config.EventHandler

	searchLatencies executionLatencyWindow
	queryLatencies  executionLatencyWindow
	// Serialize switch transitions with deadline decisions and sample recording.
	deadlineMu         sync.RWMutex
	deadlineEnabled    bool
	deadlineGeneration uint64
	deadlineHandler    config.EventHandler

	// wg is the waitgroup for internal worker goroutine
	wg sync.WaitGroup
	// lifetime controls scheduler State & make sure all requests accepted will be processed
	lifetime lifetime.Lifetime[lifetime.State]

	schedulerCounter
}

// Add a new task into scheduler,
// error will be returned if scheduler reaches some limit.
func (s *scheduler) Add(task Task) (err error) {
	if err := s.lifetime.Add(lifetime.IsWorking); err != nil {
		return err
	}
	defer s.lifetime.Done()

	errCh := make(chan error, 1)

	req := addTaskReq{
		task: task,
		err:  errCh,
	}

	// start a new in queue span and send task to add chan
	ctx := task.Context()
	select {
	case s.receiveChan <- req:
		err = <-errCh
	case <-ctx.Done():
		err = ctx.Err()
	}

	return
}

// Start schedule the owned task asynchronously and continuously.
// Start should be only call once.
func (s *scheduler) Start() {
	if _, ok := s.policy.(*fifoPolicy); ok {
		s.orderingUpdates = make(chan timestampOrderingUpdate)
	}
	s.wg.Add(2)

	// Start a background task executing loop.
	go s.exec()

	// Begin to schedule tasks.
	go s.schedule()

	s.lifetime.SetState(lifetime.Working)
	s.watchTimestampOrdering()
}

func (s *scheduler) Stop() {
	s.lifetime.SetState(lifetime.Stopped)
	// Wait for accepted Add calls and ordering updates before stopping the loop.
	s.lifetime.Wait()
	// close receiveChan start stopping process for `schedule`
	close(s.receiveChan)
	// wait workers quit
	s.wg.Wait()
	if s.pool != nil {
		s.pool.Release()
	}
	if s.gpuPool != nil {
		s.gpuPool.Release()
	}
	s.unwatchDynamicDeadline()
	if s.orderingHandler != nil {
		paramtable.Get().Unwatch(paramtable.Get().QueryNodeCfg.EnableTimestampOrdering.Key, s.orderingHandler)
	}
}

func (s *scheduler) watchTimestampOrdering() {
	if s.orderingUpdates == nil {
		return
	}
	update := func() {
		if err := s.lifetime.Add(lifetime.IsWorking); err != nil {
			return
		}
		defer s.lifetime.Done()
		req := timestampOrderingUpdate{done: make(chan struct{})}
		s.orderingUpdates <- req
		<-req.done
	}
	s.orderingHandler = config.NewHandler(fmt.Sprintf("qn.scheduler.timestampOrdering.%p", s), func(*config.Event) { update() })
	paramtable.Get().Watch(paramtable.Get().QueryNodeCfg.EnableTimestampOrdering.Key, s.orderingHandler)
	// Apply changes made between construction and Start, including events that
	// raced with registering the handler. The loop is running before we wait.
	update()
}

func (s *scheduler) applyTimestampOrdering(req timestampOrderingUpdate) {
	// Read the latest effective value on the queue's owning goroutine. Reading
	// here also prevents concurrent config callbacks from applying stale values.
	// Conversion caches can still contain the old value during event dispatch.
	enabled, _ := strconv.ParseBool(paramtable.Get().QueryNodeCfg.EnableTimestampOrdering.GetValue())
	s.policy.(*fifoPolicy).setTimestampOrdering(enabled)
	close(req.done)
}

func (s *scheduler) watchDynamicDeadline() {
	item := &paramtable.Get().QueryNodeCfg.EnableDynamicDeadline
	update := func() {
		s.deadlineMu.Lock()
		defer s.deadlineMu.Unlock()
		// Read the effective value directly: conversion caches can still hold
		// the old value until all config event handlers have run.
		enabled, _ := strconv.ParseBool(item.GetValue())
		if s.deadlineEnabled != enabled {
			s.deadlineEnabled = enabled
			s.deadlineGeneration++
			s.searchLatencies.reset()
			s.queryLatencies.reset()
		}
	}
	s.deadlineHandler = config.NewHandler(fmt.Sprintf("qn.scheduler.dynamicDeadline.%p", s), func(*config.Event) { update() })
	paramtable.Get().Watch(item.Key, s.deadlineHandler)
	update()
}

func (s *scheduler) unwatchDynamicDeadline() {
	if s.deadlineHandler != nil {
		paramtable.Get().Unwatch(paramtable.Get().QueryNodeCfg.EnableDynamicDeadline.Key, s.deadlineHandler)
	}
	s.deadlineMu.Lock()
	defer s.deadlineMu.Unlock()
	s.deadlineEnabled = false
	s.deadlineGeneration++
	s.searchLatencies.reset()
	s.queryLatencies.reset()
}

// schedule the owned task asynchronously and continuously.
func (s *scheduler) schedule() {
	defer s.wg.Done()
	for {
		s.setupReadyLenMetric()

		now := time.Now()
		task, execChan := s.setupExecListener(now)
		var execTask Task
		if task.valid() {
			execTask = task.Task
		}

		select {
		case req := <-s.orderingUpdates:
			s.applyTimestampOrdering(req)
		case req, ok := <-s.receiveChan:
			if !ok {
				log.Info("receiveChan closed, processing remaining request")
				// drain policy maintained task
				for task.valid() {
					execChan <- task.Task
					s.removeScheduledTask(task, time.Now())
					task, execChan = s.setupExecListener(time.Now())
				}
				log.Info("all task put into exeChan, schedule worker exit")
				close(s.execChan)
				return
			}
			// Receive add operation request and return the process result.
			// And consume recv chan as much as possible.
			s.consumeRecvChan(req, maxReceiveChanBatchConsumeNum, now)
		case execChan <- execTask:
			// Keep the task in the ordered policy until the executor accepts it,
			// so an earlier Proxy request can still displace it while waiting.
			s.removeScheduledTask(task, time.Now())
			// And produce new task into execChan as much as possible.
			s.produceExecChan()
		}
	}
}

// consumeRecvChan consume the recv chan as much as possible.
func (s *scheduler) consumeRecvChan(req addTaskReq, limit int, now time.Time) {
	// Check the dynamic wait task limit.
	maxWaitTaskNum := paramtable.Get().QueryNodeCfg.MaxUnsolvedQueueSize.GetAsInt64()
	if !s.handleAddTaskRequest(req, maxWaitTaskNum, now) {
		return
	}

	// consume the add chan until reaching the batch operation limit
	for i := 1; i < limit; i++ {
		select {
		case req, ok := <-s.receiveChan:
			if !ok {
				return
			}
			if !s.handleAddTaskRequest(req, maxWaitTaskNum, now) {
				return
			}
		default:
			return
		}
	}
}

// HandleAddTaskRequest handle a add task request.
// Return true if the process can be continued.
func (s *scheduler) handleAddTaskRequest(req addTaskReq, maxWaitTaskNum int64, now time.Time) bool {
	if maxWaitTaskNum > 0 && s.GetWaitingTaskTotal() >= maxWaitTaskNum {
		s.cleanupExpiredTasks(now)
	}

	if err := req.task.Context().Err(); err != nil {
		log.Warn("task canceled before enqueue", zap.Error(err))
		req.err <- err
	} else if maxWaitTaskNum > 0 && s.GetWaitingTaskTotal() >= maxWaitTaskNum {
		err := merr.WrapErrTooManyRequests(
			int32(maxWaitTaskNum),
			fmt.Sprintf("limit by %s", paramtable.Get().QueryNodeCfg.MaxUnsolvedQueueSize.Key),
		)
		req.err <- err
	} else {
		// Push the task into the policy to schedule and update the counter of the ready queue.
		queued := newQueuedTask(req.task, now)
		nq := queued.NQ()
		newTaskAdded, err := s.policy.Push(queued)
		if err == nil {
			s.updateWaitingTaskCounter(int64(newTaskAdded), nq)
		}
		req.err <- err
	}

	// Continue processing if the queue isn't reach the max limit.
	return maxWaitTaskNum <= 0 || s.GetWaitingTaskTotal() < maxWaitTaskNum
}

// produceExecChan produces tasks from scheduler into exec chan as much as possible.
func (s *scheduler) produceExecChan() {
	for {
		task, execChan := s.setupExecListener(time.Now())
		var execTask Task
		if task.valid() {
			execTask = task.Task
		}

		select {
		case req := <-s.orderingUpdates:
			s.applyTimestampOrdering(req)
		case execChan <- execTask:
			s.removeScheduledTask(task, time.Now())
		default:
			return
		}
	}
}

// exec exec the ready task in background continuously.
func (s *scheduler) exec() {
	defer s.wg.Done()
	log.Info("start execute loop")
	for {
		t, ok := <-s.execChan
		if !ok {
			log.Info("scheduler execChan closed, worker exit")
			return
		}
		// Skip this task if task is canceled.
		if err := t.Context().Err(); err != nil {
			log.Warn("task canceled before executing", zap.Error(err))
			t.Done(err)
			continue
		}
		if err := t.PreExecute(); err != nil {
			log.Warn("failed to pre-execute task", zap.Error(err))
			t.Done(err)
			continue
		}

		s.getPool(t).Submit(func() (any, error) {
			// Submit blocks while the pool is full, so the task may expire after
			// the scheduler's first check and before a worker actually starts it.
			if err := t.Context().Err(); err != nil {
				log.Warn("task canceled after waiting for executor", zap.Error(err))
				t.Done(err)
				return nil, err
			}

			// Update concurrency metric and notify task done.
			metrics.QueryNodeReadTaskConcurrency.WithLabelValues(paramtable.GetStringNodeID()).Inc()
			collector.Counter.Inc(metricsinfo.ExecuteQueueType)

			executeDuration, err := s.executeTask(t)
			metrics.QueryNodeReadTaskExecuteDuration.WithLabelValues(
				paramtable.GetStringNodeID(),
				readTaskExecuteOutcome(err),
			).Observe(float64(executeDuration.Microseconds()) / 1000.0)

			// Update all metric after task finished.
			metrics.QueryNodeReadTaskConcurrency.WithLabelValues(paramtable.GetStringNodeID()).Dec()
			collector.Counter.Dec(metricsinfo.ExecuteQueueType)

			// Notify task done.
			t.Done(err)
			return nil, err
		})
	}
}

// executeTask applies a deadline only after a pool worker is available. The
// original task context remains unchanged for admission and queue ordering.
func (s *scheduler) executeTask(t Task) (time.Duration, error) {
	latencies := &s.queryLatencies
	if t.IsSearch() {
		latencies = &s.searchLatencies
	}
	var window, timeout time.Duration
	var ratio float64
	var ok bool
	s.deadlineMu.RLock()
	enabled, generation := s.deadlineEnabled, s.deadlineGeneration
	if enabled {
		cfg := &paramtable.Get().QueryNodeCfg
		window = cfg.SchedulerTimeWindow.GetAsDurationByParse()
		ratio = cfg.SuccessLatencyRatio.GetAsFloat()
		timeout, ok = latencies.timeout(window, ratio)
	}
	ctx := t.Context()
	executeStart := time.Now()
	if ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, executeStart.Add(timeout))
		defer cancel()
	}
	s.deadlineMu.RUnlock()
	err := t.Execute(ctx)
	executeDuration := time.Since(executeStart)
	if enabled && err == nil && ctx.Err() == nil {
		s.deadlineMu.RLock()
		// A task from before a switch transition must not refill fresh windows.
		if s.deadlineEnabled && s.deadlineGeneration == generation {
			latencies.observe(executeDuration, window, ratio)
		}
		s.deadlineMu.RUnlock()
	}
	return executeDuration, err
}

func (s *scheduler) getPool(t Task) *conc.Pool[any] {
	if t.IsGpuIndex() {
		return s.gpuPool
	}

	return s.pool
}

func readTaskExecuteOutcome(err error) string {
	if err == nil {
		return metrics.SuccessLabel
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return metrics.CancelLabel
	}
	return metrics.FailLabel
}

// setupExecListener sets up the execChan and peeks the next task to run. The
// task remains owned by the policy until execChan accepts it.
func (s *scheduler) setupExecListener(now time.Time) (*queuedTask, chan Task) {
	for {
		task := s.policy.Peek(now)
		if !task.valid() {
			return nil, nil
		}
		if err := task.Context().Err(); err != nil {
			removed := s.policy.Pop(now)
			s.updateWaitingTaskCounter(-1, -removed.NQ())
			s.recordReadTaskQueueDuration(removed, now, readTaskQueueOutcomeExpired)
			removed.Done(err)
			continue
		}
		return task, s.execChan
	}
}

func (s *scheduler) removeScheduledTask(expected *queuedTask, now time.Time) {
	expectedOrder := expected.Order()
	removed := s.policy.Pop(now)
	if !removed.valid() || removed.Order() != expectedOrder {
		panic("scheduler policy returned a different task after Peek")
	}
	s.updateWaitingTaskCounter(-1, -removed.NQ())
	s.recordReadTaskQueueDuration(removed, now, readTaskQueueOutcomeScheduled)
}

func (s *scheduler) cleanupExpiredTasks(now time.Time) {
	deadlineAdvance := paramtable.Get().QueryNodeCfg.SchedulePolicyTaskDeadlineAdvance.GetAsDurationByParse()
	cleanupTime := now.Add(deadlineAdvance)
	tasks := s.policy.Cleanup(cleanupTime)
	for _, task := range tasks {
		s.updateWaitingTaskCounter(-1, -task.NQ())
		s.recordReadTaskQueueDuration(task, now, readTaskQueueOutcomeExpired)
		task.Done(cleanupTaskError(task))
	}
}

// setupReadyLenMetric update the read task ready len metric.
func (s *scheduler) setupReadyLenMetric() {
	waitingTaskCount := s.GetWaitingTaskTotal()

	// Update the ReadyQueue counter for quota.
	collector.Counter.Set(metricsinfo.ReadyQueueType, waitingTaskCount)
	// Record the waiting task length of policy as ready task metric.
	metrics.QueryNodeReadTaskReadyLen.WithLabelValues(paramtable.GetStringNodeID()).Set(float64(waitingTaskCount))
	metrics.QueryNodeReadTaskReadyNQ.WithLabelValues(paramtable.GetStringNodeID()).Set(float64(s.GetWaitingTaskTotalNQ()))
}

func (s *scheduler) recordReadTaskQueueDuration(task *queuedTask, now time.Time, outcome string) {
	if !task.valid() {
		return
	}
	metrics.QueryNodeReadTaskQueueDuration.WithLabelValues(
		paramtable.GetStringNodeID(),
		outcome,
	).Observe(float64(task.queueDuration(now).Microseconds()) / 1000.0)
}

// scheduler counter implement, concurrent safe.
type schedulerCounter struct {
	waitingTaskTotal   atomic.Int64
	waitingTaskTotalNQ atomic.Int64
}

// GetWaitingTaskTotal get ready task counts.
func (s *schedulerCounter) GetWaitingTaskTotal() int64 {
	return s.waitingTaskTotal.Load()
}

// GetWaitingTaskTotalNQ get ready task NQ.
func (s *schedulerCounter) GetWaitingTaskTotalNQ() int64 {
	return s.waitingTaskTotalNQ.Load()
}

// updateWaitingTaskCounter update the waiting task counter for observing.
func (s *schedulerCounter) updateWaitingTaskCounter(num int64, nq int64) {
	s.waitingTaskTotal.Add(num)
	s.waitingTaskTotalNQ.Add(nq)
}
