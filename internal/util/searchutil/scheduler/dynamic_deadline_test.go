package scheduler

import (
	"context"
	"testing"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v2/util/conc"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

type deadlineTestTask struct {
	*MockTask
	execute func(context.Context) error
}

func (t *deadlineTestTask) Execute(ctx context.Context) error {
	return t.execute(ctx)
}

func newDeadlineTestTask(ctx context.Context, isSearch bool, execute func(context.Context) error) *deadlineTestTask {
	return &deadlineTestTask{
		MockTask: newMockTask(mockTaskConfig{ctx: ctx, isSearch: isSearch}).(*MockTask),
		execute:  execute,
	}
}

func configureDynamicDeadline(t *testing.T) {
	paramtable.Init()
	params := paramtable.Get()
	for _, setting := range []struct {
		item  *paramtable.ParamItem
		value string
	}{
		{&params.QueryNodeCfg.SchedulerTimeWindow, "15s"},
		{&params.QueryNodeCfg.SuccessLatencyRatio, "0.9"},
	} {
		old := setting.item.GetValue()
		require.NoError(t, params.Save(setting.item.Key, setting.value))
		t.Cleanup(func() { require.NoError(t, params.Save(setting.item.Key, old)) })
	}
}

func TestDynamicDeadlineNoSamplesAndSuccessfulSamples(t *testing.T) {
	configureDynamicDeadline(t)
	s := &scheduler{}
	parent := context.WithValue(context.Background(), struct{}{}, "request value")
	task := newDeadlineTestTask(parent, true, func(ctx context.Context) error {
		require.Same(t, parent, ctx, "without samples the context must be passed through")
		time.Sleep(time.Millisecond)
		return nil
	})
	duration, err := s.executeTask(task)
	require.NoError(t, err)
	require.Len(t, s.searchLatencies.samples, 1)
	require.Equal(t, duration, s.searchLatencies.samples[0].duration)
	require.Empty(t, s.queryLatencies.samples)
	// Failed and canceled executions never enter the success window.
	for _, failure := range []error{errors.New("execute failed"), context.Canceled, context.DeadlineExceeded} {
		task.execute = func(context.Context) error { return failure }
		_, err = s.executeTask(task)
		require.ErrorIs(t, err, failure)
		require.Len(t, s.searchLatencies.samples, 1)
	}
}

func TestDynamicDeadlineSeparatesSearchAndQuery(t *testing.T) {
	configureDynamicDeadline(t)
	s := &scheduler{}
	// Different kinds use different budgets, without splitting by other factors.
	s.searchLatencies.observe(time.Second, 15*time.Second, 0.9)
	s.queryLatencies.observe(10*time.Second, 15*time.Second, 0.9)
	for _, isSearch := range []bool{true, false} {
		budget := 10 * time.Second
		if isSearch {
			budget = time.Second
		}
		var deadline time.Time
		failure := errors.New("do not change the samples")
		task := newDeadlineTestTask(context.Background(), isSearch, func(ctx context.Context) error {
			var ok bool
			deadline, ok = ctx.Deadline()
			require.True(t, ok)
			return failure
		})
		before := time.Now()
		_, err := s.executeTask(task)
		after := time.Now()
		require.ErrorIs(t, err, failure)
		require.False(t, deadline.Before(before.Add(budget)))
		require.False(t, deadline.After(after.Add(budget)))
	}
}

func TestDynamicDeadlineCancelsRunningTask(t *testing.T) {
	configureDynamicDeadline(t)
	for _, isSearch := range []bool{true, false} {
		s := newScheduler(newFIFOPolicy()).(*scheduler)
		latencies := &s.queryLatencies
		if isSearch {
			latencies = &s.searchLatencies
		}
		latencies.observe(20*time.Millisecond, 15*time.Second, 0.9)
		s.Start()
		t.Cleanup(s.Stop)
		parent := context.WithValue(context.Background(), struct{}{}, "request value")
		task := newDeadlineTestTask(parent, isSearch, func(ctx context.Context) error {
			if ctx.Value(struct{}{}) != "request value" {
				return errors.New("execution context lost request values")
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(2 * time.Second):
				return errors.New("dynamic deadline did not cancel execution")
			}
		})
		require.NoError(t, s.Add(task))
		require.ErrorIs(t, task.Wait(), context.DeadlineExceeded)
		require.NoError(t, parent.Err(), "cancellation must remain local to execution")
		require.Same(t, parent, task.Context())
		require.Len(t, latencies.samples, 1)
	}
}

func TestDynamicDeadlinePreservesEarlierParentDeadline(t *testing.T) {
	configureDynamicDeadline(t)
	s := &scheduler{}
	s.searchLatencies.observe(time.Second, 15*time.Second, 0.9)
	parent, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	want, _ := parent.Deadline()
	task := newDeadlineTestTask(parent, true, func(ctx context.Context) error {
		got, ok := ctx.Deadline()
		require.True(t, ok)
		require.Equal(t, want, got)
		<-ctx.Done()
		return ctx.Err()
	})
	_, err := s.executeTask(task)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.Len(t, s.searchLatencies.samples, 1)
}

func TestDynamicDeadlineDisabledAndExpiredWindow(t *testing.T) {
	configureDynamicDeadline(t)
	s := &scheduler{}
	parent := context.Background()
	task := newDeadlineTestTask(parent, true, func(ctx context.Context) error {
		require.True(t, parent == ctx)
		return nil
	})
	params := paramtable.Get()
	for _, item := range []*paramtable.ParamItem{&params.QueryNodeCfg.SuccessLatencyRatio, &params.QueryNodeCfg.SchedulerTimeWindow} {
		s.searchLatencies.observe(time.Second, 15*time.Second, 0.9)
		old := item.GetValue()
		require.NoError(t, params.Save(item.Key, "0"))
		_, err := s.executeTask(task)
		require.NoError(t, err)
		require.Empty(t, s.searchLatencies.samples)
		require.NoError(t, params.Save(item.Key, old))
	}
	s.searchLatencies.observeAt(time.Now().Add(-16*time.Second), time.Second, 15*time.Second, 0.9)
	_, err := s.executeTask(task)
	require.NoError(t, err)
	require.Len(t, s.searchLatencies.samples, 1, "only the new successful execution remains")
}

func TestDynamicDeadlineStartsAfterWaitingForWorker(t *testing.T) {
	configureDynamicDeadline(t)
	s := &scheduler{execChan: make(chan Task), pool: conc.NewPool[any](1, conc.WithPreAlloc(true))}
	s.searchLatencies.observe(20*time.Millisecond, 15*time.Second, 0.9)
	s.wg.Add(1)
	go s.exec()
	defer func() {
		close(s.execChan)
		s.wg.Wait()
		s.pool.Release()
	}()
	started := make(chan struct{})
	// A query without samples occupies the worker beyond the search's budget.
	first := newDeadlineTestTask(context.Background(), false, func(context.Context) error {
		close(started)
		time.Sleep(100 * time.Millisecond)
		return nil
	})
	s.execChan <- first
	<-started
	var workerStarted, deadline time.Time
	second := newDeadlineTestTask(context.Background(), true, func(ctx context.Context) error {
		workerStarted = time.Now()
		var ok bool
		deadline, ok = ctx.Deadline()
		if !ok {
			return errors.New("missing execution deadline")
		}
		return ctx.Err()
	})
	submitted := time.Now()
	s.execChan <- second
	require.NoError(t, first.Wait())
	require.NoError(t, second.Wait())
	require.Greater(t, workerStarted.Sub(submitted), 20*time.Millisecond)
	require.True(t, deadline.After(workerStarted))
}
