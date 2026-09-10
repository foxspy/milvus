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
		{&params.QueryNodeCfg.EnableDynamicDeadline, "true"},
		{&params.QueryNodeCfg.SchedulerTimeWindow, "15s"},
		{&params.QueryNodeCfg.SuccessLatencyRatio, "0.9"},
	} {
		old := setting.item.GetValue()
		require.NoError(t, params.Save(setting.item.Key, setting.value))
		t.Cleanup(func() { require.NoError(t, params.Save(setting.item.Key, old)) })
	}
}

func newDynamicDeadlineTestScheduler(t *testing.T) *scheduler {
	s := &scheduler{}
	s.watchDynamicDeadline()
	t.Cleanup(s.unwatchDynamicDeadline)
	return s
}

func TestDynamicDeadlineNoSamplesAndSuccessfulSamples(t *testing.T) {
	configureDynamicDeadline(t)
	s := newDynamicDeadlineTestScheduler(t)
	parent := context.WithValue(context.Background(), struct{}{}, "request value")
	task := newDeadlineTestTask(parent, true, func(ctx context.Context) error {
		require.Same(t, parent, ctx, "without samples the context must be passed through")
		time.Sleep(time.Millisecond)
		return nil
	})
	duration, err := s.executeTask(task)
	require.NoError(t, err)
	require.Equal(t, uint64(1), s.searchLatencies.count)
	estimate, ok := s.searchLatencies.timeout(15*time.Second, 0.9)
	require.True(t, ok)
	requireApproximateLatency(t, duration, estimate)
	require.Zero(t, s.queryLatencies.count)
	// Failed and canceled executions never enter the success window.
	for _, failure := range []error{errors.New("execute failed"), context.Canceled, context.DeadlineExceeded} {
		task.execute = func(context.Context) error { return failure }
		_, err = s.executeTask(task)
		require.ErrorIs(t, err, failure)
		require.Equal(t, uint64(1), s.searchLatencies.count)
	}
}

func TestDynamicDeadlineSeparatesSearchAndQuery(t *testing.T) {
	configureDynamicDeadline(t)
	s := newDynamicDeadlineTestScheduler(t)
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
		require.False(t, deadline.After(after.Add(budget+budget/20)))
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
		require.Equal(t, uint64(1), latencies.count)
	}
}

func TestDynamicDeadlinePreservesEarlierParentDeadline(t *testing.T) {
	configureDynamicDeadline(t)
	s := newDynamicDeadlineTestScheduler(t)
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
	require.Equal(t, uint64(1), s.searchLatencies.count)
}

func TestDynamicDeadlineDisabledAndExpiredWindow(t *testing.T) {
	configureDynamicDeadline(t)
	s := newDynamicDeadlineTestScheduler(t)
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
		require.Zero(t, s.searchLatencies.count)
		require.NoError(t, params.Save(item.Key, old))
	}
	s.searchLatencies.observeAt(time.Now().Add(-16*time.Second), time.Second, 15*time.Second, 0.9)
	_, err := s.executeTask(task)
	require.NoError(t, err)
	require.Equal(t, uint64(1), s.searchLatencies.count, "only the new successful execution remains")
}

func TestDynamicDeadlineStartsAfterWaitingForWorker(t *testing.T) {
	configureDynamicDeadline(t)
	s := &scheduler{execChan: make(chan Task), pool: conc.NewPool[any](1, conc.WithPreAlloc(true))}
	s.watchDynamicDeadline()
	t.Cleanup(s.unwatchDynamicDeadline)
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

func TestDynamicDeadlineHotSwitch(t *testing.T) {
	configureDynamicDeadline(t)
	params := paramtable.Get()
	key := params.QueryNodeCfg.EnableDynamicDeadline.Key
	require.NoError(t, params.Save(key, "false"))
	s := newDynamicDeadlineTestScheduler(t)
	parent := context.Background()
	runWithoutDeadline := func(isSearch bool) {
		task := newDeadlineTestTask(parent, isSearch, func(ctx context.Context) error {
			require.True(t, parent == ctx)
			return nil
		})
		_, err := s.executeTask(task)
		require.NoError(t, err)
	}
	// A valid P90 and window must not collect samples while the switch is off.
	runWithoutDeadline(true)
	runWithoutDeadline(false)
	require.Zero(t, s.searchLatencies.count)
	require.Zero(t, s.queryLatencies.count)

	require.NoError(t, params.Save(key, "true"))
	runWithoutDeadline(true)
	runWithoutDeadline(false)
	require.Equal(t, uint64(1), s.searchLatencies.count)
	require.Equal(t, uint64(1), s.queryLatencies.count)
	for _, isSearch := range []bool{true, false} {
		failure := errors.New("do not add another sample")
		task := newDeadlineTestTask(parent, isSearch, func(ctx context.Context) error {
			_, ok := ctx.Deadline()
			require.True(t, ok, "an enabled scheduler with samples assigns a deadline")
			return failure
		})
		_, err := s.executeTask(task)
		require.ErrorIs(t, err, failure)
	}

	// Clear immediately, including a task kind that receives no further work.
	require.NoError(t, params.Save(key, "false"))
	require.Zero(t, s.searchLatencies.count)
	require.Zero(t, s.queryLatencies.count)
	// Even a quick off/on transition without intervening requests starts fresh.
	require.NoError(t, params.Save(key, "true"))
	runWithoutDeadline(true)
	runWithoutDeadline(false)
	require.Equal(t, uint64(1), s.searchLatencies.count)
	require.Equal(t, uint64(1), s.queryLatencies.count)
}

func TestDynamicDeadlineSwitchExcludesInFlightSamples(t *testing.T) {
	configureDynamicDeadline(t)
	params := paramtable.Get()
	key := params.QueryNodeCfg.EnableDynamicDeadline.Key
	for _, tc := range []struct {
		name    string
		initial string
		updates []string
	}{
		{"disabled execution finishes after enable", "false", []string{"true"}},
		{"enabled execution finishes after disable", "true", []string{"false"}},
		{"old execution finishes after re-enable", "true", []string{"false", "true"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, isSearch := range []bool{true, false} {
				require.NoError(t, params.Save(key, tc.initial))
				s := newDynamicDeadlineTestScheduler(t)
				started := make(chan struct{})
				release := make(chan struct{})
				task := newDeadlineTestTask(context.Background(), isSearch, func(context.Context) error {
					close(started)
					<-release
					return nil
				})
				done := make(chan error, 1)
				go func() {
					_, err := s.executeTask(task)
					done <- err
				}()
				<-started
				for _, value := range tc.updates {
					require.NoError(t, params.Save(key, value))
				}
				close(release)
				require.NoError(t, <-done)
				require.Zero(t, s.searchLatencies.count)
				require.Zero(t, s.queryLatencies.count)
			}
		})
	}
}

func TestDynamicDeadlineWatcherCleanup(t *testing.T) {
	configureDynamicDeadline(t)
	s := newScheduler(newFIFOPolicy()).(*scheduler)
	other := newScheduler(newFIFOPolicy()).(*scheduler)
	defer other.Stop()
	s.Stop()
	params := paramtable.Get()
	key := params.QueryNodeCfg.EnableDynamicDeadline.Key
	require.NoError(t, params.Save(key, "false"))
	require.NoError(t, params.Save(key, "true"))
	require.False(t, s.deadlineEnabled, "a stopped scheduler must unregister its watcher")
	require.True(t, other.deadlineEnabled, "stopping one scheduler must preserve other watchers")
}

func TestDynamicDeadlineCompletionUsesCurrentSettings(t *testing.T) {
	for _, setting := range []string{"window grows", "window disabled", "ratio disabled"} {
		t.Run(setting, func(t *testing.T) {
			configureDynamicDeadline(t)
			params := paramtable.Get()
			require.NoError(t, params.Save(params.QueryNodeCfg.SchedulerTimeWindow.Key, "1s"))
			s := newDynamicDeadlineTestScheduler(t)
			started := make(chan struct{})
			release := make(chan struct{})
			done := make(chan error, 1)
			task := newDeadlineTestTask(context.Background(), true, func(context.Context) error {
				close(started)
				<-release
				return nil
			})
			go func() {
				_, err := s.executeTask(task)
				done <- err
			}()
			<-started
			switch setting {
			case "window grows":
				require.NoError(t, params.Save(params.QueryNodeCfg.SchedulerTimeWindow.Key, "5s"))
				// Simulate history retained under the new window while the old
				// execution is still running. Its completion must not prune at 1s.
				s.searchLatencies.mu.Lock()
				s.searchLatencies.observeAt(time.Now().Add(-2*time.Second), time.Second, 5*time.Second, 0.9)
				s.searchLatencies.mu.Unlock()
			case "window disabled":
				require.NoError(t, params.Save(params.QueryNodeCfg.SchedulerTimeWindow.Key, "0"))
			case "ratio disabled":
				require.NoError(t, params.Save(params.QueryNodeCfg.SuccessLatencyRatio.Key, "0"))
			}
			close(release)
			require.NoError(t, <-done)
			if setting == "window grows" {
				require.Equal(t, uint64(2), s.searchLatencies.count)
			} else {
				require.Zero(t, s.searchLatencies.count, "old executions must not refill disabled sampling")
			}
		})
	}
}

func TestDynamicDeadlineRatioUpdateWithInFlightTask(t *testing.T) {
	configureDynamicDeadline(t)
	params := paramtable.Get()
	s := newDynamicDeadlineTestScheduler(t)
	for i := 1; i <= 10; i++ {
		s.searchLatencies.observe(time.Duration(i)*time.Second, 15*time.Second, 0.9)
	}
	started := make(chan struct{})
	release := make(chan struct{})
	done := make(chan error, 1)
	oldTask := newDeadlineTestTask(context.Background(), true, func(context.Context) error {
		close(started)
		<-release
		return nil
	})
	go func() {
		_, err := s.executeTask(oldTask)
		done <- err
	}()
	<-started
	require.NoError(t, params.Save(params.QueryNodeCfg.SuccessLatencyRatio.Key, "0.1"))
	probeErr := errors.New("do not record the probe")
	probe := newDeadlineTestTask(context.Background(), true, func(ctx context.Context) error {
		deadline, ok := ctx.Deadline()
		require.True(t, ok)
		require.LessOrEqual(t, time.Until(deadline), 1050*time.Millisecond)
		return probeErr
	})
	_, err := s.executeTask(probe)
	require.ErrorIs(t, err, probeErr)
	close(release)
	require.NoError(t, <-done)
	// The old execution adds a short sample; the second of 11 samples remains 1s.
	got, ok := s.searchLatencies.timeout(15*time.Second, 0.1)
	require.True(t, ok)
	requireApproximateLatency(t, time.Second, got)
	require.Equal(t, uint64(11), s.searchLatencies.count)
}
