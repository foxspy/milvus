package scheduler

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/util/conc"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
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
	parent, cancel := context.WithTimeout(context.WithValue(context.Background(), struct{}{}, "request value"), time.Minute)
	defer cancel()
	task := newDeadlineTestTask(parent, true, func(ctx context.Context) error {
		require.Same(t, parent, ctx, "without samples the context must be passed through")
		time.Sleep(time.Millisecond)
		return nil
	})
	duration, err := s.executeTask(task)
	require.NoError(t, err)
	require.Equal(t, uint64(1), s.searchLatencies.count)
	estimate, ok := s.searchLatencies.quantile(15*time.Second, 0.9)
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
	s.searchLatencies.observe(time.Second, 15*time.Second, 0.9)
	s.queryLatencies.observe(10*time.Second, 15*time.Second, 0.9)
	parent, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for _, isSearch := range []bool{true, false} {
		called := false
		failure := errors.New("do not change the samples")
		task := newDeadlineTestTask(parent, isSearch, func(ctx context.Context) error {
			called = true
			require.Same(t, parent, ctx)
			return failure
		})
		_, err := s.executeTask(task)
		if isSearch {
			require.ErrorIs(t, err, failure)
		} else {
			require.ErrorIs(t, err, context.DeadlineExceeded)
		}
		require.Equal(t, isSearch, called, "the same budget admits search but rejects query")
	}
	require.Equal(t, uint64(1), s.searchLatencies.count)
	require.Equal(t, uint64(1), s.queryLatencies.count)
}

func TestDynamicDeadlineRejectsBeforeExecute(t *testing.T) {
	configureDynamicDeadline(t)
	for _, isSearch := range []bool{true, false} {
		s := newScheduler(newFIFOPolicy()).(*scheduler)
		latencies := &s.queryLatencies
		if isSearch {
			latencies = &s.searchLatencies
		}
		latencies.observe(2*time.Minute, 15*time.Second, 0.9)
		s.Start()
		t.Cleanup(s.Stop)
		parent, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		called := false
		task := newDeadlineTestTask(parent, isSearch, func(context.Context) error {
			called = true
			return errors.New("rejected task entered Execute")
		})
		before := readTaskExecuteDurationCount(metrics.CancelLabel)
		require.NoError(t, s.Add(task))
		err := task.Wait()
		require.ErrorIs(t, err, context.DeadlineExceeded)
		status := merr.Status(err)
		require.Equal(t, merr.TimeoutCode, status.GetCode())
		require.False(t, status.GetRetriable(), "report an application timeout without inviting RPC retries")
		require.False(t, called)
		require.NoError(t, parent.Err(), "reject before the original deadline expires")
		require.Equal(t, before, readTaskExecuteDurationCount(metrics.CancelLabel), "admission rejection is not execution cancellation")
		require.Equal(t, uint64(1), latencies.count)
	}
}

func TestDynamicDeadlineAdmittedTaskCanExceedEstimate(t *testing.T) {
	configureDynamicDeadline(t)
	for _, isSearch := range []bool{true, false} {
		s := newScheduler(newFIFOPolicy()).(*scheduler)
		latencies := &s.queryLatencies
		if isSearch {
			latencies = &s.searchLatencies
		}
		latencies.observe(time.Millisecond, 15*time.Second, 0.9)
		s.Start()
		t.Cleanup(s.Stop)
		parent, cancel := context.WithTimeout(context.WithValue(context.Background(), struct{}{}, "request value"), time.Minute)
		defer cancel()
		task := newDeadlineTestTask(parent, isSearch, func(ctx context.Context) error {
			if ctx != parent {
				return errors.New("admitted task must keep its original context and deadline")
			}
			time.Sleep(20 * time.Millisecond)
			return ctx.Err()
		})
		require.NoError(t, s.Add(task))
		require.NoError(t, task.Wait(), "the estimate must not cap execution time")
		require.Equal(t, uint64(2), latencies.count, "successful executions beyond the estimate must enter the window")
	}
}

func TestDynamicDeadlinePreservesOriginalCancellation(t *testing.T) {
	configureDynamicDeadline(t)
	for _, cancellation := range []string{"deadline", "cancel"} {
		t.Run(cancellation, func(t *testing.T) {
			s := newDynamicDeadlineTestScheduler(t)
			s.searchLatencies.observe(time.Millisecond, 15*time.Second, 0.9)
			parent, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			defer cancel()
			task := newDeadlineTestTask(parent, true, func(ctx context.Context) error {
				require.Same(t, parent, ctx)
				if cancellation == "cancel" {
					cancel()
				}
				<-ctx.Done()
				return ctx.Err()
			})
			_, err := s.executeTask(task)
			if cancellation == "deadline" {
				require.ErrorIs(t, err, context.DeadlineExceeded)
			} else {
				require.ErrorIs(t, err, context.Canceled)
			}
			require.Equal(t, uint64(1), s.searchLatencies.count)
		})
	}
}

func TestDynamicDeadlineNoRequestDeadline(t *testing.T) {
	configureDynamicDeadline(t)
	s := newDynamicDeadlineTestScheduler(t)
	s.searchLatencies.observe(time.Second, 15*time.Second, 0.9)
	parent := context.WithValue(context.Background(), struct{}{}, "request value")
	task := newDeadlineTestTask(parent, true, func(ctx context.Context) error {
		require.Same(t, parent, ctx)
		_, hasDeadline := ctx.Deadline()
		require.False(t, hasDeadline, "sampling must never introduce a deadline")
		return nil
	})
	_, err := s.executeTask(task)
	require.NoError(t, err)
	require.Equal(t, uint64(2), s.searchLatencies.count)
}

func TestDynamicDeadlineDisabledAndExpiredWindow(t *testing.T) {
	configureDynamicDeadline(t)
	s := newDynamicDeadlineTestScheduler(t)
	parent, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	task := newDeadlineTestTask(parent, true, func(ctx context.Context) error {
		require.True(t, parent == ctx)
		return nil
	})
	params := paramtable.Get()
	for _, item := range []*paramtable.ParamItem{&params.QueryNodeCfg.SuccessLatencyRatio, &params.QueryNodeCfg.SchedulerTimeWindow} {
		s.searchLatencies.observe(2*time.Minute, 15*time.Second, 0.9)
		old := item.GetValue()
		require.NoError(t, params.Save(item.Key, "0"))
		_, err := s.executeTask(task)
		require.NoError(t, err)
		require.Zero(t, s.searchLatencies.count)
		require.NoError(t, params.Save(item.Key, old))
	}
	s.searchLatencies.observeAt(time.Now().Add(-16*time.Second), 2*time.Minute, 15*time.Second, 0.9)
	_, err := s.executeTask(task)
	require.NoError(t, err)
	require.Equal(t, uint64(1), s.searchLatencies.count, "only the new successful execution remains")
}

func TestDynamicDeadlineAdmissionAfterWaitingForWorker(t *testing.T) {
	configureDynamicDeadline(t)
	s := &scheduler{execChan: make(chan Task), pool: conc.NewPool[any](1, conc.WithPreAlloc(true))}
	s.watchDynamicDeadline()
	t.Cleanup(s.unwatchDynamicDeadline)
	s.searchLatencies.observe(800*time.Millisecond, 15*time.Second, 0.9)
	s.wg.Add(1)
	go s.exec()
	defer func() {
		close(s.execChan)
		s.wg.Wait()
		s.pool.Release()
	}()
	started := make(chan struct{})
	// Waiting for this query consumes enough of the search's original budget
	// to make it inadmissible, while leaving its context unexpired.
	first := newDeadlineTestTask(context.Background(), false, func(context.Context) error {
		close(started)
		time.Sleep(400 * time.Millisecond)
		return nil
	})
	s.execChan <- first
	<-started
	parent, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	called := false
	second := newDeadlineTestTask(parent, true, func(context.Context) error {
		called = true
		return nil
	})
	estimate, ok := s.searchLatencies.quantile(15*time.Second, 0.9)
	require.True(t, ok)
	deadline, _ := parent.Deadline()
	require.Greater(t, time.Until(deadline), estimate, "sufficient budget before waiting")
	s.execChan <- second
	require.NoError(t, first.Wait())
	require.ErrorIs(t, second.Wait(), context.DeadlineExceeded)
	require.False(t, called)
	require.NoError(t, parent.Err())
	require.Equal(t, uint64(1), s.searchLatencies.count)
}

func TestDynamicDeadlineHotSwitch(t *testing.T) {
	configureDynamicDeadline(t)
	params := paramtable.Get()
	key := params.QueryNodeCfg.EnableDynamicDeadline.Key
	require.NoError(t, params.Save(key, "false"))
	s := newDynamicDeadlineTestScheduler(t)
	parent, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	runAdmitted := func(isSearch bool) {
		task := newDeadlineTestTask(parent, isSearch, func(ctx context.Context) error {
			require.True(t, parent == ctx)
			return nil
		})
		_, err := s.executeTask(task)
		require.NoError(t, err)
	}
	// A valid P90 and window must not collect samples while the switch is off.
	runAdmitted(true)
	runAdmitted(false)
	require.Zero(t, s.searchLatencies.count)
	require.Zero(t, s.queryLatencies.count)

	require.NoError(t, params.Save(key, "true"))
	runAdmitted(true)
	runAdmitted(false)
	require.Equal(t, uint64(1), s.searchLatencies.count)
	require.Equal(t, uint64(1), s.queryLatencies.count)
	for _, isSearch := range []bool{true, false} {
		latencies := &s.queryLatencies
		if isSearch {
			latencies = &s.searchLatencies
		}
		latencies.reset()
		latencies.observe(2*time.Minute, 15*time.Second, 0.9)
		task := newDeadlineTestTask(parent, isSearch, func(context.Context) error {
			return errors.New("enabled admission should reject insufficient budget")
		})
		_, err := s.executeTask(task)
		require.ErrorIs(t, err, context.DeadlineExceeded)
	}

	// Clear immediately, including a task kind that receives no further work.
	require.NoError(t, params.Save(key, "false"))
	require.Zero(t, s.searchLatencies.count)
	require.Zero(t, s.queryLatencies.count)
	runAdmitted(true)
	runAdmitted(false)
	require.Zero(t, s.searchLatencies.count)
	require.Zero(t, s.queryLatencies.count)
	// Even a quick off/on transition without intervening requests starts fresh.
	require.NoError(t, params.Save(key, "true"))
	runAdmitted(true)
	runAdmitted(false)
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
	releaseTask := sync.OnceFunc(func() { close(release) })
	t.Cleanup(releaseTask)
	done := make(chan error, 1)
	parent, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	oldTask := newDeadlineTestTask(parent, true, func(ctx context.Context) error {
		close(started)
		<-release
		if ctx != parent {
			return errors.New("admission changed the original context")
		}
		return nil
	})
	go func() {
		_, err := s.executeTask(oldTask)
		done <- err
	}()
	<-started
	probeErr := errors.New("do not record the probe")
	probeContext, cancelProbe := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancelProbe()
	probe := newDeadlineTestTask(probeContext, true, func(ctx context.Context) error {
		require.Same(t, probeContext, ctx)
		return probeErr
	})
	_, err := s.executeTask(probe)
	require.ErrorIs(t, err, context.DeadlineExceeded, "P90 exceeds the remaining budget")
	require.NoError(t, params.Save(params.QueryNodeCfg.SuccessLatencyRatio.Key, "0.1"))
	_, err = s.executeTask(probe)
	require.ErrorIs(t, err, probeErr, "P10 now fits the same budget")
	require.NoError(t, params.Save(params.QueryNodeCfg.SuccessLatencyRatio.Key, "0.99"))
	_, err = s.executeTask(probe)
	require.ErrorIs(t, err, context.DeadlineExceeded, "P99 rejects again using retained samples")
	releaseTask()
	require.NoError(t, <-done)
	// The old execution adds a short sample; the second of 11 samples remains 1s.
	got, ok := s.searchLatencies.quantile(15*time.Second, 0.1)
	require.True(t, ok)
	requireApproximateLatency(t, time.Second, got)
	require.Equal(t, uint64(11), s.searchLatencies.count)
}
