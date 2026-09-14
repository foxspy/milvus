package scheduler

import (
	"context"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/cockroachdb/errors"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

func admissionThresholdHistogram(t *testing.T, queryType string) *dto.Histogram {
	t.Helper()
	metric := &dto.Metric{}
	observer := metrics.QueryNodeReadTaskAdmissionThreshold.WithLabelValues(paramtable.GetStringNodeID(), queryType)
	require.NoError(t, observer.(interface{ Write(*dto.Metric) error }).Write(metric))
	return metric.GetHistogram()
}

func admissionThresholdCurrent(t *testing.T, queryType string) float64 {
	t.Helper()
	metric := &dto.Metric{}
	require.NoError(t, metrics.QueryNodeReadTaskAdmissionThresholdCurrent.WithLabelValues(
		paramtable.GetStringNodeID(), queryType,
	).Write(metric))
	return metric.GetGauge().GetValue()
}

func admissionRemainingDeadlineHistogram(t *testing.T, queryType string) *dto.Histogram {
	t.Helper()
	metric := &dto.Metric{}
	observer := metrics.QueryNodeReadTaskAdmissionRemainingDeadline.WithLabelValues(paramtable.GetStringNodeID(), queryType)
	require.NoError(t, observer.(interface{ Write(*dto.Metric) error }).Write(metric))
	return metric.GetHistogram()
}

func TestScaleAdmissionEstimate(t *testing.T) {
	for _, tc := range []struct {
		name     string
		estimate time.Duration
		ratio    float64
		want     time.Duration
	}{
		{"half", 10 * time.Millisecond, 0.5, 5 * time.Millisecond},
		{"double", 10 * time.Millisecond, 2, 20 * time.Millisecond},
		{"triple", 10 * time.Millisecond, 3, 30 * time.Millisecond},
		{"round up", 3 * time.Nanosecond, 0.5, 2 * time.Nanosecond},
		{"underflow", time.Nanosecond, math.SmallestNonzeroFloat64, time.Nanosecond},
		{"identity at max duration", time.Duration(math.MaxInt64), 1, time.Duration(math.MaxInt64)},
		{"duration overflow", time.Duration(math.MaxInt64 / 2), 3, time.Duration(math.MaxInt64)},
		{"float overflow", time.Second, math.MaxFloat64, time.Duration(math.MaxInt64)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, scaleAdmissionEstimate(tc.estimate, tc.ratio))
		})
	}
}

func TestDynamicDeadlineAdmissionRatioHotUpdate(t *testing.T) {
	for _, isSearch := range []bool{true, false} {
		queryType := metrics.QueryLabel
		if isSearch {
			queryType = metrics.SearchLabel
		}
		t.Run(queryType, func(t *testing.T) {
			configureDynamicDeadline(t)
			params := paramtable.Get()
			s := newDynamicDeadlineTestScheduler(t)
			latencies := &s.queryLatencies
			if isSearch {
				latencies = &s.searchLatencies
			}
			latencies.observe(2*time.Second, 15*time.Second, 0.9)
			base, ok := latencies.quantile(15*time.Second, 0.9)
			require.True(t, ok)
			before := admissionThresholdHistogram(t, queryType)
			expectedSum := before.GetSampleSum()
			probeErr := errors.New("preserve the latency samples")
			for _, tc := range []struct {
				value    string
				ratio    float64
				admitted bool
			}{{"1", 1, true}, {"0.5", 0.5, true}, {"2", 2, false}, {"3", 3, false}} {
				require.NoError(t, params.Save(params.QueryNodeCfg.AdmissionRatio.Key, tc.value))
				parent, cancel := context.WithTimeout(context.Background(), 3*time.Second)
				deadline, _ := parent.Deadline()
				budgetBefore := admissionRemainingDeadlineHistogram(t, queryType)
				called := false
				task := newDeadlineTestTask(parent, isSearch, func(ctx context.Context) error {
					called = true
					require.Same(t, parent, ctx)
					return probeErr
				})
				remainingBefore := time.Until(deadline)
				_, err := s.executeTask(task)
				remainingAfter := time.Until(deadline)
				budgetAfter := admissionRemainingDeadlineHistogram(t, queryType)
				observedMS := budgetAfter.GetSampleSum() - budgetBefore.GetSampleSum()
				require.Equal(t, budgetBefore.GetSampleCount()+1, budgetAfter.GetSampleCount())
				require.GreaterOrEqual(t, observedMS, float64(remainingAfter)/float64(time.Millisecond)-1e-6)
				require.LessOrEqual(t, observedMS, float64(remainingBefore)/float64(time.Millisecond)+1e-6)
				if tc.admitted {
					require.ErrorIs(t, err, probeErr)
				} else {
					require.ErrorIs(t, err, context.DeadlineExceeded)
					require.ErrorContains(t, err, scaleAdmissionEstimate(base, tc.ratio).String())
					remainingText := strings.Split(strings.Split(err.Error(), "remaining ")[1], ",")[0]
					comparedBudget, parseErr := time.ParseDuration(remainingText)
					require.NoError(t, parseErr)
					require.InDelta(t, float64(comparedBudget)/float64(time.Millisecond), observedMS, 1e-6, "report the exact budget used by the rejection comparison")
				}
				require.Equal(t, tc.admitted, called)
				require.NoError(t, parent.Err(), "rejection must not cancel the original context")
				cancel()
				wantMS := float64(scaleAdmissionEstimate(base, tc.ratio)) / float64(time.Millisecond)
				require.Equal(t, wantMS, admissionThresholdCurrent(t, queryType))
				expectedSum += wantMS
				require.Equal(t, uint64(1), latencies.count, "multiplier updates must retain unscaled samples")
				unchanged, ok := latencies.quantile(15*time.Second, 0.9)
				require.True(t, ok)
				require.Equal(t, base, unchanged)
			}
			after := admissionThresholdHistogram(t, queryType)
			require.Equal(t, before.GetSampleCount()+4, after.GetSampleCount(), "record both admitted and rejected checks")
			require.InDelta(t, expectedSum, after.GetSampleSum(), 1e-6)
		})
	}
}

func TestDynamicDeadlineAdmissionSamplesRemainUnscaled(t *testing.T) {
	configureDynamicDeadline(t)
	params := paramtable.Get()
	require.NoError(t, params.Save(params.QueryNodeCfg.AdmissionRatio.Key, "3"))
	s := newDynamicDeadlineTestScheduler(t)
	parent, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	task := newDeadlineTestTask(parent, true, func(ctx context.Context) error {
		require.Same(t, parent, ctx)
		time.Sleep(time.Millisecond)
		return nil
	})
	duration, err := s.executeTask(task)
	require.NoError(t, err)
	base, ok := s.searchLatencies.quantile(15*time.Second, 0.9)
	require.True(t, ok)
	requireApproximateLatency(t, duration, base)
	probeErr := errors.New("do not add a second sample")
	task.execute = func(context.Context) error { return probeErr }
	_, err = s.executeTask(task)
	require.ErrorIs(t, err, probeErr)
	require.Equal(t, float64(scaleAdmissionEstimate(base, 3))/float64(time.Millisecond), admissionThresholdCurrent(t, metrics.SearchLabel))
	require.Equal(t, uint64(1), s.searchLatencies.count)
}

func TestDynamicDeadlineAdmissionMetricLifecycle(t *testing.T) {
	configureDynamicDeadline(t)
	params := paramtable.Get()
	s := newDynamicDeadlineTestScheduler(t)
	parent, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	probeErr := errors.New("do not add a sample")
	run := func(ctx context.Context, isSearch bool) {
		_, err := s.executeTask(newDeadlineTestTask(ctx, isSearch, func(context.Context) error { return probeErr }))
		require.ErrorIs(t, err, probeErr)
	}
	before := admissionThresholdHistogram(t, metrics.SearchLabel).GetSampleCount()
	budgetBefore := admissionRemainingDeadlineHistogram(t, metrics.SearchLabel).GetSampleCount()
	run(parent, true)
	require.Zero(t, admissionThresholdCurrent(t, metrics.SearchLabel))
	require.Equal(t, before, admissionThresholdHistogram(t, metrics.SearchLabel).GetSampleCount(), "no estimate is not a zero-threshold observation")
	require.Equal(t, budgetBefore, admissionRemainingDeadlineHistogram(t, metrics.SearchLabel).GetSampleCount())
	s.searchLatencies.observe(time.Second, 15*time.Second, 0.9)
	s.queryLatencies.observe(2*time.Second, 15*time.Second, 0.9)
	run(context.Background(), true)
	require.Equal(t, before, admissionThresholdHistogram(t, metrics.SearchLabel).GetSampleCount(), "without a deadline there is no admission check")
	require.Equal(t, budgetBefore, admissionRemainingDeadlineHistogram(t, metrics.SearchLabel).GetSampleCount())
	run(parent, true)
	run(parent, false)
	require.Positive(t, admissionThresholdCurrent(t, metrics.SearchLabel))
	require.Positive(t, admissionThresholdCurrent(t, metrics.QueryLabel))
	canceled, cancelTask := context.WithCancel(parent)
	cancelTask()
	_, err := s.executeTask(newDeadlineTestTask(canceled, true, func(context.Context) error { t.Fatal("canceled task executed"); return nil }))
	require.ErrorIs(t, err, context.Canceled)
	require.Equal(t, before+1, admissionThresholdHistogram(t, metrics.SearchLabel).GetSampleCount())
	require.Equal(t, budgetBefore+1, admissionRemainingDeadlineHistogram(t, metrics.SearchLabel).GetSampleCount())
	require.NoError(t, params.Save(params.QueryNodeCfg.EnableDynamicDeadline.Key, "false"))
	require.Zero(t, admissionThresholdCurrent(t, metrics.SearchLabel), "clear immediately without further search requests")
	require.Zero(t, admissionThresholdCurrent(t, metrics.QueryLabel), "clear immediately without further query requests")
	run(parent, true)
	require.Equal(t, before+1, admissionThresholdHistogram(t, metrics.SearchLabel).GetSampleCount(), "disable preserves the cumulative histogram")
	require.Equal(t, budgetBefore+1, admissionRemainingDeadlineHistogram(t, metrics.SearchLabel).GetSampleCount())
	require.NoError(t, params.Save(params.QueryNodeCfg.EnableDynamicDeadline.Key, "true"))
	s.searchLatencies.observeAt(time.Now().Add(-16*time.Second), time.Second, 15*time.Second, 0.9)
	run(parent, true)
	require.Zero(t, admissionThresholdCurrent(t, metrics.SearchLabel), "expired window has no estimate")
	require.Equal(t, budgetBefore+1, admissionRemainingDeadlineHistogram(t, metrics.SearchLabel).GetSampleCount())
	s.searchLatencies.observe(time.Second, 15*time.Second, 0.9)
	run(parent, true)
	require.Positive(t, admissionThresholdCurrent(t, metrics.SearchLabel))
	require.NoError(t, params.Save(params.QueryNodeCfg.SchedulerTimeWindow.Key, "0"))
	run(parent, true)
	require.Zero(t, admissionThresholdCurrent(t, metrics.SearchLabel), "invalid window disables the estimate")
	require.Equal(t, budgetBefore+2, admissionRemainingDeadlineHistogram(t, metrics.SearchLabel).GetSampleCount())
}

// Model a deadline that has elapsed before the context cancellation is visible.
type elapsedAdmissionDeadlineContext struct {
	context.Context
	deadline time.Time
}

func (c elapsedAdmissionDeadlineContext) Deadline() (time.Time, bool) {
	return c.deadline, true
}

func TestDynamicDeadlineAdmissionRemainingDeadlineElapsed(t *testing.T) {
	configureDynamicDeadline(t)
	s := newDynamicDeadlineTestScheduler(t)
	s.searchLatencies.observe(time.Second, 15*time.Second, 0.9)
	before := admissionRemainingDeadlineHistogram(t, metrics.SearchLabel)
	thresholdBefore := admissionThresholdHistogram(t, metrics.SearchLabel).GetSampleCount()
	ctx := elapsedAdmissionDeadlineContext{Context: context.Background(), deadline: time.Now().Add(-time.Millisecond)}
	_, err := s.executeTask(newDeadlineTestTask(ctx, true, func(context.Context) error {
		t.Fatal("elapsed budget must not enter Execute")
		return nil
	}))
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.ErrorContains(t, err, "remaining 0s")
	after := admissionRemainingDeadlineHistogram(t, metrics.SearchLabel)
	require.Equal(t, before.GetSampleCount()+1, after.GetSampleCount())
	require.Equal(t, before.GetSampleSum(), after.GetSampleSum(), "an elapsed budget must not decrease the cumulative sum")
	require.Equal(t, thresholdBefore+1, admissionThresholdHistogram(t, metrics.SearchLabel).GetSampleCount())
}
