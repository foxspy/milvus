package scheduler

import (
	"math"
	"math/rand"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestExecutionLatencyWindow(t *testing.T) {
	var w executionLatencyWindow
	now := time.Now()
	window := 15 * time.Second
	_, ok := w.timeoutAt(now, window, 0.9)
	require.False(t, ok)
	for i := 10; i > 0; i-- {
		w.observeAt(now, time.Duration(i)*time.Millisecond, window, 0.9)
	}
	for _, tc := range []struct {
		ratio float64
		want  time.Duration
	}{{0.9, 9 * time.Millisecond}, {1, 10 * time.Millisecond}, {0.01, time.Millisecond}, {0.5, 5 * time.Millisecond}} {
		got, ok := w.timeoutAt(now, window, tc.ratio)
		require.True(t, ok)
		require.Equal(t, tc.want, got)
	}
	// A hot window change expires old samples; it does not leave a cached cutoff.
	w.observeAt(now.Add(10*time.Second), 20*time.Millisecond, window, 0.9)
	got, ok := w.timeoutAt(now.Add(10*time.Second), 5*time.Second, 0.9)
	require.True(t, ok)
	require.Equal(t, 20*time.Millisecond, got)
	_, ok = w.timeoutAt(now.Add(15*time.Second), 5*time.Second, 0.9)
	require.False(t, ok)
	require.Empty(t, w.samples)
	// Start sampling again after the entire window has expired.
	w.observeAt(now.Add(16*time.Second), 3*time.Millisecond, window, 0.9)
	got, ok = w.timeoutAt(now.Add(16*time.Second), window, 0.9)
	require.True(t, ok)
	require.Equal(t, 3*time.Millisecond, got)
}

func TestExecutionLatencyWindowDisabled(t *testing.T) {
	for _, tc := range []struct {
		window time.Duration
		ratio  float64
	}{{0, 0.9}, {-time.Second, 0.9}, {time.Second, 0}, {time.Second, -1}, {time.Second, 1.1}, {time.Second, math.NaN()}, {time.Second, math.Inf(1)}} {
		var w executionLatencyWindow
		w.observe(time.Millisecond, time.Second, 0.9)
		_, ok := w.timeout(tc.window, tc.ratio)
		require.False(t, ok)
		require.Empty(t, w.samples)
		w.observe(time.Millisecond, tc.window, tc.ratio)
		require.Empty(t, w.samples)
		_, ok = w.timeout(time.Second, 0.9)
		require.False(t, ok, "re-enabling without fresh samples makes no decision")
	}
}

func TestExecutionLatencyWindowMatchesSortedSamples(t *testing.T) {
	var w executionLatencyWindow
	rng := rand.New(rand.NewSource(42))
	window := time.Second
	now := time.Now()
	var samples []executionLatencySample
	for i := 0; i < 2000; i++ {
		now = now.Add(time.Duration(rng.Intn(20)) * time.Millisecond)
		// Deliberately include duplicate durations and frequent quantile changes.
		duration := time.Duration(1+rng.Intn(20)) * time.Millisecond
		ratio := []float64{0.01, 0.5, 0.9, 1}[rng.Intn(4)]
		w.observeAt(now, duration, window, ratio)
		samples = append(samples, executionLatencySample{completedAt: now, duration: duration})
		for len(samples) > 0 && !samples[0].completedAt.After(now.Add(-window)) {
			samples = samples[1:]
		}
		values := make([]time.Duration, len(samples))
		for j, sample := range samples {
			values[j] = sample.duration
		}
		sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
		want := values[int(math.Ceil(ratio*float64(len(values))))-1]
		got, ok := w.timeoutAt(now, window, ratio)
		require.True(t, ok)
		require.Equal(t, want, got)
		require.Len(t, w.samples, len(values))
		require.Equal(t, len(values), w.lower.Len()+w.upper.Len())
	}
}

func TestExecutionLatencyWindowConcurrent(t *testing.T) {
	var w executionLatencyWindow
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				w.observe(time.Duration(j+1)*time.Millisecond, time.Minute, 0.9)
				w.timeout(time.Minute, 0.9)
			}
		}()
	}
	wg.Wait()
	got, ok := w.timeout(time.Minute, 0.9)
	require.True(t, ok)
	require.Equal(t, 180*time.Millisecond, got)
	for i := 1; i < len(w.samples); i++ {
		require.False(t, w.samples[i].completedAt.Before(w.samples[i-1].completedAt))
	}
}
