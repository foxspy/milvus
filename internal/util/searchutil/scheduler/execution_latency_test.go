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
	_, ok := w.quantileAt(now, window, 0.9)
	require.False(t, ok)
	for i := 10; i > 0; i-- {
		w.observeAt(now, time.Duration(i)*time.Millisecond, window, 0.9)
	}
	for _, tc := range []struct {
		ratio float64
		want  time.Duration
	}{{0.9, 9 * time.Millisecond}, {1, 10 * time.Millisecond}, {0.01, time.Millisecond}, {0.5, 5 * time.Millisecond}} {
		got, ok := w.quantileAt(now, window, tc.ratio)
		require.True(t, ok)
		requireApproximateLatency(t, tc.want, got)
	}
	// A hot window change expires old samples; it does not leave a cached cutoff.
	w.observeAt(now.Add(10*time.Second), 20*time.Millisecond, window, 0.9)
	got, ok := w.quantileAt(now.Add(10*time.Second), 5*time.Second, 0.9)
	require.True(t, ok)
	requireApproximateLatency(t, 20*time.Millisecond, got)
	_, ok = w.quantileAt(now.Add(15*time.Second), 5*time.Second, 0.9)
	require.False(t, ok)
	require.Zero(t, w.count)
	// Start sampling again after the entire window has expired.
	w.observeAt(now.Add(16*time.Second), 3*time.Millisecond, window, 0.9)
	got, ok = w.quantileAt(now.Add(16*time.Second), window, 0.9)
	require.True(t, ok)
	requireApproximateLatency(t, 3*time.Millisecond, got)
}

func TestExecutionLatencyWindowDisabled(t *testing.T) {
	for _, tc := range []struct {
		window time.Duration
		ratio  float64
	}{{0, 0.9}, {-time.Second, 0.9}, {time.Second, 0}, {time.Second, -1}, {time.Second, 1.1}, {time.Second, math.NaN()}, {time.Second, math.Inf(1)}} {
		var w executionLatencyWindow
		w.observe(time.Millisecond, time.Second, 0.9)
		_, ok := w.quantile(tc.window, tc.ratio)
		require.False(t, ok)
		require.Zero(t, w.count)
		w.observe(time.Millisecond, tc.window, tc.ratio)
		require.Zero(t, w.count)
		_, ok = w.quantile(time.Second, 0.9)
		require.False(t, ok, "re-enabling without fresh samples makes no decision")
	}
}

// Compare Fenwick lookup against independently sorted samples. A deterministic
// completion schedule gives known one-second groups for the expiration oracle.
func TestExecutionLatencyWindowMatchesSortedSamples(t *testing.T) {
	var w executionLatencyWindow
	rng := rand.New(rand.NewSource(42))
	window := 15 * time.Second
	start := time.Now()
	type sample struct {
		bucketStart time.Time
		duration    time.Duration
	}
	var samples []sample
	for i := 0; i < 2000; i++ {
		now := start.Add(time.Duration(i) * 100 * time.Millisecond)
		duration := time.Duration(1+rng.Intn(20)) * time.Millisecond
		ratio := []float64{0.01, 0.5, 0.9, 1}[rng.Intn(4)]
		w.observeAt(now, duration, window, ratio)
		samples = append(samples, sample{start.Add(time.Duration(i/10) * time.Second), duration})
		for len(samples) > 0 && !samples[0].bucketStart.After(now.Add(-window)) {
			samples = samples[1:]
		}
		values := make([]time.Duration, len(samples))
		for j, sample := range samples {
			values[j] = sample.duration
		}
		sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
		want := values[int(math.Ceil(ratio*float64(len(values))))-1]
		got, ok := w.quantileAt(now, window, ratio)
		require.True(t, ok)
		requireApproximateLatency(t, want, got)
		require.Equal(t, uint64(len(values)), w.count)
		require.LessOrEqual(t, len(w.buckets), 15)
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
				w.quantile(time.Minute, []float64{0.1, 0.9}[j%2])
			}
		}()
	}
	wg.Wait()
	got, ok := w.quantile(time.Minute, 0.9)
	require.True(t, ok)
	requireApproximateLatency(t, 180*time.Millisecond, got)
	require.Equal(t, uint64(1600), w.count)
}

func requireApproximateLatency(t *testing.T, exact, got time.Duration) {
	t.Helper()
	require.GreaterOrEqual(t, got, exact)
	// Integer comparison remains accurate even close to MaxInt64 durations.
	require.LessOrEqual(t, got-exact, exact/20)
}

func TestExecutionLatencyHistogramBounds(t *testing.T) {
	var w executionLatencyWindow
	now := time.Now()
	values := []time.Duration{1, time.Duration(math.MaxInt64)}
	for i, upper := range executionLatencyBounds {
		if i == 0 {
			continue
		}
		lower := executionLatencyBounds[i-1] + 1
		require.Greater(t, upper, executionLatencyBounds[i-1])
		requireApproximateLatency(t, lower, upper)
		values = append(values, lower, upper)
	}
	for _, duration := range values {
		w.observeAt(now, duration, time.Minute, 0.9)
	}
	sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
	snapshot := append([]uint64(nil), w.tree...)
	for _, ratio := range []float64{math.SmallestNonzeroFloat64, 0.01, 0.1, 0.5, 0.9, 0.99, 1, 0.1} {
		want := values[int(math.Ceil(ratio*float64(len(values))))-1]
		got, ok := w.quantileAt(now, time.Minute, ratio)
		require.True(t, ok)
		requireApproximateLatency(t, want, got)
		require.Equal(t, snapshot, w.tree, "changing ratio must not mutate the aggregate histogram")
	}
}

func TestExecutionLatencyTimeBucketBoundary(t *testing.T) {
	var w executionLatencyWindow
	now := time.Now()
	w.observeAt(now, time.Millisecond, 15*time.Second, 0.9)
	w.observeAt(now.Add(999*time.Millisecond), 2*time.Millisecond, 15*time.Second, 0.9)
	w.observeAt(now.Add(time.Second), 3*time.Millisecond, 15*time.Second, 0.9)
	_, ok := w.quantileAt(now.Add(15*time.Second-time.Nanosecond), 15*time.Second, 0.9)
	require.True(t, ok)
	require.Equal(t, uint64(3), w.count)
	got, ok := w.quantileAt(now.Add(15*time.Second), 15*time.Second, 0.9)
	require.True(t, ok)
	requireApproximateLatency(t, 3*time.Millisecond, got)
	require.Equal(t, uint64(1), w.count, "the first bucket expires, including its sample completed 999ms later")
	_, ok = w.quantileAt(now.Add(16*time.Second), 15*time.Second, 0.9)
	require.False(t, ok)
	require.Empty(t, w.buckets)
	require.Empty(t, w.tree)
}

func TestExecutionLatencyWindowResetAndSubsecondWindow(t *testing.T) {
	var w executionLatencyWindow
	now := time.Now()
	w.observeAt(now, time.Millisecond, 500*time.Millisecond, 0.9)
	_, ok := w.quantileAt(now.Add(499*time.Millisecond), 500*time.Millisecond, 0.9)
	require.True(t, ok)
	_, ok = w.quantileAt(now.Add(500*time.Millisecond), 500*time.Millisecond, 0.9)
	require.False(t, ok)
	w.observeAt(now.Add(time.Second), time.Millisecond, 500*time.Millisecond, 0.9)
	w.reset()
	require.Zero(t, w.count)
	require.Empty(t, w.buckets)
	require.Empty(t, w.tree)
	w.observeAt(now.Add(time.Second), time.Millisecond, 500*time.Millisecond, 0.9)
	got, ok := w.quantileAt(now.Add(time.Second), 500*time.Millisecond, 0.9)
	require.True(t, ok)
	requireApproximateLatency(t, time.Millisecond, got)
	for _, invalid := range []time.Duration{0, -time.Nanosecond} {
		w.observeAt(now.Add(time.Second), invalid, 500*time.Millisecond, 0.9)
	}
	require.Equal(t, uint64(1), w.count)
}

func TestExecutionLatencyStorageDependsOnTimeBuckets(t *testing.T) {
	var w executionLatencyWindow
	now := time.Now()
	for i := 0; i < 150000; i++ {
		w.observeAt(now.Add(time.Duration(i)*100*time.Microsecond), time.Duration(i+1), 15*time.Second, 0.9)
	}
	require.Len(t, w.buckets, 15)
	require.Equal(t, uint64(150000), w.count)
	require.Len(t, w.tree, len(executionLatencyBounds)+1)
	for _, bucket := range w.buckets {
		require.Len(t, bucket.counts, len(executionLatencyBounds))
	}
	_, ok := w.quantileAt(now.Add(time.Minute), 15*time.Second, 0.9)
	require.False(t, ok)
	require.Empty(t, w.tree)
	require.Empty(t, w.buckets)
}
