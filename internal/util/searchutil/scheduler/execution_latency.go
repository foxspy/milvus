package scheduler

import (
	"math"
	"sort"
	"sync"
	"time"
)

const (
	executionLatencyTimeBucket = time.Second
	// Increasing each upper bound by ceil(bound / 20) limits rounding to 5%.
	executionLatencyGrowthDivisor = 20
)

// Inclusive duration bounds limit rounding up to 5%, including nanosecond
// durations. Integer arithmetic avoids floating point errors and overflow.
var executionLatencyBounds = func() []time.Duration {
	bounds := []time.Duration{1}
	for upper := time.Duration(1); upper < time.Duration(math.MaxInt64); {
		step := (upper-1)/executionLatencyGrowthDivisor + 1
		if upper > time.Duration(math.MaxInt64)-step {
			upper = time.Duration(math.MaxInt64)
		} else {
			upper += step
		}
		bounds = append(bounds, upper)
	}
	return bounds
}()

type executionLatencyBucket struct {
	startedAt time.Time
	counts    []uint64
}

// executionLatencyWindow aggregates samples into time and duration buckets.
// The Fenwick tree stores duration counts independently of the chosen quantile.
// Histogram updates and quantile lookup are O(log(number of duration buckets)).
// Expiration subtracts entire time-bucket histograms. No individual samples are
// stored, and changing the ratio does not rebalance histogram counts.
type executionLatencyWindow struct {
	mu      sync.Mutex
	buckets []executionLatencyBucket
	tree    []uint64 // One-based Fenwick tree of the aggregate histogram.
	count   uint64
}

func dynamicDeadlineEnabled(window time.Duration, ratio float64) bool {
	return window > 0 && ratio > 0 && ratio <= 1
}

func (w *executionLatencyWindow) reset() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.resetLocked()
}

func (w *executionLatencyWindow) resetLocked() {
	w.buckets = nil
	w.tree = nil
	w.count = 0
}

func (w *executionLatencyWindow) timeout(window time.Duration, ratio float64) (time.Duration, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.timeoutAt(time.Now(), window, ratio)
}

// timeoutAt and observeAt require exclusive access. Clock reads stay inside
// the lock so completion-time buckets remain ordered under concurrency.
func (w *executionLatencyWindow) timeoutAt(now time.Time, window time.Duration, ratio float64) (time.Duration, bool) {
	if !dynamicDeadlineEnabled(window, ratio) {
		w.resetLocked()
		return 0, false
	}
	w.expire(now.Add(-window))
	if w.count == 0 {
		return 0, false
	}
	rank := uint64(math.Ceil(ratio * float64(w.count)))
	if rank > w.count {
		rank = w.count
	}
	// Find the first duration bucket whose cumulative count reaches rank.
	// Ratio changes only change this lookup; they never move histogram counts.
	index := 0
	step := 1
	for step < len(w.tree) {
		step <<= 1
	}
	for step >>= 1; step > 0; step >>= 1 {
		if next := index + step; next < len(w.tree) && w.tree[next] < rank {
			rank -= w.tree[next]
			index = next
		}
	}
	return executionLatencyBounds[index], true
}

func (w *executionLatencyWindow) observe(duration, window time.Duration, ratio float64) {
	if !dynamicDeadlineEnabled(window, ratio) || duration <= 0 {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.observeAt(time.Now(), duration, window, ratio)
}

func (w *executionLatencyWindow) observeAt(now time.Time, duration, window time.Duration, ratio float64) {
	if !dynamicDeadlineEnabled(window, ratio) || duration <= 0 {
		return
	}
	w.expire(now.Add(-window))
	if w.tree == nil {
		w.tree = make([]uint64, len(executionLatencyBounds)+1)
	}
	// Start a bucket at its first completion. Sparse traffic does not allocate
	// empty buckets. Subsecond windows use a correspondingly shorter bucket.
	width := min(executionLatencyTimeBucket, window)
	if len(w.buckets) == 0 || now.Sub(w.buckets[len(w.buckets)-1].startedAt) >= width {
		w.buckets = append(w.buckets, executionLatencyBucket{
			startedAt: now,
			counts:    make([]uint64, len(executionLatencyBounds)),
		})
	}
	index := sort.Search(len(executionLatencyBounds), func(i int) bool { return executionLatencyBounds[i] >= duration })
	w.buckets[len(w.buckets)-1].counts[index]++
	w.count++
	for i := index + 1; i < len(w.tree); i += i & -i {
		w.tree[i]++
	}
}

// expire drops whole time buckets by their start time. This excludes stale
// samples but can discard still-valid samples less than one second early.
// Work depends on the number of time/duration buckets, not the sample count.
func (w *executionLatencyWindow) expire(cutoff time.Time) {
	if len(w.buckets) == 0 {
		return
	}
	if !w.buckets[len(w.buckets)-1].startedAt.After(cutoff) {
		// Drop an entirely expired window without visiting histogram entries.
		w.resetLocked()
		return
	}
	expired := 0
	for expired < len(w.buckets) && !w.buckets[expired].startedAt.After(cutoff) {
		for index, count := range w.buckets[expired].counts {
			if count == 0 {
				continue
			}
			w.count -= count
			for i := index + 1; i < len(w.tree); i += i & -i {
				w.tree[i] -= count
			}
		}
		expired++
	}
	clear(w.buckets[:expired])
	w.buckets = w.buckets[expired:]
}
