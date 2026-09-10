package scheduler

import (
	"math"
	"sync"
	"time"

	"github.com/google/btree"
)

type executionLatencySample struct {
	completedAt time.Time
	duration    time.Duration
	sequence    uint64
}

func executionLatencyLess(a, b executionLatencySample) bool {
	if a.duration != b.duration {
		return a.duration < b.duration
	}
	return a.sequence < b.sequence
}

// executionLatencyWindow retains successful executions in completion order.
// The two trees partition their durations at the requested nearest-rank
// quantile, avoiding sorting the entire window for every submitted task.
// The zero value is ready to use; callers share one window per task kind.
type executionLatencyWindow struct {
	mu      sync.Mutex
	samples []executionLatencySample
	nextID  uint64
	lower   *btree.BTreeG[executionLatencySample]
	upper   *btree.BTreeG[executionLatencySample]
}

func dynamicDeadlineEnabled(window time.Duration, ratio float64) bool {
	return window > 0 && ratio > 0 && ratio <= 1
}

func (w *executionLatencyWindow) reset() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.samples = nil
	w.lower, w.upper = nil, nil
}

func (w *executionLatencyWindow) timeout(window time.Duration, ratio float64) (time.Duration, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.timeoutAt(time.Now(), window, ratio)
}

// timeoutAt and observeAt require exclusive access. Keeping clock reads inside
// the lock ensures samples remain ordered even when many workers finish at once.
func (w *executionLatencyWindow) timeoutAt(now time.Time, window time.Duration, ratio float64) (time.Duration, bool) {
	if !dynamicDeadlineEnabled(window, ratio) {
		w.samples = nil
		w.lower, w.upper = nil, nil
		return 0, false
	}
	w.expire(now.Add(-window))
	if len(w.samples) == 0 {
		return 0, false
	}
	w.rebalance(ratio)
	sample, _ := w.lower.Max()
	return sample.duration, true
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
	w.expire(now.Add(-window))
	if w.lower == nil {
		w.lower = btree.NewG(16, executionLatencyLess)
		w.upper = btree.NewG(16, executionLatencyLess)
	}
	w.nextID++
	sample := executionLatencySample{completedAt: now, duration: duration, sequence: w.nextID}
	w.samples = append(w.samples, sample)
	if boundary, ok := w.lower.Max(); ok && executionLatencyLess(sample, boundary) {
		w.lower.ReplaceOrInsert(sample)
	} else {
		w.upper.ReplaceOrInsert(sample)
	}
	w.rebalance(ratio)
}

func (w *executionLatencyWindow) expire(cutoff time.Time) {
	count := 0
	for count < len(w.samples) && !w.samples[count].completedAt.After(cutoff) {
		sample := w.samples[count]
		if _, found := w.lower.Delete(sample); !found {
			w.upper.Delete(sample)
		}
		count++
	}
	clear(w.samples[:count])
	w.samples = w.samples[count:]
	if len(w.samples) == 0 {
		w.samples = nil
	}
}

func (w *executionLatencyWindow) rebalance(ratio float64) {
	rank := int(math.Ceil(ratio * float64(len(w.samples))))
	for w.lower.Len() > rank {
		sample, _ := w.lower.DeleteMax()
		w.upper.ReplaceOrInsert(sample)
	}
	for w.lower.Len() < rank {
		sample, _ := w.upper.DeleteMin()
		w.lower.ReplaceOrInsert(sample)
	}
}
