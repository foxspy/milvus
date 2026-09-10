package scheduler

import (
	"strconv"
	"testing"
	"time"
)

// A logical completion clock keeps the sample rate and window size bounded
// during saturation. Each operation reads a quantile and records a success,
// including the corresponding steady-state expiration work.
func BenchmarkExecutionLatencyWindow(b *testing.B) {
	for _, samples := range []int{15000, 150000} {
		b.Run(strconv.Itoa(samples), func(b *testing.B) {
			var w executionLatencyWindow
			const window = 15 * time.Second
			now := time.Now()
			step := window / time.Duration(samples)
			duration := func(i uint64) time.Duration {
				return time.Duration(10000+(i*2654435761)%100000) * time.Microsecond
			}
			for i := 0; i < samples; i++ {
				w.observeAt(now.Add(-time.Duration(samples-1-i)*step), duration(uint64(i)), window, 0.9)
			}
			sequence := uint64(samples)
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					w.mu.Lock()
					w.quantileAt(now, window, 0.9)
					w.mu.Unlock()
					w.mu.Lock()
					now = now.Add(step)
					w.observeAt(now, duration(sequence), window, 0.9)
					sequence++
					w.mu.Unlock()
				}
			})
		})
	}
}
