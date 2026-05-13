package adaptive

// DynamicBatchScheduler grows the segment batch size as init, init*factor, ...
// while capping each batch by max and remaining segments.
type DynamicBatchScheduler struct {
	next   int
	max    int
	factor int
}

func NewDynamicBatchScheduler(init, max, factor int) *DynamicBatchScheduler {
	if init <= 0 {
		init = 1
	}
	if max <= 0 {
		max = int(^uint(0) >> 1)
	}
	if max < init {
		max = init
	}
	if factor < 2 {
		factor = 2
	}
	return &DynamicBatchScheduler{next: init, max: max, factor: factor}
}

func (s *DynamicBatchScheduler) NextBatchSize(consumed, total int) int {
	remaining := total - consumed
	if remaining <= 0 {
		return 0
	}
	size := s.next
	if size > s.max {
		size = s.max
	}
	if size > remaining {
		size = remaining
	}
	if s.next < s.max {
		if s.next > s.max/s.factor {
			s.next = s.max
			return size
		}
		s.next *= s.factor
		if s.next > s.max {
			s.next = s.max
		}
	}
	return size
}

// NoBetterBatchConverger stops after a configured number of consecutive
// batches that do not improve the running top-K threshold.
type NoBetterBatchConverger struct {
	Window     int
	MinBatches int
	noBetter   int
}

func NewNoBetterBatchConverger(window, minBatches int) *NoBetterBatchConverger {
	if window <= 0 {
		window = 0
	}
	if minBatches < 0 {
		minBatches = 0
	}
	return &NoBetterBatchConverger{Window: window, MinBatches: minBatches}
}

func (c *NoBetterBatchConverger) Check(before []float32, beforeFull bool, after []float32, afterFull bool, batchIdx int) (bool, string, bool) {
	if c.Window <= 0 || !beforeFull || !afterFull {
		c.noBetter = 0
		return false, "", false
	}
	improved := ThresholdsImproved(before, after)
	if improved {
		c.noBetter = 0
		return false, "", true
	}
	c.noBetter++
	if batchIdx+1 >= c.MinBatches && c.noBetter >= c.Window {
		return true, "no_better_batch", false
	}
	return false, "", false
}

func ThresholdsImproved(before []float32, after []float32) bool {
	if len(before) != len(after) {
		return len(after) > len(before)
	}
	for i := range before {
		if thresholdImproved(before[i], after[i]) {
			return true
		}
	}
	return false
}

func thresholdImproved(before, after float32) bool {
	diff := float64(after - before)
	if diff <= 0 {
		return false
	}
	mag := float64(before)
	if mag < 0 {
		mag = -mag
	}
	return diff > stableEpsilon*mag+1e-8
}
