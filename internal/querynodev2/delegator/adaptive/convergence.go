package adaptive

// Converger decides whether the executor can stop after merging a batch.
type Converger interface {
	// Check returns (converged, reason). reason is emitted to metrics.
	Check(thresholds []float32, newEntriesInBatch int, batchIdx int) (converged bool, reason string)
}

// StableTopKConverger implements a pre-validated convergence algorithm:
// top-K thresholds stable across the last `window` batches AND at least
// `minBatches` batches executed.
type StableTopKConverger struct {
	Window     int
	MinBatches int
	prevThresh [][]float32
}

// NewStableTopKConverger returns a converger with validated defaults.
func NewStableTopKConverger() *StableTopKConverger {
	return &StableTopKConverger{Window: 2, MinBatches: 2}
}

func (c *StableTopKConverger) Check(thresh []float32, newEntries, batchIdx int) (bool, string) {
	cp := make([]float32, len(thresh))
	copy(cp, thresh)
	c.prevThresh = append(c.prevThresh, cp)

	if batchIdx+1 < c.MinBatches {
		return false, ""
	}
	if len(c.prevThresh) < c.Window+1 {
		return false, ""
	}
	// Compare last window+1 entries: all equal → stable.
	tail := c.prevThresh[len(c.prevThresh)-c.Window-1:]
	for i := 1; i < len(tail); i++ {
		if !sliceEqF32(tail[i], tail[0]) {
			return false, ""
		}
	}
	return true, "stable"
}

// stableEpsilon is the relative tolerance for declaring two worst-of-top-K
// thresholds equivalent across batches. Knowhere IVF search has SIMD-level
// non-determinism on the order of 1e-6, so a strict == comparison makes
// convergence fire at random batch boundaries. A 1e-5 relative (plus 1e-8
// absolute for near-zero values) tolerance is tight enough that genuine
// top-K updates still fail the check, but immune to float noise.
const stableEpsilon = 1e-5

func sliceEqF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		diff := x - y
		if diff < 0 {
			diff = -diff
		}
		mag := x
		if y > mag {
			mag = y
		}
		if mag < 0 {
			mag = -mag
		}
		if diff > stableEpsilon*mag+1e-8 {
			return false
		}
	}
	return true
}
