package adaptive

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDynamicBatchScheduler_GrowsExponentially(t *testing.T) {
	s := NewDynamicBatchScheduler(1, 16, 2)

	assert.Equal(t, 1, s.NextBatchSize(0, 100))
	assert.Equal(t, 2, s.NextBatchSize(1, 100))
	assert.Equal(t, 4, s.NextBatchSize(3, 100))
	assert.Equal(t, 8, s.NextBatchSize(7, 100))
	assert.Equal(t, 16, s.NextBatchSize(15, 100))
	assert.Equal(t, 5, s.NextBatchSize(95, 100))
}

func TestDynamicBatchScheduler_NormalizesInvalidConfig(t *testing.T) {
	s := NewDynamicBatchScheduler(0, 0, 1)

	assert.Equal(t, 1, s.NextBatchSize(0, 10))
	assert.Equal(t, 2, s.NextBatchSize(1, 10))
}

func TestNoBetterBatchConverger_StopsAfterConsecutiveNoImprovement(t *testing.T) {
	c := NewNoBetterBatchConverger(2, 2)

	converged, reason, improved := c.Check([]float32{1, 2}, true, []float32{1, 2.1}, true, 0)
	assert.False(t, converged)
	assert.True(t, improved)
	assert.Empty(t, reason)

	converged, reason, improved = c.Check([]float32{1, 2.1}, true, []float32{1, 2.1}, true, 1)
	assert.False(t, converged)
	assert.False(t, improved)
	assert.Empty(t, reason)

	converged, reason, improved = c.Check([]float32{1, 2.1}, true, []float32{1, 2.1}, true, 2)
	assert.True(t, converged)
	assert.False(t, improved)
	assert.Equal(t, "no_better_batch", reason)
}

func TestNoBetterBatchConverger_ImprovementResetsWindow(t *testing.T) {
	c := NewNoBetterBatchConverger(2, 1)

	converged, _, improved := c.Check([]float32{1}, true, []float32{1}, true, 0)
	assert.False(t, converged)
	assert.False(t, improved)

	converged, _, improved = c.Check([]float32{1}, true, []float32{1.1}, true, 1)
	assert.False(t, converged)
	assert.True(t, improved)

	converged, _, improved = c.Check([]float32{1.1}, true, []float32{1.1}, true, 2)
	assert.False(t, converged)
	assert.False(t, improved)
}
