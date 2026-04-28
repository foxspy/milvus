package adaptive

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStableTopKConverger_NeverConvergesBelowMinBatches(t *testing.T) {
	c := NewStableTopKConverger()
	conv, _ := c.Check([]float32{1, 2}, 5, 0)
	assert.False(t, conv)
}

func TestStableTopKConverger_ConvergesWhenStable(t *testing.T) {
	c := NewStableTopKConverger()
	// Feed same thresholds 4 times (batches 0-3).
	// With Window=2, MinBatches=2: should converge once we have 3 identical
	// entries (window+1) AND batchIdx+1 >= 2.
	c.Check([]float32{1, 2}, 5, 0) // batch 0
	c.Check([]float32{1, 2}, 3, 1) // batch 1
	conv, reason := c.Check([]float32{1, 2}, 0, 2) // batch 2: 3 stable entries
	assert.True(t, conv)
	assert.Equal(t, "stable", reason)
}

func TestStableTopKConverger_DoesNotConvergeWhenChanging(t *testing.T) {
	c := NewStableTopKConverger()
	c.Check([]float32{1, 2}, 5, 0)
	c.Check([]float32{0.9, 2}, 2, 1)
	conv, _ := c.Check([]float32{0.8, 2}, 1, 2)
	assert.False(t, conv)
}

func TestStableTopKConverger_ConvergesAfterSettling(t *testing.T) {
	c := NewStableTopKConverger()
	c.Check([]float32{5, 5}, 10, 0) // initial
	c.Check([]float32{3, 3}, 5, 1)  // changed
	c.Check([]float32{3, 3}, 2, 2)  // same
	conv, reason := c.Check([]float32{3, 3}, 0, 3) // 3 identical (window+1)
	assert.True(t, conv)
	assert.Equal(t, "stable", reason)
}

func TestStableTopKConverger_EmptyThresholds(t *testing.T) {
	c := NewStableTopKConverger()
	c.Check([]float32{}, 0, 0)
	c.Check([]float32{}, 0, 1)
	conv, _ := c.Check([]float32{}, 0, 2)
	assert.True(t, conv)  // empty slices are equal → stable
}
