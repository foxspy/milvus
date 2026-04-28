package adaptive

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// testReq implements Request for testing.
type testReq struct {
	collectionID         int64
	isRangeSearch        bool
	isIterator           bool
	hasGroupBy           bool
	isHybrid             bool
	segCount             int
	hasClusteringKey     bool
	hasPartStats         bool
	perCollectionOverride *bool
}

func (r testReq) CollectionID() int64          { return r.collectionID }
func (r testReq) IsRangeSearch() bool          { return r.isRangeSearch }
func (r testReq) IsIterator() bool             { return r.isIterator }
func (r testReq) HasGroupBy() bool             { return r.hasGroupBy }
func (r testReq) IsHybridSearch() bool         { return r.isHybrid }
func (r testReq) SegmentCount() int            { return r.segCount }
func (r testReq) HasClusteringKey() bool       { return r.hasClusteringKey }
func (r testReq) HasPartitionStats() bool      { return r.hasPartStats }
func (r testReq) PerCollectionOverride() *bool { return r.perCollectionOverride }

func boolPtr(v bool) *bool { return &v }

func allGoodReq() testReq {
	return testReq{
		collectionID:     1,
		segCount:         100,
		hasClusteringKey: true,
		hasPartStats:     true,
	}
}

func enabledCfg() Config {
	return Config{Enabled: true, MinSegments: DefaultMinSegments}
}

func TestShouldUseAdaptive_AllConditionsPass(t *testing.T) {
	assert.True(t, ShouldUseAdaptive(enabledCfg(), allGoodReq()))
}

func TestShouldUseAdaptive_DisabledGlobal(t *testing.T) {
	cfg := Config{Enabled: false, MinSegments: DefaultMinSegments}
	assert.False(t, ShouldUseAdaptive(cfg, allGoodReq()))
}

func TestShouldUseAdaptive_PerCollectionOverrideTrue(t *testing.T) {
	cfg := Config{Enabled: false, MinSegments: DefaultMinSegments}
	req := allGoodReq()
	req.perCollectionOverride = boolPtr(true)
	assert.True(t, ShouldUseAdaptive(cfg, req))
}

func TestShouldUseAdaptive_PerCollectionOverrideFalse(t *testing.T) {
	cfg := Config{Enabled: true, MinSegments: DefaultMinSegments}
	req := allGoodReq()
	req.perCollectionOverride = boolPtr(false)
	assert.False(t, ShouldUseAdaptive(cfg, req))
}

func TestShouldUseAdaptive_NoClusteringKey(t *testing.T) {
	req := allGoodReq()
	req.hasClusteringKey = false
	assert.False(t, ShouldUseAdaptive(enabledCfg(), req))
}

func TestShouldUseAdaptive_NoPartStats(t *testing.T) {
	req := allGoodReq()
	req.hasPartStats = false
	assert.False(t, ShouldUseAdaptive(enabledCfg(), req))
}

func TestShouldUseAdaptive_RangeSearch(t *testing.T) {
	req := allGoodReq()
	req.isRangeSearch = true
	assert.False(t, ShouldUseAdaptive(enabledCfg(), req))
}

func TestShouldUseAdaptive_Iterator(t *testing.T) {
	req := allGoodReq()
	req.isIterator = true
	assert.False(t, ShouldUseAdaptive(enabledCfg(), req))
}

func TestShouldUseAdaptive_GroupBy(t *testing.T) {
	req := allGoodReq()
	req.hasGroupBy = true
	assert.False(t, ShouldUseAdaptive(enabledCfg(), req))
}

func TestShouldUseAdaptive_HybridSearch(t *testing.T) {
	req := allGoodReq()
	req.isHybrid = true
	assert.False(t, ShouldUseAdaptive(enabledCfg(), req))
}

func TestShouldUseAdaptive_BelowMinSegments(t *testing.T) {
	req := allGoodReq()
	req.segCount = 8
	assert.False(t, ShouldUseAdaptive(enabledCfg(), req))
}
