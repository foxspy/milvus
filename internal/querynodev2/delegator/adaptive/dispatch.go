package adaptive

// Request is the minimal request shape needed by the dispatcher.
type Request interface {
	CollectionID() int64
	IsRangeSearch() bool
	IsIterator() bool
	HasGroupBy() bool
	IsHybridSearch() bool
	SegmentCount() int
	HasClusteringKey() bool
	HasPartitionStats() bool
	PerCollectionOverride() *bool // nil = unset; true/false = explicit
}

// Config holds adaptive search configuration.
type Config struct {
	Enabled     bool
	BatchSize   string // "auto" or an integer string
	MinSegments int
}

const DefaultMinSegments = 16

// ShouldUseAdaptive decides whether a query should use the adaptive path.
// Per-collection override takes precedence over global flag.
func ShouldUseAdaptive(cfg Config, req Request) bool {
	// Per-collection override wins.
	if o := req.PerCollectionOverride(); o != nil {
		if !*o {
			return false
		}
		// *o == true: proceed to remaining checks
	} else if !cfg.Enabled {
		return false
	}

	if !req.HasClusteringKey() || !req.HasPartitionStats() {
		return false
	}
	if req.IsRangeSearch() || req.IsIterator() || req.HasGroupBy() || req.IsHybridSearch() {
		return false
	}
	if req.SegmentCount() < cfg.MinSegments {
		return false
	}
	return true
}
