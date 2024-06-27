package indexparamcheck

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_cardinalChecker_StaticCheck(t *testing.T) {
	// TODO
	assert.Error(t, newCardinalChecker().StaticCheck(nil))
}
