package datacoord

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/util/sessionutil"
)

func Test_knowhereVersionManager_GetMinimalVersion(t *testing.T) {
	m := newKnowhereVersionManager()

	// empty
	assert.Zero(t, m.GetMinimalVersion())

	// startup
	m.Startup(map[string]*sessionutil.Session{
		"1": {
			ServerID:        1,
			KnowhereVersion: sessionutil.KnowhereVersion{KnowhereMinimalVersion: 20},
		},
	})
	assert.Equal(t, 20, m.GetMinimalVersion())

	// add node
	m.AddNode(&sessionutil.Session{
		ServerID:        2,
		KnowhereVersion: sessionutil.KnowhereVersion{KnowhereMinimalVersion: 10},
	})
	assert.Equal(t, 10, m.GetMinimalVersion())

	// update
	m.Update(&sessionutil.Session{
		ServerID:        2,
		KnowhereVersion: sessionutil.KnowhereVersion{KnowhereMinimalVersion: 5},
	})
	assert.Equal(t, 5, m.GetMinimalVersion())

	// remove
	m.RemoveNode(&sessionutil.Session{
		ServerID:        2,
		KnowhereVersion: sessionutil.KnowhereVersion{KnowhereMinimalVersion: 5},
	})
	assert.Equal(t, 20, m.GetMinimalVersion())
}
