package datacoord

import (
	"math"
	"sync"

	"github.com/milvus-io/milvus/internal/util/sessionutil"
)

type knowhereVersionManager struct {
	mu       sync.Mutex
	versions map[int64]sessionutil.KnowhereVersion
}

func newKnowhereVersionManager() *knowhereVersionManager {
	return &knowhereVersionManager{
		versions: map[int64]sessionutil.KnowhereVersion{},
	}
}

func (m *knowhereVersionManager) Startup(sessions map[string]*sessionutil.Session) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, session := range sessions {
		m.addOrUpdate(session)
	}
}

func (m *knowhereVersionManager) AddNode(session *sessionutil.Session) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.addOrUpdate(session)
}

func (m *knowhereVersionManager) RemoveNode(session *sessionutil.Session) {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.versions, session.ServerID)
}

func (m *knowhereVersionManager) Update(session *sessionutil.Session) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.addOrUpdate(session)
}

func (m *knowhereVersionManager) addOrUpdate(session *sessionutil.Session) {
	m.versions[session.ServerID] = session.KnowhereVersion
}

func (m *knowhereVersionManager) GetMinimalVersion() int {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.versions) == 0 {
		return 0
	}

	minimal := math.MaxInt
	for _, version := range m.versions {
		if version.KnowhereMinimalVersion < minimal {
			minimal = version.KnowhereMinimalVersion
		}
	}
	return minimal
}
