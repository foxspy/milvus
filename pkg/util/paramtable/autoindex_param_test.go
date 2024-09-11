// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package paramtable

import (
	"encoding/json"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/config"
	"github.com/milvus-io/milvus/pkg/util/metric"
)

const (
	MetricTypeKey = common.MetricTypeKey
	IndexTypeKey  = common.IndexTypeKey
)

func TestAutoIndexParams_build(t *testing.T) {
	var CParams ComponentParam
	bt := NewBaseTable(SkipRemote(true))
	CParams.Init(bt)

	t.Run("test parseBuildParams success", func(t *testing.T) {
		// Params := CParams.AutoIndexConfig
		// buildParams := make([string]interface)
		var err error
		map1 := map[string]any{
			IndexTypeKey:     "HNSW",
			"M":              48,
			"efConstruction": 500,
		}
		var jsonStrBytes []byte
		jsonStrBytes, err = json.Marshal(map1)
		assert.NoError(t, err)
		bt.Save(CParams.AutoIndexConfig.IndexParams.Key, string(jsonStrBytes))
		assert.Equal(t, "HNSW", CParams.AutoIndexConfig.IndexType.GetValue())
		assert.Equal(t, strconv.Itoa(map1["M"].(int)), CParams.AutoIndexConfig.IndexParams.GetAsJSONMap()["M"])
		assert.Equal(t, strconv.Itoa(map1["efConstruction"].(int)), CParams.AutoIndexConfig.IndexParams.GetAsJSONMap()["efConstruction"])

		map2 := map[string]interface{}{
			IndexTypeKey: "IVF_FLAT",
			"nlist":      1024,
		}
		jsonStrBytes, err = json.Marshal(map2)
		assert.NoError(t, err)
		bt.Save(CParams.AutoIndexConfig.IndexParams.Key, string(jsonStrBytes))
		assert.Equal(t, "IVF_FLAT", CParams.AutoIndexConfig.IndexType.GetValue())
		assert.Equal(t, strconv.Itoa(map2["nlist"].(int)), CParams.AutoIndexConfig.IndexParams.GetAsJSONMap()["nlist"])
	})

	t.Run("test parseSparseBuildParams success", func(t *testing.T) {
		// Params := CParams.AutoIndexConfig
		// buildParams := make([string]interface)
		var err error
		map1 := map[string]any{
			IndexTypeKey:       "SPARSE_INVERTED_INDEX",
			"drop_ratio_build": 0.1,
		}
		var jsonStrBytes []byte
		jsonStrBytes, err = json.Marshal(map1)
		assert.NoError(t, err)
		bt.Save(CParams.AutoIndexConfig.SparseIndexParams.Key, string(jsonStrBytes))
		assert.Equal(t, "SPARSE_INVERTED_INDEX", CParams.AutoIndexConfig.SparseIndexParams.GetAsJSONMap()[IndexTypeKey])
		assert.Equal(t, "0.1", CParams.AutoIndexConfig.SparseIndexParams.GetAsJSONMap()["drop_ratio_build"])

		map2 := map[string]interface{}{
			IndexTypeKey:       "SPARSE_WAND",
			"drop_ratio_build": 0.2,
		}
		jsonStrBytes, err = json.Marshal(map2)
		assert.NoError(t, err)
		bt.Save(CParams.AutoIndexConfig.SparseIndexParams.Key, string(jsonStrBytes))
		assert.Equal(t, "SPARSE_WAND", CParams.AutoIndexConfig.SparseIndexParams.GetAsJSONMap()[IndexTypeKey])
		assert.Equal(t, "0.2", CParams.AutoIndexConfig.SparseIndexParams.GetAsJSONMap()["drop_ratio_build"])
	})

	t.Run("test parseBinaryParams success", func(t *testing.T) {
		// Params := CParams.AutoIndexConfig
		// buildParams := make([string]interface)
		var err error
		map1 := map[string]any{
			IndexTypeKey: "BIN_IVF_FLAT",
			"nlist":      768,
		}
		var jsonStrBytes []byte
		jsonStrBytes, err = json.Marshal(map1)
		assert.NoError(t, err)
		bt.Save(CParams.AutoIndexConfig.BinaryIndexParams.Key, string(jsonStrBytes))
		assert.Equal(t, "BIN_IVF_FLAT", CParams.AutoIndexConfig.BinaryIndexParams.GetAsJSONMap()[IndexTypeKey])
		assert.Equal(t, strconv.Itoa(map1["nlist"].(int)), CParams.AutoIndexConfig.BinaryIndexParams.GetAsJSONMap()["nlist"])

		map2 := map[string]interface{}{
			IndexTypeKey: "BIN_FLAT",
		}
		jsonStrBytes, err = json.Marshal(map2)
		assert.NoError(t, err)
		bt.Save(CParams.AutoIndexConfig.BinaryIndexParams.Key, string(jsonStrBytes))
		assert.Equal(t, "BIN_FLAT", CParams.AutoIndexConfig.BinaryIndexParams.GetAsJSONMap()[IndexTypeKey])
	})

	t.Run("test parsePrepareParams success", func(t *testing.T) {
		var err error
		map1 := map[string]any{
			"key1": 25,
		}
		var jsonStrBytes []byte
		jsonStrBytes, err = json.Marshal(map1)
		assert.NoError(t, err)
		bt.Save(CParams.AutoIndexConfig.IndexParams.Key, string(jsonStrBytes))
		assert.Equal(t, strconv.Itoa(map1["key1"].(int)), CParams.AutoIndexConfig.IndexParams.GetAsJSONMap()["key1"])
	})
}

func Test_autoIndexConfig_panicIfNotValid(t *testing.T) {
	t.Run("not in json format", func(t *testing.T) {
		mgr := config.NewManager()
		mgr.SetConfig("autoIndex.params.build", "not in json format")
		p := &autoIndexConfig{
			IndexParams: ParamItem{
				Key: "autoIndex.params.build",
			},
		}
		p.IndexParams.Init(mgr)
		assert.Panics(t, func() {
			p.panicIfNotValidAndSetDefaultMetricTypeHelper(p.IndexParams.Key, p.IndexParams.GetAsJSONMap(), schemapb.DataType_FloatVector, mgr)
		})
	})

	t.Run("index type not found", func(t *testing.T) {
		mgr := config.NewManager()
		mgr.SetConfig("autoIndex.params.build", `{"M": 30}`)
		p := &autoIndexConfig{
			IndexParams: ParamItem{
				Key: "autoIndex.params.build",
			},
		}
		p.IndexParams.Init(mgr)
		assert.Panics(t, func() {
			p.panicIfNotValidAndSetDefaultMetricTypeHelper(p.IndexParams.Key, p.IndexParams.GetAsJSONMap(), schemapb.DataType_FloatVector, mgr)
		})
	})

	t.Run("unsupported index type", func(t *testing.T) {
		mgr := config.NewManager()
		mgr.SetConfig("autoIndex.params.build", `{"index_type": "not supported"}`)
		p := &autoIndexConfig{
			IndexParams: ParamItem{
				Key: "autoIndex.params.build",
			},
		}
		p.IndexParams.Init(mgr)
		assert.Panics(t, func() {
			p.panicIfNotValidAndSetDefaultMetricTypeHelper(p.IndexParams.Key, p.IndexParams.GetAsJSONMap(), schemapb.DataType_FloatVector, mgr)
		})
	})

	t.Run("normal case, hnsw", func(t *testing.T) {
		mgr := config.NewManager()
		mgr.SetConfig("autoIndex.params.build", `{"M": 30,"efConstruction": 360,"index_type": "HNSW"}`)
		p := &autoIndexConfig{
			IndexParams: ParamItem{
				Key: "autoIndex.params.build",
			},
		}
		p.IndexParams.Init(mgr)
		assert.NotPanics(t, func() {
			p.panicIfNotValidAndSetDefaultMetricTypeHelper(p.IndexParams.Key, p.IndexParams.GetAsJSONMap(), schemapb.DataType_FloatVector, mgr)
		})
		metricType, exist := p.IndexParams.GetAsJSONMap()[common.MetricTypeKey]
		assert.True(t, exist)
		assert.Equal(t, metric.COSINE, metricType)
	})

	t.Run("normal case, binary vector", func(t *testing.T) {
		mgr := config.NewManager()
		mgr.SetConfig("autoIndex.params.binary.build", `{"nlist": 1024, "index_type": "BIN_IVF_FLAT"}`)
		p := &autoIndexConfig{
			BinaryIndexParams: ParamItem{
				Key: "autoIndex.params.binary.build",
			},
		}
		p.BinaryIndexParams.Init(mgr)
		assert.NotPanics(t, func() {
			p.panicIfNotValidAndSetDefaultMetricTypeHelper(p.BinaryIndexParams.Key, p.BinaryIndexParams.GetAsJSONMap(), schemapb.DataType_BinaryVector, mgr)
		})
		metricType, exist := p.BinaryIndexParams.GetAsJSONMap()[common.MetricTypeKey]
		assert.True(t, exist)
		assert.Equal(t, metric.HAMMING, metricType)
	})

	t.Run("normal case, sparse vector", func(t *testing.T) {
		mgr := config.NewManager()
		mgr.SetConfig("autoIndex.params.sparse.build", `{"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}`)
		p := &autoIndexConfig{
			SparseIndexParams: ParamItem{
				Key: "autoIndex.params.sparse.build",
			},
		}
		p.SparseIndexParams.Init(mgr)
		assert.NotPanics(t, func() {
			p.panicIfNotValidAndSetDefaultMetricTypeHelper(p.SparseIndexParams.Key, p.SparseIndexParams.GetAsJSONMap(), schemapb.DataType_SparseFloatVector, mgr)
		})
		metricType, exist := p.SparseIndexParams.GetAsJSONMap()[common.MetricTypeKey]
		assert.True(t, exist)
		assert.Equal(t, metric.IP, metricType)
	})

	t.Run("normal case, ivf flat", func(t *testing.T) {
		mgr := config.NewManager()
		mgr.SetConfig("autoIndex.params.build", `{"nlist": 30, "index_type": "IVF_FLAT"}`)
		p := &autoIndexConfig{
			IndexParams: ParamItem{
				Key: "autoIndex.params.build",
			},
		}
		p.IndexParams.Init(mgr)
		assert.NotPanics(t, func() {
			p.panicIfNotValidAndSetDefaultMetricTypeHelper(p.IndexParams.Key, p.IndexParams.GetAsJSONMap(), schemapb.DataType_FloatVector, mgr)
		})
		metricType, exist := p.IndexParams.GetAsJSONMap()[common.MetricTypeKey]
		assert.True(t, exist)
		assert.Equal(t, metric.COSINE, metricType)
	})

	t.Run("normal case, ivf flat", func(t *testing.T) {
		mgr := config.NewManager()
		mgr.SetConfig("autoIndex.params.build", `{"nlist": 30, "index_type": "IVF_FLAT"}`)
		p := &autoIndexConfig{
			IndexParams: ParamItem{
				Key: "autoIndex.params.build",
			},
		}
		p.IndexParams.Init(mgr)
		assert.NotPanics(t, func() {
			p.panicIfNotValidAndSetDefaultMetricTypeHelper(p.IndexParams.Key, p.IndexParams.GetAsJSONMap(), schemapb.DataType_FloatVector, mgr)
		})
		metricType, exist := p.IndexParams.GetAsJSONMap()[common.MetricTypeKey]
		assert.True(t, exist)
		assert.Equal(t, metric.COSINE, metricType)
	})

	t.Run("normal case, diskann", func(t *testing.T) {
		mgr := config.NewManager()
		mgr.SetConfig("autoIndex.params.build", `{"index_type": "DISKANN"}`)
		p := &autoIndexConfig{
			IndexParams: ParamItem{
				Key: "autoIndex.params.build",
			},
		}
		p.IndexParams.Init(mgr)
		assert.NotPanics(t, func() {
			p.panicIfNotValidAndSetDefaultMetricTypeHelper(p.IndexParams.Key, p.IndexParams.GetAsJSONMap(), schemapb.DataType_FloatVector, mgr)
		})
		metricType, exist := p.IndexParams.GetAsJSONMap()[common.MetricTypeKey]
		assert.True(t, exist)
		assert.Equal(t, metric.COSINE, metricType)
	})

	t.Run("normal case, bin flat", func(t *testing.T) {
		mgr := config.NewManager()
		mgr.SetConfig("autoIndex.params.build", `{"index_type": "BIN_FLAT"}`)
		p := &autoIndexConfig{
			IndexParams: ParamItem{
				Key: "autoIndex.params.build",
			},
		}
		p.IndexParams.Init(mgr)
		assert.NotPanics(t, func() {
			p.panicIfNotValidAndSetDefaultMetricTypeHelper(p.IndexParams.Key, p.IndexParams.GetAsJSONMap(), schemapb.DataType_FloatVector, mgr)
		})
		metricType, exist := p.IndexParams.GetAsJSONMap()[common.MetricTypeKey]
		assert.True(t, exist)
		assert.Equal(t, metric.HAMMING, metricType)
	})

	t.Run("normal case, bin ivf flat", func(t *testing.T) {
		mgr := config.NewManager()
		mgr.SetConfig("autoIndex.params.build", `{"nlist": 30, "index_type": "BIN_IVF_FLAT"}`)
		p := &autoIndexConfig{
			IndexParams: ParamItem{
				Key: "autoIndex.params.build",
			},
		}
		p.IndexParams.Init(mgr)
		assert.NotPanics(t, func() {
			p.panicIfNotValidAndSetDefaultMetricTypeHelper(p.IndexParams.Key, p.IndexParams.GetAsJSONMap(), schemapb.DataType_FloatVector, mgr)
		})
		metricType, exist := p.IndexParams.GetAsJSONMap()[common.MetricTypeKey]
		assert.True(t, exist)
		assert.Equal(t, metric.HAMMING, metricType)
	})
}

func TestScalarAutoIndexParams_build(t *testing.T) {
	var CParams ComponentParam
	bt := NewBaseTable(SkipRemote(true))
	CParams.Init(bt)

	t.Run("parse scalar auto index param success", func(t *testing.T) {
		var err error
		map1 := map[string]any{
			"numeric": "STL_SORT",
			"varchar": "TRIE",
			"bool":    "INVERTED",
		}
		var jsonStrBytes []byte
		jsonStrBytes, err = json.Marshal(map1)
		assert.NoError(t, err)
		err = bt.Save(CParams.AutoIndexConfig.ScalarAutoIndexParams.Key, string(jsonStrBytes))
		assert.NoError(t, err)
		assert.Equal(t, "STL_SORT", CParams.AutoIndexConfig.ScalarNumericIndexType.GetValue())
		assert.Equal(t, "TRIE", CParams.AutoIndexConfig.ScalarVarcharIndexType.GetValue())
		assert.Equal(t, "INVERTED", CParams.AutoIndexConfig.ScalarBoolIndexType.GetValue())
	})
}
