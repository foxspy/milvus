package indexparamcheck

import (
	"fmt"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type cardinalChecker struct {
	baseChecker
}

func (c cardinalChecker) StaticCheck(params map[string]string) error {
	return nil
}

func (c cardinalChecker) CheckTrain(params map[string]string) error {
	if err := c.StaticCheck(params); err != nil {
		return err
	}
	return c.baseChecker.CheckTrain(params)
}

func (c cardinalChecker) CheckValidDataType(dType schemapb.DataType) error {
	if !typeutil.IsVectorType(dType) {
		return fmt.Errorf("can't create hnsw in not vector type")
	}
	return nil
}

func (c cardinalChecker) SetDefaultMetricTypeIfNotExist(params map[string]string, dType schemapb.DataType) {
	if typeutil.IsDenseFloatVectorType(dType) {
		setDefaultIfNotExist(params, common.MetricTypeKey, FloatVectorDefaultMetricType)
	} else if typeutil.IsSparseFloatVectorType(dType) {
		setDefaultIfNotExist(params, common.MetricTypeKey, SparseFloatVectorDefaultMetricType)
	} else if typeutil.IsBinaryVectorType(dType) {
		setDefaultIfNotExist(params, common.MetricTypeKey, BinaryVectorDefaultMetricType)
	}
}

func newCardinalChecker() IndexChecker {
	return &cardinalChecker{}
}
