package optimizers

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"

	"github.com/golang/protobuf/proto"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/proto/planpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/merr"
)

// QueryHook is the interface for search/query parameter optimizer.
type QueryHook interface {
	Run(map[string]any) error
	Init(string) error
	InitTuningConfig(map[string]string) error
	DeleteTuningConfig(string) error
}

func OptimizeSearchParams(ctx context.Context, req *querypb.SearchRequest, queryHook QueryHook, numSegments int) (*querypb.SearchRequest, error) {
	log := log.Ctx(ctx).With(zap.Int64("collection", req.GetReq().GetCollectionID()))

	serializedPlan := req.GetReq().GetSerializedExprPlan()
	// plan not found
	if serializedPlan == nil {
		log.Warn("serialized plan not found")
		return req, merr.WrapErrParameterInvalid("serialized search plan", "nil")
	}

	channelNum := req.GetTotalChannelNum()
	// not set, change to conservative channel num 1
	if channelNum <= 0 {
		channelNum = 1
	}

	plan := planpb.PlanNode{}
	err := proto.Unmarshal(serializedPlan, &plan)
	if err != nil {
		log.Warn("failed to unmarshal plan", zap.Error(err))
		return nil, merr.WrapErrParameterInvalid("valid serialized search plan", "no unmarshalable one", err.Error())
	}

	switch plan.GetNode().(type) {
	case *planpb.PlanNode_VectorAnns:
		// use shardNum * segments num in shard to estimate total segment number
		estSegmentNum := numSegments * int(channelNum)
		withFilter := (plan.GetVectorAnns().GetPredicates() != nil)
		queryInfo := plan.GetVectorAnns().GetQueryInfo()

		searchParamMap := make(map[string]interface{})
		if queryInfo.GetSearchParams() != "" {
			err := json.Unmarshal([]byte(queryInfo.GetSearchParams()), &searchParamMap)
			if err != nil {
				return nil, merr.WrapErrParameterInvalid("unmarshal search plan", "plan with unmarshal error", err.Error())
			}
		}

		var level int
		levelValue, ok := searchParamMap["level"]
		if !ok { // if level is not specified, set to default 1
			level = 1
		} else {
			switch lValue := levelValue.(type) {
			case float64: // for numeric values, json unmarshal will interpret it as float64
				level = int(lValue)
			case string:
				level, err = strconv.Atoi(lValue)
			default:
				err = fmt.Errorf("wrong level in search params")
			}
		}
		if err != nil {
			level = 1
		}
		topk := queryInfo.GetTopk()
		newTopk := float64(topk) / float64(estSegmentNum)
		ef := 0
		if level == 1 {
			if newTopk < 10 {
				ef = int(newTopk*1.2 + 31)
			} else if newTopk < 90 {
				ef = int(newTopk*0.58 + 39)
			} else {
				ef = int(newTopk)
			}
		} else if level == 2 {
			if newTopk < 10 {
				ef = int(newTopk*2 + 54)
			} else if newTopk < 200 {
				ef = int(8*math.Pow(newTopk, 0.56) + 40)
			} else {
				ef = int(newTopk)
			}
		} else {
			if newTopk < 10 {
				ef = int(10*math.Pow(newTopk, 0.5) + 70)
			} else if newTopk < 300 {
				ef = int(10*math.Pow(newTopk, 0.56) + 64)
			} else {
				ef = int(newTopk)
			}
		}
		// with filter, we ensure ef is no less than k
		if withFilter && ef < int(topk) {
			ef = int(topk)
		}
		if ef < int(topk) {
			topk = int64(ef)
		}

		searchParamMap["ef"] = ef
		searchParamValue, err := json.Marshal(searchParamMap)
		if err != nil {
			log.Warn("failed to execute queryHook", zap.Error(err))
			return nil, merr.WrapErrServiceUnavailable(err.Error(), "queryHook execution failed")
		}
		queryInfo.Topk = topk
		queryInfo.SearchParams = string(searchParamValue)
		serializedExprPlan, err := proto.Marshal(&plan)
		if err != nil {
			log.Warn("failed to marshal optimized plan", zap.Error(err))
			return nil, merr.WrapErrParameterInvalid("marshalable search plan", "plan with marshal error", err.Error())
		}
		req.Req.SerializedExprPlan = serializedExprPlan
		log.Debug("optimized search params done", zap.Any("queryInfo", queryInfo))
	default:
		log.Warn("not supported node type", zap.String("nodeType", fmt.Sprintf("%T", plan.GetNode())))
	}
	return req, nil
}
