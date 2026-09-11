package optimizers

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"

	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v2/common"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

// QueryHook is the interface for search/query parameter optimizer.
type QueryHook interface {
	Run(map[string]any) error
	Init(string) error
	InitTuningConfig(map[string]string) error
	DeleteTuningConfig(string) error
}

func OptimizeSearchParams(ctx context.Context, req *querypb.SearchRequest, queryHook QueryHook, numSegments int) (*querypb.SearchRequest, error) {
	useQueryHook := queryHook != nil && paramtable.Get().AutoIndexConfig.Enable.GetAsBool()
	if !useQueryHook {
		req.Req.IsTopkReduce = false
		req.Req.IsRecallEvaluation = false
	}

	collectionId := req.GetReq().GetCollectionID()
	log := log.Ctx(ctx).With(zap.Int64("collection", collectionId))

	serializedPlan := req.GetReq().GetSerializedExprPlan()
	// plan not found
	if serializedPlan == nil {
		if !useQueryHook {
			return req, nil
		}
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
		queryInfo := plan.GetVectorAnns().GetQueryInfo()
		if queryInfo == nil {
			return nil, merr.WrapErrParameterInvalidMsg("missing search query info")
		}
		if useQueryHook {
			// use shardNum * segments num in shard to estimate total segment number
			estSegmentNum := numSegments * int(channelNum)
			metrics.QueryNodeSearchHitSegmentNum.WithLabelValues(fmt.Sprint(paramtable.GetNodeID()), fmt.Sprint(collectionId), metrics.SearchLabel).Observe(float64(estSegmentNum))

			withFilter := (plan.GetVectorAnns().GetPredicates() != nil)
			params := map[string]any{
				common.TopKKey:         queryInfo.GetTopk(),
				common.SearchParamKey:  queryInfo.GetSearchParams(),
				common.SegmentNumKey:   estSegmentNum,
				common.WithFilterKey:   withFilter,
				common.DataTypeKey:     int32(plan.GetVectorAnns().GetVectorType()),
				common.WithOptimizeKey: paramtable.Get().AutoIndexConfig.EnableOptimize.GetAsBool() && req.GetReq().GetIsTopkReduce() && queryInfo.GetGroupByFieldId() < 0,
				common.CollectionKey:   req.GetReq().GetCollectionID(),
				common.RecallEvalKey:   req.GetReq().GetIsRecallEvaluation(),
			}
			if withFilter && channelNum > 1 {
				params[common.ChannelNumKey] = channelNum
			}
			err := queryHook.Run(params)
			if err != nil {
				log.Warn("failed to execute queryHook", zap.Error(err))
				return nil, merr.WrapErrServiceUnavailable(err.Error(), "queryHook execution failed")
			}
			finalTopk := params[common.TopKKey].(int64)
			isTopkReduce := req.GetReq().GetIsTopkReduce() && (finalTopk < queryInfo.GetTopk())
			queryInfo.Topk = finalTopk
			queryInfo.SearchParams = params[common.SearchParamKey].(string)
			req.Req.IsTopkReduce = isTopkReduce
			if isRecallEvaluation, ok := params[common.RecallEvalKey]; ok {
				req.Req.IsRecallEvaluation = isRecallEvaluation.(bool) && queryInfo.GetGroupByFieldId() < 0
			} else {
				req.Req.IsRecallEvaluation = false
			}
		}
		changed, err := applyStrictGroupSettings(queryInfo)
		if err != nil {
			return nil, err
		}
		if useQueryHook || changed {
			serializedExprPlan, err := proto.Marshal(&plan)
			if err != nil {
				log.Warn("failed to marshal optimized plan", zap.Error(err))
				return nil, merr.WrapErrParameterInvalid("marshalable search plan", "plan with marshal error", err.Error())
			}
			req.Req.SerializedExprPlan = serializedExprPlan
		}
		log.Debug("optimized search params done", zap.Any("queryInfo", queryInfo))
	default:
		log.Warn("not supported node type", zap.String("nodeType", fmt.Sprintf("%T", plan.GetNode())))
	}
	return req, nil
}

// applyStrictGroupSettings runs after the hook, including when it is disabled.
// Server settings override caller/hook values; unrelated JSON values retain
// their exact numeric/string types. The serialized plan freezes this snapshot.
func applyStrictGroupSettings(info *planpb.QueryInfo) (bool, error) {
	raw := info.GetSearchParams()
	if raw == "" {
		raw = "{}"
	}
	var params map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &params); err != nil {
		return false, merr.WrapErrParameterInvalidMsg("invalid search params: %s", err)
	}
	if params == nil {
		params = make(map[string]json.RawMessage)
	}
	_, hadEnabled := params[common.EnableSearchPathKey]
	_, hadK := params[common.EnableSearchPathKKey]
	delete(params, common.EnableSearchPathKey)
	delete(params, common.EnableSearchPathKKey)
	// Retired controls must not reach the backend, even in a mixed-version plan.
	_, hadOldThreshold := params["strict_group_acceptance_threshold"]
	_, hadOldProbe := params["strict_group_probe_candidates"]
	delete(params, "strict_group_acceptance_threshold")
	delete(params, "strict_group_probe_candidates")
	eligible := info.GetStrictGroupSize() && info.GetGroupSize() > 1 && info.GetGroupByFieldId() > 0
	if eligible {
		cfg := &paramtable.Get().QueryNodeCfg
		enabled, err := strconv.ParseBool(cfg.EnableSearchPath.GetValue())
		if err != nil {
			return false, merr.WrapErrServiceUnavailable("invalid server config: " + cfg.EnableSearchPath.Key)
		}
		minK, err := strconv.ParseInt(cfg.EnableSearchPathK.GetValue(), 10, 64)
		if err != nil || minK <= 0 {
			return false, merr.WrapErrServiceUnavailable("invalid server config: " + cfg.EnableSearchPathK.Key)
		}
		params[common.EnableSearchPathKey] = json.RawMessage(strconv.FormatBool(enabled))
		params[common.EnableSearchPathKKey] = json.RawMessage(strconv.FormatInt(minK, 10))
	}
	if !eligible && !hadEnabled && !hadK && !hadOldThreshold && !hadOldProbe {
		return false, nil
	}
	encoded, err := json.Marshal(params)
	if err != nil {
		return false, err
	}
	info.SearchParams = string(encoded)
	return true, nil
}
