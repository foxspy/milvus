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
#include "SearchGroupByOperator.h"

#include <chrono>

#include "common/Tracer.h"
#include "common/Consts.h"
#include "common/JsonUtils.h"
#include "exec/operator/groupby/GroupMembership.h"
#include "fmt/format.h"
#include "monitor/Monitor.h"
#include "query/Utils.h"
#include "segcore/Utils.h"

namespace milvus {
namespace exec {

namespace {

struct StrictGroupPhase2Context {
    milvus::OpContext* op_ctx;
    const segcore::SegmentInternalInterface& segment;
    FieldId group_by_field_id;
    SearchResult* search_result;
    bool eligible;
    int64_t group_size;
};

template <typename T, typename StopPredicate>
size_t
ConsumeGroupByIteratorUntil(const std::shared_ptr<VectorIterator>& iterator,
                            const std::shared_ptr<DataGetter<T>>& data_getter,
                            GroupByMap<T>& group_map,
                            GroupByResultCollector<T>& collector,
                            StopPredicate&& should_stop) {
    size_t candidates = 0;
    while (!should_stop() && iterator->HasNext()) {
        auto offset_dis_pair = iterator->Next();
        ++candidates;
        AssertInfo(
            offset_dis_pair.has_value(),
            "Wrong state! iterator cannot return valid result whereas it "
            "still tells hasNext, terminate groupBy operation");
        auto offset = offset_dis_pair->first;
        auto distance = offset_dis_pair->second;
        if (collector.IsAcceptedOffset(offset)) {
            continue;
        }
        auto group = data_getter->Get(offset);
        if (group_map.Push(group)) {
            collector.Add(offset, distance, std::move(group));
        }
    }
    return candidates;
}

template <typename T>
bool
TryStrictGroupFilteredPhase2(const std::shared_ptr<VectorIterator>& iterator,
                             const std::shared_ptr<DataGetter<T>>& data_getter,
                             GroupByMap<T>& group_map,
                             GroupByResultCollector<T>& collector,
                             const StrictGroupPhase2Context* context) {
    if (!context || !context->eligible || !context->search_result ||
        !context->search_result->CanSearchGroups() ||
        context->search_result->total_data_cnt_ < 0) {
        return false;
    }
    collector.EnableBestGroupResults(context->group_size);
    auto phase1 = ConsumeGroupByIteratorUntil(
        iterator, data_getter, group_map, collector, [&] {
            return group_map.IsGroupCapacityReached();
        });
    if (!group_map.IsGroupCapacityReached()) {
        return true;
    }

    auto* result = context->search_result;
    auto start = std::chrono::steady_clock::now();
    const auto& groups = group_map.GetGroupOrder();
    auto membership =
        PrepareGroupMembership<T>(context->op_ctx,
                                  context->segment,
                                  context->group_by_field_id,
                                  result->total_data_cnt_,
                                  groups,
                                  result->GetGroupSearchBaseFilter());
    auto membership_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    size_t searches = 0;
    size_t phase2 = 0;
    int64_t bitmap_us = 0;
    TargetBitmap filter;
    if (membership) {
        for (size_t i = 0; i < groups.size(); ++i) {
            segcore::CheckCancellation(context->op_ctx,
                                       context->segment.get_segment_id(),
                                       context->group_by_field_id.get(),
                                       "strict group search");
            auto bitmap_start = std::chrono::steady_clock::now();
            if (!(*membership)(i, filter)) {
                break;
            }
            bitmap_us += std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - bitmap_start)
                             .count();
            // Search every locked label for the full group_size, including
            // groups already filled by phase one. Phase-one rows stay eligible.
            auto searched = result->SearchGroup(filter);
            if (!searched) {
                break;
            }
            ++searches;
            const auto& batch = **searched;
            AssertInfo(
                batch.seg_offsets_.size() == batch.distances_.size(),
                "group search returned inconsistent offsets and distances");
            for (size_t j = 0; j < batch.seg_offsets_.size(); ++j) {
                const auto offset = batch.seg_offsets_[j];
                if (offset == INVALID_SEG_OFFSET) {
                    continue;
                }
                AssertInfo(offset >= 0 &&
                               static_cast<size_t>(offset) < filter.size() &&
                               !filter[offset],
                           "group search returned an excluded row {}",
                           offset);
                ++phase2;
                if (!collector.IsAcceptedOffset(offset)) {
                    group_map.Push(groups[i]);
                }
                collector.Add(offset, batch.distances_[j], groups[i]);
            }
            result->search_storage_cost_.scanned_remote_bytes +=
                batch.search_storage_cost_.scanned_remote_bytes;
            result->search_storage_cost_.scanned_total_bytes +=
                batch.search_storage_cost_.scanned_total_bytes;
        }
    }
    // Approximate Search can underfill. Resume the original iterator while
    // keeping the selected labels and deduplicating all phase-two hits.
    auto remaining = ConsumeGroupByIteratorUntil(
        iterator, data_getter, group_map, collector, [&] {
            return group_map.IsGroupResEnough();
        });
    milvus::monitor::internal_core_strict_group_phase2_phase1_candidates
        .Observe(phase1);
    milvus::monitor::internal_core_strict_group_phase2_phase2_candidates
        .Observe(phase2);
    milvus::monitor::internal_core_strict_group_phase2_batch_count.Observe(
        searches);
    milvus::monitor::
        internal_core_strict_group_phase2_original_remaining_candidates.Observe(
            remaining);
    milvus::monitor::internal_core_strict_group_phase2_membership_build_latency
        .Observe(membership_us / 1000.0);
    milvus::monitor::internal_core_strict_group_phase2_bitmap_build_latency
        .Observe(bitmap_us / 1000.0);
    tracer::AddEvent(
        fmt::format("strict_group_search: searches={}, phase1_candidates={}, "
                    "phase2_candidates={}, remaining_candidates={}",
                    searches,
                    phase1,
                    phase2,
                    remaining));
    return true;
}

}  // namespace

template <typename T>
void
GroupIteratorsByType(
    const std::vector<std::shared_ptr<VectorIterator>>& iterators,
    int64_t topK,
    int64_t group_size,
    bool strict_group_size,
    const std::shared_ptr<DataGetter<T>>& data_getter,
    std::vector<GroupByValueType>& group_by_values,
    std::vector<int64_t>& seg_offsets,
    std::vector<float>& distances,
    const knowhere::MetricType& metrics_type,
    std::vector<size_t>& topk_per_nq_prefix_sum,
    const StrictGroupPhase2Context* context = nullptr);

void
SearchGroupBy(milvus::OpContext* op_ctx,
              const std::vector<std::shared_ptr<VectorIterator>>& iterators,
              const SearchInfo& search_info,
              std::vector<GroupByValueType>& group_by_values,
              const segcore::SegmentInternalInterface& segment,
              std::vector<int64_t>& seg_offsets,
              std::vector<float>& distances,
              std::vector<size_t>& topk_per_nq_prefix_sum,
              SearchResult* search_result) {
    Defer clear_group_search([&] {
        if (search_result != nullptr) {
            search_result->ClearGroupSearch();
        }
    });
    //1. get search meta
    FieldId group_by_field_id = search_info.group_by_field_id_.value();
    auto data_type = segment.GetFieldDataType(group_by_field_id);
    int max_total_size =
        search_info.topk_ * search_info.group_size_ * iterators.size();
    seg_offsets.reserve(max_total_size);
    distances.reserve(max_total_size);
    group_by_values.reserve(max_total_size);
    topk_per_nq_prefix_sum.reserve(iterators.size() + 1);
    StrictGroupPhase2Context phase2_context{
        op_ctx,
        segment,
        group_by_field_id,
        search_result,
        query::CanUseStrictGroupSearch(search_info, iterators.size()),
        search_info.group_size_};
    switch (data_type) {
        case DataType::INT8: {
            auto dataGetter =
                GetDataGetter<int8_t>(op_ctx, segment, group_by_field_id);
            GroupIteratorsByType<int8_t>(iterators,
                                         search_info.topk_,
                                         search_info.group_size_,
                                         search_info.strict_group_size_,
                                         dataGetter,
                                         group_by_values,
                                         seg_offsets,
                                         distances,
                                         search_info.metric_type_,
                                         topk_per_nq_prefix_sum,
                                         &phase2_context);
            break;
        }
        case DataType::INT16: {
            auto dataGetter =
                GetDataGetter<int16_t>(op_ctx, segment, group_by_field_id);
            GroupIteratorsByType<int16_t>(iterators,
                                          search_info.topk_,
                                          search_info.group_size_,
                                          search_info.strict_group_size_,
                                          dataGetter,
                                          group_by_values,
                                          seg_offsets,
                                          distances,
                                          search_info.metric_type_,
                                          topk_per_nq_prefix_sum,
                                          &phase2_context);
            break;
        }
        case DataType::INT32: {
            auto dataGetter =
                GetDataGetter<int32_t>(op_ctx, segment, group_by_field_id);
            GroupIteratorsByType<int32_t>(iterators,
                                          search_info.topk_,
                                          search_info.group_size_,
                                          search_info.strict_group_size_,
                                          dataGetter,
                                          group_by_values,
                                          seg_offsets,
                                          distances,
                                          search_info.metric_type_,
                                          topk_per_nq_prefix_sum,
                                          &phase2_context);
            break;
        }
        case DataType::INT64: {
            auto dataGetter =
                GetDataGetter<int64_t>(op_ctx, segment, group_by_field_id);
            GroupIteratorsByType<int64_t>(iterators,
                                          search_info.topk_,
                                          search_info.group_size_,
                                          search_info.strict_group_size_,
                                          dataGetter,
                                          group_by_values,
                                          seg_offsets,
                                          distances,
                                          search_info.metric_type_,
                                          topk_per_nq_prefix_sum,
                                          &phase2_context);
            break;
        }
        case DataType::TIMESTAMPTZ: {
            auto dataGetter =
                GetDataGetter<int64_t>(op_ctx, segment, group_by_field_id);
            GroupIteratorsByType<int64_t>(iterators,
                                          search_info.topk_,
                                          search_info.group_size_,
                                          search_info.strict_group_size_,
                                          dataGetter,
                                          group_by_values,
                                          seg_offsets,
                                          distances,
                                          search_info.metric_type_,
                                          topk_per_nq_prefix_sum,
                                          &phase2_context);
            break;
        }
        case DataType::BOOL: {
            auto dataGetter =
                GetDataGetter<bool>(op_ctx, segment, group_by_field_id);
            GroupIteratorsByType<bool>(iterators,
                                       search_info.topk_,
                                       search_info.group_size_,
                                       search_info.strict_group_size_,
                                       dataGetter,
                                       group_by_values,
                                       seg_offsets,
                                       distances,
                                       search_info.metric_type_,
                                       topk_per_nq_prefix_sum,
                                       &phase2_context);
            break;
        }
        case DataType::VARCHAR: {
            auto dataGetter =
                GetDataGetter<std::string>(op_ctx, segment, group_by_field_id);
            GroupIteratorsByType<std::string>(iterators,
                                              search_info.topk_,
                                              search_info.group_size_,
                                              search_info.strict_group_size_,
                                              dataGetter,
                                              group_by_values,
                                              seg_offsets,
                                              distances,
                                              search_info.metric_type_,
                                              topk_per_nq_prefix_sum,
                                              &phase2_context);
            break;
        }
        case DataType::JSON: {
            AssertInfo(search_info.json_path_.has_value(),
                       "json_path is required for json field when doing "
                       "search_group_by");
            if (search_info.json_type_.has_value()) {
                switch (search_info.json_type_.value()) {
                    case DataType::BOOL: {
                        auto data_getter = GetDataGetter<bool, milvus::Json>(
                            op_ctx,
                            segment,
                            group_by_field_id,
                            search_info.json_path_,
                            search_info.json_type_,
                            search_info.strict_cast_);
                        GroupIteratorsByType<bool>(
                            iterators,
                            search_info.topk_,
                            search_info.group_size_,
                            search_info.strict_group_size_,
                            data_getter,
                            group_by_values,
                            seg_offsets,
                            distances,
                            search_info.metric_type_,
                            topk_per_nq_prefix_sum);
                        break;
                    }
                    case DataType::INT8: {
                        auto data_getter = GetDataGetter<int8_t, milvus::Json>(
                            op_ctx,
                            segment,
                            group_by_field_id,
                            search_info.json_path_,
                            search_info.json_type_,
                            search_info.strict_cast_);
                        GroupIteratorsByType<int8_t>(
                            iterators,
                            search_info.topk_,
                            search_info.group_size_,
                            search_info.strict_group_size_,
                            data_getter,
                            group_by_values,
                            seg_offsets,
                            distances,
                            search_info.metric_type_,
                            topk_per_nq_prefix_sum);
                        break;
                    }
                    case DataType::INT16: {
                        auto data_getter = GetDataGetter<int16_t, milvus::Json>(
                            op_ctx,
                            segment,
                            group_by_field_id,
                            search_info.json_path_,
                            search_info.json_type_,
                            search_info.strict_cast_);
                        GroupIteratorsByType<int16_t>(
                            iterators,
                            search_info.topk_,
                            search_info.group_size_,
                            search_info.strict_group_size_,
                            data_getter,
                            group_by_values,
                            seg_offsets,
                            distances,
                            search_info.metric_type_,
                            topk_per_nq_prefix_sum);
                        break;
                    }
                    case DataType::INT32: {
                        auto data_getter = GetDataGetter<int32_t, milvus::Json>(
                            op_ctx,
                            segment,
                            group_by_field_id,
                            search_info.json_path_,
                            search_info.json_type_,
                            search_info.strict_cast_);
                        GroupIteratorsByType<int32_t>(
                            iterators,
                            search_info.topk_,
                            search_info.group_size_,
                            search_info.strict_group_size_,
                            data_getter,
                            group_by_values,
                            seg_offsets,
                            distances,
                            search_info.metric_type_,
                            topk_per_nq_prefix_sum);
                        break;
                    }
                    case DataType::INT64: {
                        auto data_getter = GetDataGetter<int64_t, milvus::Json>(
                            op_ctx,
                            segment,
                            group_by_field_id,
                            search_info.json_path_,
                            search_info.json_type_,
                            search_info.strict_cast_);
                        GroupIteratorsByType<int64_t>(
                            iterators,
                            search_info.topk_,
                            search_info.group_size_,
                            search_info.strict_group_size_,
                            data_getter,
                            group_by_values,
                            seg_offsets,
                            distances,
                            search_info.metric_type_,
                            topk_per_nq_prefix_sum);
                        break;
                    }
                    case DataType::VARCHAR: {
                        auto data_getter =
                            GetDataGetter<std::string, milvus::Json>(
                                op_ctx,
                                segment,
                                group_by_field_id,
                                search_info.json_path_,
                                search_info.json_type_,
                                search_info.strict_cast_);
                        GroupIteratorsByType<std::string>(
                            iterators,
                            search_info.topk_,
                            search_info.group_size_,
                            search_info.strict_group_size_,
                            data_getter,
                            group_by_values,
                            seg_offsets,
                            distances,
                            search_info.metric_type_,
                            topk_per_nq_prefix_sum);
                        break;
                    }
                    default: {
                        ThrowInfo(Unsupported,
                                  fmt::format("unsupported data type {} for "
                                              "group by operator",
                                              data_type));
                    }
                }
            } else {
                auto data_getter = GetDataGetter<std::string, milvus::Json>(
                    op_ctx,
                    segment,
                    group_by_field_id,
                    search_info.json_path_,
                    search_info.json_type_,
                    search_info.strict_cast_);
                GroupIteratorsByType<std::string>(
                    iterators,
                    search_info.topk_,
                    search_info.group_size_,
                    search_info.strict_group_size_,
                    data_getter,
                    group_by_values,
                    seg_offsets,
                    distances,
                    search_info.metric_type_,
                    topk_per_nq_prefix_sum);
            }
            break;
        }
        default: {
            ThrowInfo(
                Unsupported,
                fmt::format("unsupported data type {} for group by operator",
                            data_type));
        }
    }
}

template <typename T>
void
GroupIteratorResult(const std::shared_ptr<VectorIterator>& iterator,
                    int64_t topK,
                    int64_t group_size,
                    bool strict_group_size,
                    const std::shared_ptr<DataGetter<T>>& data_getter,
                    std::vector<GroupByValueType>& group_by_values,
                    std::vector<int64_t>& offsets,
                    std::vector<float>& distances,
                    const knowhere::MetricType& metrics_type,
                    const StrictGroupPhase2Context* context);

template <typename T>
void
GroupIteratorsByType(
    const std::vector<std::shared_ptr<VectorIterator>>& iterators,
    int64_t topK,
    int64_t group_size,
    bool strict_group_size,
    const std::shared_ptr<DataGetter<T>>& data_getter,
    std::vector<GroupByValueType>& group_by_values,
    std::vector<int64_t>& seg_offsets,
    std::vector<float>& distances,
    const knowhere::MetricType& metrics_type,
    std::vector<size_t>& topk_per_nq_prefix_sum,
    const StrictGroupPhase2Context* context) {
    topk_per_nq_prefix_sum.push_back(0);
    for (auto& iterator : iterators) {
        GroupIteratorResult<T>(iterator,
                               topK,
                               group_size,
                               strict_group_size,
                               data_getter,
                               group_by_values,
                               seg_offsets,
                               distances,
                               metrics_type,
                               context);
        topk_per_nq_prefix_sum.push_back(seg_offsets.size());
    }
}

template <typename T>
void
GroupIteratorResult(const std::shared_ptr<VectorIterator>& iterator,
                    int64_t topK,
                    int64_t group_size,
                    bool strict_group_size,
                    const std::shared_ptr<DataGetter<T>>& data_getter,
                    std::vector<GroupByValueType>& group_by_values,
                    std::vector<int64_t>& offsets,
                    std::vector<float>& distances,
                    const knowhere::MetricType& metrics_type,
                    const StrictGroupPhase2Context* context) {
    GroupByMap<T> group_map(topK, group_size, strict_group_size);
    GroupByResultCollector<T> collector;

    auto handled_by_filtered_phase2 =
        strict_group_size &&
        TryStrictGroupFilteredPhase2(
            iterator, data_getter, group_map, collector, context);
    if (!handled_by_filtered_phase2) {
        // Do iteration until fill the whole map or run out of all data. It may
        // enumerate every row in a segment and block following work.
        ConsumeGroupByIteratorUntil(
            iterator, data_getter, group_map, collector, [&] {
                return group_map.IsGroupResEnough();
            });
    }

    collector.SortAndAppend(metrics_type, group_by_values, offsets, distances);
}

}  // namespace exec
}  // namespace milvus
