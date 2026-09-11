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

#include "exec/operator/groupby/GroupMembership.h"

#include <algorithm>
#include <memory>
#include <unordered_map>

#include "index/ScalarIndex.h"
#include "segcore/SegmentChunkReader.h"
#include "segcore/SegmentGrowingImpl.h"
#include "segcore/Utils.h"

namespace milvus::exec {
namespace {

template <typename T>
using GroupKey = std::optional<T>;

bool
IsEligible(const TargetBitmap* base_filter, size_t offset) {
    return base_filter == nullptr || !(*base_filter)[offset];
}

template <typename T>
bool
BuildIndexFilter(const segcore::PinnedIndexView& indexes,
                 size_t row_count,
                 const GroupKey<T>& group,
                 const TargetBitmap* base_filter,
                 TargetBitmap& filter) {
    filter.clear();
    filter.reserve(row_count);
    size_t remaining = row_count;
    for (auto& pinned : indexes) {
        auto* index = dynamic_cast<const index::ScalarIndex<T>*>(pinned.get());
        if (index == nullptr) {
            return false;
        }
        auto* mutable_index = const_cast<index::ScalarIndex<T>*>(index);
        auto matches =
            group ? mutable_index->In(1, &*group) : mutable_index->IsNull();
        auto count = std::min(remaining, matches.size());
        matches.flip();
        filter.append(matches, 0, count);
        remaining -= count;
        if (remaining == 0) {
            break;
        }
    }
    if (filter.size() != row_count) {
        return false;
    }
    if (base_filter) {
        filter |= *base_filter;
    }
    return true;
}

template <typename T, typename Visitor>
bool
ScanRawField(milvus::OpContext* op_ctx,
             const segcore::SegmentInternalInterface& segment,
             FieldId field_id,
             size_t row_count,
             Visitor&& visitor) {
    segcore::CheckCancellation(op_ctx,
                               segment.get_segment_id(),
                               field_id.get(),
                               "strict group membership");
    if (!segment.HasFieldData(field_id)) {
        return false;
    }
    if (row_count == 0) {
        return true;
    }
    if (auto growing =
            dynamic_cast<const segcore::SegmentGrowingImpl*>(&segment)) {
        auto values = growing->get_insert_record().get_data<T>(field_id);
        auto valid = growing->get_insert_record().is_valid_data_exist(field_id)
                         ? growing->get_insert_record().get_valid_data(field_id)
                         : nullptr;
        for (size_t offset = 0; offset < row_count; ++offset) {
            if ((offset & 1023) == 0) {
                segcore::CheckCancellation(op_ctx,
                                           segment.get_segment_id(),
                                           field_id.get(),
                                           "strict group membership");
            }
            if (valid && !valid->is_valid(offset)) {
                visitor(offset, GroupKey<T>(std::nullopt));
                continue;
            }
            if constexpr (std::is_same_v<T, std::string>) {
                if (values->is_mmap()) {
                    visitor(
                        offset,
                        GroupKey<T>(std::string(values->view_element(offset))));
                    continue;
                }
            }
            visitor(offset, GroupKey<T>(static_cast<T>((*values)[offset])));
        }
        return true;
    }
    auto raw_chunk_count = segment.num_chunk_data(field_id);
    if (raw_chunk_count == 0 ||
        segment.num_rows_until_chunk(field_id, 0) != 0) {
        return false;
    }
    size_t raw_row_count = 0;
    for (int64_t chunk = 0; chunk < raw_chunk_count; ++chunk) {
        raw_row_count += segment.chunk_size(field_id, chunk);
    }
    if (raw_row_count < row_count) {
        // A partially indexed field may only retain raw data for a suffix of
        // the segment. Do not reinterpret that suffix as logical offset zero.
        return false;
    }

    int64_t chunk_id = 0;
    int64_t chunk_pos = 0;
    segcore::SegmentChunkReader reader(op_ctx, &segment, row_count);
    auto accessor =
        reader.GetMultipleChunkDataAccessor(segment.GetFieldDataType(field_id),
                                            field_id,
                                            chunk_id,
                                            chunk_pos,
                                            segcore::PinnedIndexView{});
    for (size_t offset = 0; offset < row_count; ++offset) {
        if ((offset & 1023) == 0) {
            segcore::CheckCancellation(op_ctx,
                                       segment.get_segment_id(),
                                       field_id.get(),
                                       "strict group membership");
        }
        auto value = accessor();
        if (value.has_value()) {
            visitor(offset, GroupKey<T>(segcore::get_from_variant<T>(value)));
        } else {
            visitor(offset, GroupKey<T>(std::nullopt));
        }
    }
    return true;
}

}  // namespace

template <typename T>
std::optional<GroupMembershipFilter>
PrepareGroupMembership(milvus::OpContext* op_ctx,
                       const segcore::SegmentInternalInterface& segment,
                       FieldId field_id,
                       int64_t row_count,
                       const std::vector<GroupKey<T>>& groups,
                       const TargetBitmap* base_filter) {
    if (row_count < 0 || (base_filter && base_filter->size() !=
                                             static_cast<size_t>(row_count))) {
        return std::nullopt;
    }
    auto count = static_cast<size_t>(row_count);
    if (segment.HasFieldData(field_id)) {
        std::unordered_map<GroupKey<T>, size_t> target_groups;
        for (size_t i = 0; i < groups.size(); ++i) {
            target_groups.emplace(groups[i], i);
        }
        auto rows =
            std::make_shared<std::vector<std::vector<size_t>>>(groups.size());
        auto scanned = ScanRawField<T>(
            op_ctx, segment, field_id, count, [&](size_t offset, auto group) {
                if (IsEligible(base_filter, offset)) {
                    auto it = target_groups.find(group);
                    if (it != target_groups.end()) {
                        (*rows)[it->second].push_back(offset);
                    }
                }
            });
        if (scanned) {
            return GroupMembershipFilter(
                [rows, count](size_t group, TargetBitmap& filter) {
                    filter.resize(count);
                    filter.set();
                    for (auto offset : rows->at(group)) {
                        filter[offset] = false;
                    }
                    return true;
                });
        }
    }
    auto indexes =
        std::make_shared<std::vector<PinWrapper<const index::IndexBase*>>>(
            segment.PinIndex(op_ctx, field_id));
    if (indexes->empty()) {
        return std::nullopt;
    }
    return GroupMembershipFilter([indexes, count, groups, base_filter](
                                     size_t group, TargetBitmap& filter) {
        return BuildIndexFilter<T>(
            *indexes, count, groups.at(group), base_filter, filter);
    });
}

template std::optional<GroupMembershipFilter>
PrepareGroupMembership<bool>(milvus::OpContext*,
                             const segcore::SegmentInternalInterface&,
                             FieldId,
                             int64_t,
                             const std::vector<std::optional<bool>>&,
                             const TargetBitmap*);
template std::optional<GroupMembershipFilter>
PrepareGroupMembership<int8_t>(milvus::OpContext*,
                               const segcore::SegmentInternalInterface&,
                               FieldId,
                               int64_t,
                               const std::vector<std::optional<int8_t>>&,
                               const TargetBitmap*);
template std::optional<GroupMembershipFilter>
PrepareGroupMembership<int16_t>(milvus::OpContext*,
                                const segcore::SegmentInternalInterface&,
                                FieldId,
                                int64_t,
                                const std::vector<std::optional<int16_t>>&,
                                const TargetBitmap*);
template std::optional<GroupMembershipFilter>
PrepareGroupMembership<int32_t>(milvus::OpContext*,
                                const segcore::SegmentInternalInterface&,
                                FieldId,
                                int64_t,
                                const std::vector<std::optional<int32_t>>&,
                                const TargetBitmap*);
template std::optional<GroupMembershipFilter>
PrepareGroupMembership<int64_t>(milvus::OpContext*,
                                const segcore::SegmentInternalInterface&,
                                FieldId,
                                int64_t,
                                const std::vector<std::optional<int64_t>>&,
                                const TargetBitmap*);
template std::optional<GroupMembershipFilter>
PrepareGroupMembership<std::string>(
    milvus::OpContext*,
    const segcore::SegmentInternalInterface&,
    FieldId,
    int64_t,
    const std::vector<std::optional<std::string>>&,
    const TargetBitmap*);

}  // namespace milvus::exec
