// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <unordered_map>
#include <unordered_set>

#include "common/PrometheusClient.h"
#include "exec/operator/groupby/GroupMembership.h"
#include "exec/operator/groupby/SearchGroupByOperator.h"
#include "common/Consts.h"
#include "index/ScalarIndexSort.h"
#include "index/VectorMemIndex.h"
#include "exec/operator/Utils.h"
#include "monitor/Monitor.h"
#include "segcore/ChunkedSegmentSealedImpl.h"
#include "segcore/IndexConfigGenerator.h"
#include "test_utils/DataGen.h"
#include "test_utils/cachinglayer_test_utils.h"
#include "test_utils/storage_test_utils.h"

namespace milvus::exec {

namespace {

class CountingScalarIndex : public index::ScalarIndexSort<int64_t> {
 public:
    size_t in_calls = 0;
    size_t in_values = 0;
    size_t null_calls = 0;

    const TargetBitmap
    In(size_t n, const int64_t* values) override {
        ++in_calls;
        in_values += n;
        return index::ScalarIndexSort<int64_t>::In(n, values);
    }

    const TargetBitmap
    IsNull() override {
        ++null_calls;
        return index::ScalarIndexSort<int64_t>::IsNull();
    }
};

class SequenceIterator final : public knowhere::IndexNode::iterator {
 public:
    explicit SequenceIterator(std::vector<std::pair<int64_t, float>> values)
        : values_(std::move(values)) {
    }

    std::pair<int64_t, float>
    Next() override {
        return values_.at(position_++);
    }

    bool
    HasNext() override {
        return position_ < values_.size();
    }

 private:
    std::vector<std::pair<int64_t, float>> values_;
    size_t position_{0};
};

std::shared_ptr<VectorIterator>
MakeSequenceVectorIterator(
    const std::vector<std::pair<int64_t, float>>& candidates,
    const BitsetView& invalid = {},
    bool empty_leading_chunk = false) {
    std::vector<std::pair<int64_t, float>> eligible;
    eligible.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (invalid.empty() || !invalid.test(candidate.first)) {
            eligible.emplace_back(candidate);
        }
    }
    auto iterator = std::make_shared<VectorIterator>(
        /*chunk_count=*/empty_leading_chunk ? 2 : 1,
        /*offset_mapping=*/nullptr);
    if (empty_leading_chunk) {
        iterator->AddIterator(std::make_shared<SequenceIterator>(
            std::vector<std::pair<int64_t, float>>{}));
    }
    iterator->AddIterator(std::make_shared<SequenceIterator>(eligible));
    iterator->seal();
    return iterator;
}

// Combine only in tests to compare the union of independently built labels
// with a row-wise oracle. Production never builds a union filter.
template <typename T>
std::optional<TargetBitmap>
BuildGroupMembership(milvus::OpContext* ctx,
                     const segcore::SegmentInternalInterface& segment,
                     FieldId field,
                     int64_t count,
                     const std::vector<std::optional<T>>& groups,
                     const TargetBitmap* base) {
    auto prepared =
        PrepareGroupMembership<T>(ctx, segment, field, count, groups, base);
    if (!prepared)
        return std::nullopt;
    TargetBitmap matches(count, false), filter;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (!(*prepared)(i, filter))
            return std::nullopt;
        filter.flip();
        matches |= filter;
    }
    return matches;
}

}  // namespace

TEST(VectorIteratorFilteredChunksTest, EmptyLeadingChunkKeepsSuccessors) {
    VectorIterator iterator(2, nullptr);
    iterator.AddIterator(std::make_shared<SequenceIterator>(
        std::vector<std::pair<int64_t, float>>{}));
    iterator.AddIterator(std::make_shared<SequenceIterator>(
        std::vector<std::pair<int64_t, float>>{{3, 1.0F}, {4, 2.0F}}));
    iterator.seal();
    ASSERT_TRUE(iterator.HasNext());
    EXPECT_EQ(iterator.Next()->first, 3);
    ASSERT_TRUE(iterator.HasNext());
    EXPECT_EQ(iterator.Next()->first, 4);
    EXPECT_FALSE(iterator.HasNext());
}

TEST(StrictGroupFilteredIteratorEligibilityTest,
     RequiresStrictMultiResultSingleQueryRowLevelSearch) {
    SearchInfo eligible;
    eligible.topk_ = 10;
    eligible.enable_search_path_ = true;
    eligible.group_by_field_id_ = FieldId(101);
    eligible.group_size_ = 3;
    eligible.strict_group_size_ = true;
    EXPECT_TRUE(query::CanUseStrictGroupSearch(eligible, 1));

    eligible.enable_search_path_k_ = 11;
    EXPECT_FALSE(query::CanUseStrictGroupSearch(eligible, 1));
    eligible.enable_search_path_k_ = 10;
    EXPECT_TRUE(query::CanUseStrictGroupSearch(eligible, 1));

    auto disabled = eligible;
    disabled.enable_search_path_ = false;
    EXPECT_FALSE(query::CanUseStrictGroupSearch(disabled, 1));

    auto non_strict = eligible;
    non_strict.strict_group_size_ = false;
    EXPECT_FALSE(query::CanUseStrictGroupSearch(non_strict, 1));

    auto single_result_group = eligible;
    single_result_group.group_size_ = 1;
    EXPECT_FALSE(query::CanUseStrictGroupSearch(single_result_group, 1));

    auto empty_topk = eligible;
    empty_topk.topk_ = 0;
    EXPECT_FALSE(query::CanUseStrictGroupSearch(empty_topk, 1));
    EXPECT_FALSE(query::CanUseStrictGroupSearch(eligible, 2));

    auto element_level = eligible;
    element_level.array_offsets_ = std::make_shared<ArrayOffsetsSealed>();
    EXPECT_FALSE(query::CanUseStrictGroupSearch(element_level, 1));
}

TEST(GroupMembershipTest, RawScansHonorCancellation) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    auto field = schema->AddDebugField("group", DataType::INT64);
    schema->set_primary_field_id(pk);
    constexpr size_t rows = 4096;
    auto data = segcore::DataGen(schema, rows);
    auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
    auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
    auto offset = growing->PreInsert(rows);
    growing->Insert(
        offset, rows, data.row_ids_.data(), data.timestamps_.data(), data.raw_);
    folly::CancellationSource source;
    milvus::OpContext ctx(source.getToken());
    source.requestCancellation();
    for (const auto* segment :
         {dynamic_cast<const segcore::SegmentInternalInterface*>(sealed.get()),
          dynamic_cast<const segcore::SegmentInternalInterface*>(
              growing.get())}) {
        ASSERT_NE(segment, nullptr);
        try {
            BuildGroupMembership<int64_t>(
                &ctx, *segment, field, rows, {int64_t(1)}, nullptr);
            FAIL() << "cancelled membership scan completed";
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), ErrorCode::FollyCancel);
        }
    }
}

TEST(GroupMembershipTest, CancellationDuringRawScanStopsBeforeNextChunk) {
    // Cancel when the accessor reaches chunk 1, after chunk 0 was consumed.
    // No timing or thread scheduling dependency: the next periodic check must
    // throw before chunk 2 is pinned.
    class CancellingSegment : public segcore::ChunkedSegmentSealedImpl {
     public:
        CancellingSegment(SchemaPtr schema, folly::CancellationSource& source)
            : ChunkedSegmentSealedImpl(schema,
                                       empty_index_meta,
                                       segcore::SegcoreConfig::default_config(),
                                       991),
              source_(source),
              values_(2048, 1) {
        }
        bool
        HasFieldData(FieldId) const override {
            return true;
        }
        int64_t
        num_chunk_data(FieldId) const override {
            return 4;
        }
        int64_t
        size_per_chunk() const override {
            return 2048;
        }
        int64_t
        chunk_size(FieldId, int64_t) const override {
            return 2048;
        }
        int64_t
        num_rows_until_chunk(FieldId, int64_t id) const override {
            return id * 2048;
        }
        mutable int pins = 0;

     protected:
        PinWrapper<SpanBase>
        chunk_data_impl(milvus::OpContext*,
                        FieldId,
                        int64_t chunk) const override {
            ++pins;
            if (chunk == 1) {
                source_.requestCancellation();
            }
            return PinWrapper<SpanBase>(
                SpanBase(values_.data(), 2048, sizeof(int64_t)));
        }

     private:
        folly::CancellationSource& source_;
        std::vector<int64_t> values_;
    };
    auto schema = std::make_shared<Schema>();
    auto field = schema->AddDebugField("group", DataType::INT64);
    schema->set_primary_field_id(field);
    folly::CancellationSource source;
    milvus::OpContext ctx(source.getToken());
    CancellingSegment segment(schema, source);
    try {
        BuildGroupMembership<int64_t>(
            &ctx, segment, field, 8192, {int64_t(1)}, nullptr);
        FAIL() << "membership continued after cancellation";
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), ErrorCode::FollyCancel);
    }
    EXPECT_EQ(segment.pins, 2);
}

TEST(StrictGroupPhase2ExecutorTest, SharedBaseFilterIsLazyAndReleased) {
    SearchResult result;
    auto owner = std::make_shared<TargetBitmap>(1000, false);
    (*owner)[17] = true;
    std::weak_ptr<TargetBitmap> weak_owner = owner;
    result.group_search_filter_owner_ = owner;
    result.SetGroupSearch(BitsetView(*owner),
                          [](const BitsetView& filter, SearchResult&) {
                              EXPECT_TRUE(filter.test(17));
                              EXPECT_TRUE(filter.test(33));
                          });
    EXPECT_EQ(result.group_search_base_filter_, nullptr);
    owner.reset();
    EXPECT_FALSE(weak_owner.expired());
    TargetBitmap extra(1000, false);
    extra[33] = true;
    auto recreated = result.SearchGroup(extra);
    ASSERT_TRUE(recreated.has_value());
    EXPECT_NE(result.group_search_base_filter_, nullptr);
    result.ClearGroupSearch();
    EXPECT_TRUE(weak_owner.expired());
    EXPECT_FALSE(result.CanSearchGroups());
    EXPECT_EQ(result.GetGroupSearchBaseFilter(), nullptr);
}

TEST(GroupMembershipTest, GrowingMmapStringUsesElementView) {
    auto& config = storage::MmapManager::GetInstance().GetMmapConfig();
    const bool previous = config.GetEnableGrowingMmap();
    config.SetEnableGrowingMmap(true);
    auto restore = std::shared_ptr<void>(
        nullptr, [&](void*) { config.SetEnableGrowingMmap(previous); });
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    auto field = schema->AddDebugField("group", DataType::VARCHAR);
    schema->set_primary_field_id(pk);
    auto data = segcore::DataGen(schema, 100, 42, 0, 4);
    auto segment = segcore::CreateGrowingSegment(schema, empty_index_meta);
    auto offset = segment->PreInsert(100);
    segment->Insert(
        offset, 100, data.row_ids_.data(), data.timestamps_.data(), data.raw_);
    auto values = data.get_col<std::string>(field);
    auto* growing = dynamic_cast<segcore::SegmentGrowingImpl*>(segment.get());
    ASSERT_NE(growing, nullptr);
    ASSERT_TRUE(
        growing->get_insert_record().get_data<std::string>(field)->is_mmap());
    std::vector<std::optional<std::string>> groups{values[0]};
    auto membership = BuildGroupMembership<std::string>(
        nullptr, *growing, field, 100, groups, nullptr);
    ASSERT_TRUE(membership.has_value());
    auto bitmap = std::move(membership);
    ASSERT_TRUE(bitmap.has_value());
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ((*bitmap)[i], values[i] == values[0]);
    }
}

TEST(GroupByMapTest, StrictTracksLockedGroupsAndRemainingQuota) {
    GroupByMap<int64_t> groups(/*group_capacity=*/2,
                               /*group_size=*/3,
                               /*strict_group_size=*/true);

    EXPECT_TRUE(groups.Push(7));
    EXPECT_TRUE(groups.Push(7));
    EXPECT_TRUE(groups.Push(7));
    EXPECT_TRUE(groups.IsGroupFull(7));
    EXPECT_EQ(groups.GetRemainingGroupSize(7), 0);
    EXPECT_EQ(groups.GetEnoughGroupCount(), 1);
    EXPECT_FALSE(groups.IsGroupCapacityReached());
    EXPECT_FALSE(groups.IsGroupResEnough());

    EXPECT_TRUE(groups.Push(8));
    EXPECT_TRUE(groups.IsGroupCapacityReached());
    EXPECT_FALSE(groups.IsGroupResEnough());
    EXPECT_EQ(groups.GetGroupResultCount(8), 1);
    EXPECT_EQ(groups.GetRemainingGroupSize(8), 2);

    EXPECT_FALSE(groups.Push(9));
    EXPECT_FALSE(groups.Contains(9));
    EXPECT_EQ(groups.GetGroupCount(), 2);

    EXPECT_TRUE(groups.Push(8));
    EXPECT_TRUE(groups.Push(8));
    EXPECT_TRUE(groups.IsGroupResEnough());
    EXPECT_EQ(groups.GetEnoughGroupCount(), 2);
    EXPECT_FALSE(groups.Push(7));

    ASSERT_EQ(groups.GetGroupOrder().size(), 2);
    EXPECT_EQ(groups.GetGroupOrder()[0], std::optional<int64_t>(7));
    EXPECT_EQ(groups.GetGroupOrder()[1], std::optional<int64_t>(8));
}

TEST(GroupByMapTest, NonStrictStopsWhenCapacityIsReached) {
    GroupByMap<int64_t> groups(/*group_capacity=*/2,
                               /*group_size=*/3,
                               /*strict_group_size=*/false);

    EXPECT_TRUE(groups.Push(1));
    EXPECT_TRUE(groups.Push(1));
    EXPECT_FALSE(groups.IsGroupResEnough());
    EXPECT_TRUE(groups.Push(2));
    EXPECT_TRUE(groups.IsGroupResEnough());
    EXPECT_EQ(groups.GetRemainingGroupSize(1), 1);
    EXPECT_EQ(groups.GetRemainingGroupSize(2), 2);
}

TEST(GroupByMapTest, NullIsTrackedAsAStableGroupKey) {
    GroupByMap<int64_t> groups(/*group_capacity=*/2,
                               /*group_size=*/2,
                               /*strict_group_size=*/true);

    EXPECT_TRUE(groups.Push(std::nullopt));
    EXPECT_TRUE(groups.Contains(std::nullopt));
    EXPECT_EQ(groups.GetGroupResultCount(std::nullopt), 1);
    ASSERT_EQ(groups.GetGroupOrder().size(), 1);
    EXPECT_FALSE(groups.GetGroupOrder()[0].has_value());
}

TEST(GroupByResultCollectorTest, SortsAndAppendsByMetric) {
    {
        GroupByResultCollector<int64_t> collector;
        collector.Add(10, 0.8F, 1);
        collector.Add(20, 0.2F, 2);

        std::vector<GroupByValueType> groups;
        std::vector<int64_t> offsets;
        std::vector<float> distances;
        collector.SortAndAppend(
            knowhere::metric::L2, groups, offsets, distances);

        EXPECT_EQ(offsets, (std::vector<int64_t>{20, 10}));
        EXPECT_EQ(distances, (std::vector<float>{0.2F, 0.8F}));
        EXPECT_EQ(groups.size(), 2);
        EXPECT_EQ(collector.Size(), 0);
    }

    {
        GroupByResultCollector<int64_t> collector;
        collector.Add(10, 0.8F, 1);
        collector.Add(20, 0.2F, 2);

        std::vector<GroupByValueType> groups;
        std::vector<int64_t> offsets;
        std::vector<float> distances;
        collector.SortAndAppend(
            knowhere::metric::IP, groups, offsets, distances);

        EXPECT_EQ(offsets, (std::vector<int64_t>{10, 20}));
        EXPECT_EQ(distances, (std::vector<float>{0.8F, 0.2F}));
        EXPECT_EQ(groups.size(), 2);
        EXPECT_EQ(collector.Size(), 0);
    }
}

TEST(GroupMembershipTest, ScalarIndexAndRawFieldProduceIdenticalMembership) {
    constexpr int64_t kRowCount = 120;
    auto schema = std::make_shared<Schema>();
    auto pk_field = schema->AddDebugField("pk", DataType::INT64);
    auto group_field =
        schema->AddDebugField("nullable_group", DataType::INT64, true);
    schema->set_primary_field_id(pk_field);
    auto data = segcore::DataGen(schema,
                                 kRowCount,
                                 /*seed=*/42,
                                 /*ts_offset=*/0,
                                 /*repeat_count=*/4);

    auto raw_segment = CreateSealedWithFieldDataLoaded(schema, data);
    auto index_segment =
        segcore::CreateSealedSegment(schema, empty_index_meta, 7001);
    LoadGeneratedDataIntoSegment(
        data, index_segment.get(), false, {group_field.get()});

    auto values = data.get_col<int64_t>(group_field);
    auto valid = data.get_col_valid(group_field);
    auto scalar_index = std::make_unique<CountingScalarIndex>();
    auto* counters = scalar_index.get();
    scalar_index->Build(kRowCount, values.data(), valid.data());
    segcore::LoadIndexInfo load_info;
    load_info.field_id = group_field.get();
    load_info.field_type = DataType::INT64;
    load_info.index_params = GenIndexParams(scalar_index.get());
    load_info.cache_index =
        CreateTestCacheIndex("group-membership", std::move(scalar_index));
    index_segment->LoadIndex(load_info);
    auto getter = GetDataGetter<int64_t>(nullptr, *index_segment, group_field);
    for (size_t i = 0; i < kRowCount; ++i) {
        EXPECT_EQ(getter->Get(i),
                  valid[i] ? std::optional<int64_t>(values[i]) : std::nullopt);
    }

    TargetBitmap base_filter(kRowCount, false);
    base_filter[1] = true;  // filtered null
    base_filter[4] = true;  // filtered value group
    base_filter[117] = true;
    std::vector<std::optional<int64_t>> groups{
        std::nullopt, values[0], values[4], values[20]};

    auto raw = BuildGroupMembership<int64_t>(
        nullptr, *raw_segment, group_field, kRowCount, groups, &base_filter);
    auto indexed = BuildGroupMembership<int64_t>(
        nullptr, *index_segment, group_field, kRowCount, groups, &base_filter);
    ASSERT_TRUE(raw.has_value());
    ASSERT_TRUE(indexed.has_value());
    EXPECT_EQ(counters->in_calls, 3);
    EXPECT_EQ(counters->in_values, 3);
    EXPECT_EQ(counters->null_calls, 1);
    // Both sources present: phase two must use raw data, like phase one.
    LoadGeneratedDataIntoSegment(
        data,
        index_segment.get(),
        false,
        {pk_field.get(), RowFieldID.get(), TimestampFieldID.get()});
    ASSERT_TRUE(index_segment->HasFieldData(group_field));
    auto both = BuildGroupMembership<int64_t>(
        nullptr, *index_segment, group_field, kRowCount, groups, &base_filter);
    ASSERT_TRUE(both.has_value());
    EXPECT_EQ(counters->in_calls, 3);
    EXPECT_EQ(counters->null_calls, 1);
    // The union bitmap owns its bits and no longer needs the source column.
    raw_segment->DropFieldData(group_field);
    auto raw_bitmap = std::move(raw);
    auto index_bitmap = std::move(indexed);
    ASSERT_TRUE(raw_bitmap.has_value());
    ASSERT_TRUE(index_bitmap.has_value());
    ASSERT_EQ(raw_bitmap->size(), index_bitmap->size());
    for (size_t i = 0; i < raw_bitmap->size(); ++i) {
        EXPECT_EQ((*raw_bitmap)[i], (*index_bitmap)[i]) << "offset " << i;
        EXPECT_EQ((*raw_bitmap)[i], (*both)[i]) << "offset " << i;
        if (base_filter[i]) {
            EXPECT_FALSE((*raw_bitmap)[i]);
        }
    }
}

TEST(GroupMembershipTest, RawStringBoolAndNullGroupsRespectBaseFilter) {
    constexpr int64_t kRowCount = 40;
    auto schema = std::make_shared<Schema>();
    auto pk_field = schema->AddDebugField("pk", DataType::INT64);
    auto string_field =
        schema->AddDebugField("nullable_string", DataType::VARCHAR, true);
    auto bool_field = schema->AddDebugField("bool_group", DataType::BOOL);
    schema->set_primary_field_id(pk_field);
    auto data = segcore::DataGen(schema,
                                 kRowCount,
                                 /*seed=*/99,
                                 /*ts_offset=*/0,
                                 /*repeat_count=*/4);
    auto segment = CreateSealedWithFieldDataLoaded(schema, data);
    TargetBitmap base_filter(kRowCount, false);
    base_filter[0] = true;
    base_filter[3] = true;
    base_filter[10] = true;

    auto strings = data.get_col<std::string>(string_field);
    auto valid = data.get_col_valid(string_field);
    std::vector<std::optional<std::string>> string_groups{
        std::nullopt, strings[0], strings[8]};
    auto string_membership = BuildGroupMembership<std::string>(nullptr,
                                                               *segment,
                                                               string_field,
                                                               kRowCount,
                                                               string_groups,
                                                               &base_filter);
    ASSERT_TRUE(string_membership.has_value());
    auto string_bitmap = std::move(string_membership);
    ASSERT_TRUE(string_bitmap.has_value());

    for (size_t i = 0; i < kRowCount; ++i) {
        std::optional<std::string> value =
            valid[i] ? std::optional<std::string>(strings[i]) : std::nullopt;
        auto found =
            std::find(string_groups.begin(), string_groups.end(), value);
        auto expected = !base_filter[i] && found != string_groups.end();
        EXPECT_EQ((*string_bitmap)[i], expected) << "offset " << i;
    }

    std::vector<std::optional<bool>> bool_groups{false, true};
    auto bool_membership = BuildGroupMembership<bool>(
        nullptr, *segment, bool_field, kRowCount, bool_groups, &base_filter);
    ASSERT_TRUE(bool_membership.has_value());
    auto bool_bitmap = std::move(bool_membership);
    ASSERT_TRUE(bool_bitmap.has_value());
    EXPECT_EQ(bool_bitmap->count(), kRowCount - base_filter.count());
}

TEST(GroupMembershipTest, RejectsMismatchedFilterSize) {
    auto schema = std::make_shared<Schema>();
    auto pk_field = schema->AddDebugField("pk", DataType::INT64);
    auto group_field = schema->AddDebugField("group", DataType::INT64);
    schema->set_primary_field_id(pk_field);
    auto data = segcore::DataGen(schema, 10);
    auto segment = CreateSealedWithFieldDataLoaded(schema, data);
    TargetBitmap wrong_size(9, false);

    auto membership = BuildGroupMembership<int64_t>(
        nullptr, *segment, group_field, 10, {0}, &wrong_size);
    EXPECT_FALSE(membership.has_value());
}

TEST(StrictGroupSearchTest, FullGroupsAreSearchedAndMergedByMetric) {
    for (const auto& metric : {knowhere::metric::L2, knowhere::metric::IP}) {
        auto schema = std::make_shared<Schema>();
        auto pk = schema->AddDebugField("pk", DataType::INT64);
        auto field = schema->AddDebugField("group", DataType::INT64);
        schema->set_primary_field_id(pk);
        auto data = segcore::DataGen(schema, 100, 42, 0, 10);
        auto segment = CreateSealedWithFieldDataLoaded(schema, data);
        auto labels = data.get_col<int64_t>(field);
        std::unordered_map<int64_t, std::vector<int64_t>> rows;
        for (int64_t i = 0; i < 100; ++i) rows[labels[i]].push_back(i);
        ASSERT_GE(rows.size(), 2);
        auto it = rows.begin();
        auto a = it++->second;
        auto b = it->second;
        ASSERT_GE(a.size(), 5);
        ASSERT_GE(b.size(), 5);
        auto sign = metric == knowhere::metric::L2 ? 1.0F : -1.0F;
        std::vector<std::pair<int64_t, float>> candidates{{a[0], sign * 100},
                                                          {a[1], sign * 90},
                                                          {b[0], sign * 80},
                                                          {b[1], sign * 70},
                                                          {a[2], sign * 60}};
        SearchResult result;
        result.total_nq_ = 1;
        result.total_data_cnt_ = 100;
        result.vector_iterators_ = std::vector<std::shared_ptr<VectorIterator>>{
            MakeSequenceVectorIterator(candidates)};
        TargetBitmap base(100, false);
        base[a[4]] = true;
        int calls = 0;
        result.SetGroupSearch(
            BitsetView(base),
            [&](const BitsetView& filter, SearchResult& batch) {
                auto label = calls == 0 ? labels[a[0]] : labels[b[0]];
                for (size_t i = 0; i < 100; ++i) {
                    EXPECT_EQ(filter.test(i), base[i] || labels[i] != label);
                }
                // First group was already full in phase one; its worse hits must
                // both be replaceable. The second Search repeats a phase-one hit.
                batch.seg_offsets_ = calls == 0
                                         ? std::vector<int64_t>{a[2], a[3]}
                                         : std::vector<int64_t>{b[1], b[0]};
                batch.distances_ =
                    calls == 0 ? std::vector<float>{sign * 1, sign * 2}
                               : std::vector<float>{sign * 3, sign * 80};
                ++calls;
            });
        SearchInfo info;
        info.topk_ = 2;
        info.group_size_ = 2;
        info.strict_group_size_ = true;
        info.enable_search_path_ = true;
        info.enable_search_path_k_ = 2;
        info.group_by_field_id_ = field;
        info.metric_type_ = metric;
        std::vector<GroupByValueType> groups;
        std::vector<int64_t> offsets;
        std::vector<float> distances;
        std::vector<size_t> prefix;
        SearchGroupBy(nullptr,
                      *result.vector_iterators_,
                      info,
                      groups,
                      *segment,
                      offsets,
                      distances,
                      prefix,
                      &result);
        EXPECT_EQ(calls, 2);
        EXPECT_EQ(offsets, (std::vector<int64_t>{a[2], a[3], b[1], b[0]}));
        EXPECT_EQ(prefix, (std::vector<size_t>{0, 4}));
        EXPECT_FALSE(result.CanSearchGroups());
    }
}

TEST(StrictGroupSearchTest,
     UnderfilledSearchResumesOriginalAndPreservesErrors) {
    auto schema = std::make_shared<Schema>();
    auto field = schema->AddDebugField("group", DataType::INT64);
    schema->set_primary_field_id(field);
    auto data = segcore::DataGen(schema, 100, 42, 0, 10);
    auto segment = CreateSealedWithFieldDataLoaded(schema, data);
    auto labels = data.get_col<int64_t>(field);
    std::vector<int64_t> same;
    for (int64_t i = 0; i < 100; ++i)
        if (labels[i] == labels[0])
            same.push_back(i);
    ASSERT_GE(same.size(), 3);
    for (auto code : {ErrorCode::Success,
                      ErrorCode::FollyCancel,
                      ErrorCode::DataFormatBroken}) {
        SearchResult result;
        result.total_nq_ = 1;
        result.total_data_cnt_ = 100;
        result.vector_iterators_ = std::vector<std::shared_ptr<VectorIterator>>{
            MakeSequenceVectorIterator(
                {{same[0], 1}, {same[1], 2}, {same[2], 3}})};
        result.SetGroupSearch({}, [&](const BitsetView&, SearchResult& batch) {
            if (code != ErrorCode::Success)
                throw SegcoreError(code, "injected group Search failure");
            batch.seg_offsets_ = {same[0], INVALID_SEG_OFFSET};
            batch.distances_ = {1, 0};
        });
        SearchInfo info;
        info.topk_ = 1;
        info.group_size_ = 3;
        info.strict_group_size_ = true;
        info.enable_search_path_ = true;
        info.group_by_field_id_ = field;
        info.metric_type_ = knowhere::metric::L2;
        std::vector<GroupByValueType> groups;
        std::vector<int64_t> offsets;
        std::vector<float> distances;
        std::vector<size_t> prefix;
        try {
            SearchGroupBy(nullptr,
                          *result.vector_iterators_,
                          info,
                          groups,
                          *segment,
                          offsets,
                          distances,
                          prefix,
                          &result);
            EXPECT_EQ(code, ErrorCode::Success);
            EXPECT_EQ(offsets,
                      (std::vector<int64_t>{same[0], same[1], same[2]}));
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), code);
        }
        EXPECT_FALSE(result.CanSearchGroups());
    }
}

TEST(StrictGroupSearchTest, ClearsIteratorAndInheritedBudgetParameters) {
    SearchInfo info;
    info.topk_ = 100;
    info.group_size_ = 3;
    info.group_by_field_id_ = FieldId(100);
    info.strict_group_size_ = true;
    info.round_decimal_ = 2;
    info.iterative_filter_execution = true;
    info.iterator_v2_info_.emplace();
    info.materialized_view_involved = true;
    info.metric_type_ = knowhere::metric::IP;
    info.search_params_ = {{"ef", 1000},
                           {"search_list_size", 1000},
                           {"search_list", 1000},
                           {"iterator_ef", 50},
                           {"iterator_refine_ratio", .5},
                           {"retain_iterator_order", true},
                           {"hints", "iterative_filter"},
                           {"materialized_view_search_info", {{"stale", true}}},
                           {"nprobe", 8}};
    auto search = info.ForGroupSearch();
    EXPECT_EQ(search.topk_, 3);
    EXPECT_EQ(search.round_decimal_, -1);
    EXPECT_FALSE(search.group_by_field_id_);
    EXPECT_FALSE(search.strict_group_size_);
    EXPECT_FALSE(search.iterative_filter_execution);
    EXPECT_FALSE(search.iterator_v2_info_);
    EXPECT_FALSE(search.materialized_view_involved);
    EXPECT_EQ(search.metric_type_, knowhere::metric::IP);
    EXPECT_EQ(search.search_params_, (knowhere::Json{{"nprobe", 8}}));
    EXPECT_EQ(info.topk_, 100);
    EXPECT_TRUE(info.search_params_.contains("ef"));
}

TEST(StrictGroupSearchTest, GrowingConfigurationKeepsGroupKAndBm25Context) {
    FieldIndexMeta meta(FieldId(100),
                        {{"index_type", "SPARSE_INVERTED_INDEX"},
                         {"metric_type", "BM25"},
                         {"bm25_k1", "1.2"},
                         {"bm25_b", "0.75"}},
                        {});
    auto& config = segcore::SegcoreConfig::default_config();
    segcore::VecIndexConfig index_config(
        10000, meta, config, SegmentType::Growing, true);
    SearchInfo info;
    info.group_size_ = 3;
    info.topk_ = 100;
    info.search_params_ = {{"bm25_avgdl", 42.5},
                           {"ef", 100},
                           {"search_list", 100},
                           {"search_list_size", 100}};
    auto params = index_config.GetSearchConf(info.ForGroupSearch());
    EXPECT_EQ(params.topk_, 3);
    EXPECT_EQ(params.metric_type_, knowhere::metric::BM25);
    EXPECT_EQ(params.search_params_["bm25_avgdl"], 42.5);
    for (const auto* key : {"ef", "search_list_size", "search_list"}) {
        EXPECT_FALSE(params.search_params_.contains(key));
    }
}

}  // namespace milvus::exec
