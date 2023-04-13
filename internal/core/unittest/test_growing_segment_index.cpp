// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <gtest/gtest.h>

#include "segcore/SegmentGrowing.h"
#include "segcore/SegmentGrowingImpl.h"
#include "pb/schema.pb.h"
#include "test_utils/DataGen.h"
#include "query/Plan.h"

using namespace milvus::segcore;
using namespace milvus;
namespace pb = milvus::proto;

TEST(GrowingIndex, Correctness) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    auto random = schema->AddDebugField("random", DataType::DOUBLE);
    auto vec = schema->AddDebugField("embeddings", DataType::VECTOR_FLOAT, 128, knowhere::metric::L2);
    schema->set_primary_field_id(pk);

    std::map<std::string, std::string> index_params = {{"index_type", "IVF_FLAT"}, {"metric_type", "L2"}, {"nlist", "128"}};
    std::map<std::string, std::string> type_params = {{"dim", "128"}};
    FieldIndexMeta fieldIndexMeta(vec, std::move(index_params),  std::move(type_params));
    auto& config = SegcoreConfig::default_config();
    config.set_chunk_rows(1024);
    config.set_enable_growing_segment_index(true);
    std::map<FieldId, FieldIndexMeta> filedMap = {{vec, fieldIndexMeta}};
    IndexMetaPtr metaPtr = std::make_shared<CollectionIndexMeta>(226985, std::move(filedMap));
    auto segment = CreateGrowingSegment(schema, metaPtr);

    std::string dsl = R"({
        "bool": {
            "must": [
            {
                "vector": {
                    "embeddings": {
                        "metric_type": "l2",
                        "params": {
                            "nprobe": 10
                        },
                        "query": "$0",
                        "topk": 5,
                        "round_decimal":3
                    }
                }
            }
            ]
        }
    })";

    int64_t per_batch = 50000;
    int64_t n_batch = 20;
    int64_t n_query = 10000;
    for (int64_t i = 0; i < n_batch; i++) {
        auto dataset = DataGen(schema, per_batch);
        auto queryset = DataGen(schema, per_batch);
        auto offset = segment->PreInsert(per_batch);
        auto pks = dataset.get_col<int64_t>(pk);
        segment->Insert(offset,
                        per_batch,
                        dataset.row_ids_.data(),
                        dataset.timestamps_.data(),
                        dataset.raw_);

        auto plan = milvus::query::CreatePlan(*schema, dsl);
        auto num_queries = 5;
        auto ph_group_raw = CreatePlaceholderGroup(num_queries, 128, 1024);
        auto ph_group =
            ParsePlaceholderGroup(plan.get(), ph_group_raw.SerializeAsString());
        Timestamp time = 1000000;
        auto sr = segment->Search(plan.get(), ph_group.get(), time);
    }
}
