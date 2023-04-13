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

#pragma once

#include <memory>
#include <unordered_map>

#include "pb/common.pb.h"
#include "pb/meta.pb.h"
#include "Types.h"

namespace milvus {
class FieldIndexMeta {
 public:
    //For unittest init
    FieldIndexMeta(FieldId fieldId, std::map<std::string, std::string>&& index_params, std::map<std::string, std::string>&& type_params );

    FieldIndexMeta(const milvus::proto::meta::FieldIndexMeta& fieldIndexMeta);

    knowhere::MetricType
    get_metric_type() const {
        return index_params_.at(knowhere::meta::METRIC_TYPE);
    }

    knowhere::IndexType
    get_index_type() const {
        return index_params_.at(knowhere::meta::INDEX_TYPE);
    }

    const std::map<std::string, std::string>&
    get_index_params() const {
        return index_params_;
    }

    const std::map<std::string, std::string>&
    get_type_params() const {
        return type_params_;
    }

    std::optional<std::string>
    get_index_param(std::string param_key) const {
        if (index_params_.find(param_key) != index_params_.end())  {
            return index_params_.at(param_key);
        }
        std::nullopt_t;
    }
 private:
    FieldId fieldId_;
    std::map<std::string, std::string> index_params_;
    std::map<std::string, std::string> type_params_;
    std::map<std::string, std::string> user_index_params_;
};

class CollectionIndexMeta {
 public:
    //just for unittest
    CollectionIndexMeta(int64_t max_segment_row_cnt, std::map<FieldId, FieldIndexMeta>&& fieldMetas);

    CollectionIndexMeta(const milvus::proto::meta::CollectionIndexMeta& collectionIndexMeta);

    int64_t GetSegmentMaxRowCount() const;

    bool has_field(FieldId fieldId) const;

    const FieldIndexMeta& GetFieldIndexMeta(FieldId fieldId) const;

    void PrintParams();
 public:
    std::string collection_name_;
 private:
    int64_t max_segment_row_cnt_;
    std::map<FieldId, FieldIndexMeta> fieldMetas_;
};

using IndexMetaPtr = std::shared_ptr<CollectionIndexMeta>;

}  //namespace milvus