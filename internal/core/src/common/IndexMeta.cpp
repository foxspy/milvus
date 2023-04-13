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

#include <memory>

#include "IndexMeta.h"
#include "protobuf_utils.h"
#include "log/Log.h"

namespace milvus {

FieldIndexMeta::FieldIndexMeta(FieldId fieldId, std::map<std::string, std::string>&& index_params, std::map<std::string, std::string>&& type_params ) {
    fieldId_ = fieldId;
    index_params_ = std::move(index_params);
    type_params_ = std::move(type_params);
}


FieldIndexMeta::FieldIndexMeta(const milvus::proto::meta::FieldIndexMeta& fieldIndexMeta) {
    fieldId_ = FieldId(fieldIndexMeta.fieldid());
    index_params_ = RepeatedKeyValToMap(fieldIndexMeta.index_params());
    type_params_ = RepeatedKeyValToMap(fieldIndexMeta.type_params());
    user_index_params_ = RepeatedKeyValToMap(fieldIndexMeta.user_index_params());
}

CollectionIndexMeta::CollectionIndexMeta(int64_t max_segment_row_cnt, std::map<FieldId, FieldIndexMeta>&& fieldMetas): max_segment_row_cnt_(max_segment_row_cnt), fieldMetas_(std::move(fieldMetas)) {
}

CollectionIndexMeta::CollectionIndexMeta(const milvus::proto::meta::CollectionIndexMeta& collectionIndexMeta) {
    max_segment_row_cnt_ = collectionIndexMeta.maxrowcount();
    for (auto& filed_index_meta : collectionIndexMeta.index_metas()) {
        FieldIndexMeta fieldIndexMeta(filed_index_meta);
        fieldMetas_.emplace(FieldId(filed_index_meta.fieldid()), fieldIndexMeta);
    }
}

int64_t
CollectionIndexMeta::GetSegmentMaxRowCount() const {
    return max_segment_row_cnt_;
}

bool
CollectionIndexMeta::has_field(FieldId fieldId) const {
    return fieldMetas_.count(fieldId);
}

const FieldIndexMeta&
CollectionIndexMeta::GetFieldIndexMeta(FieldId fieldId) const {
    if (fieldMetas_.find(fieldId) != fieldMetas_.end()) {
        return fieldMetas_.at(fieldId);
    }
}

void
CollectionIndexMeta::PrintParams() {
    std::stringstream ss;
    ss<<"maxRowCount : {"<< max_segment_row_cnt_<<"} ";
    for (auto& filed_meta : fieldMetas_) {
        ss<<"FieldId : {"<< abs(filed_meta.first.get())<<" ";
        ss<<"IndexParams : { ";
        for (auto& kv : filed_meta.second.get_index_params()) {
            ss<<kv.first<<" : "<<kv.second<<", ";
        }
        ss<<" }";
        ss<<"TypeParams : {";
        for (auto& kv : filed_meta.second.get_type_params()) {
            ss<<kv.first<<" : "<<kv.second<<", ";
        }
        ss<<"}";
        ss<<"}";
    }
    LOG_SEGCORE_INFO_<<"IndexMeta: {"<<ss.str()<<" } ";
}

}
