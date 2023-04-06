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

#pragma once

#include <map>
#include <string>

#include "common/Types.h"
#include "index/Utils.h"
#include "exceptions/EasyAssert.h"
#include "utils/Json.h"

namespace milvus::segcore {

class SegcoreConfig {
 public:
    static SegcoreConfig&
    default_config() {
        // TODO: remove this when go side is ready
        static SegcoreConfig config;
        return config;
    }

    void
    parse_from(const std::string& string_path);

    int64_t
    get_chunk_rows() const {
        return chunk_rows_;
    }

    void
    set_chunk_rows(int64_t chunk_rows) {
        chunk_rows_ = chunk_rows;
    }

    void
    set_nlist(int64_t nlist) {
        nlist_ = nlist;
    }

    void
    set_nprobe(int64_t nprobe) {
        nprobe_ = nprobe;
    }

    int64_t
    get_train_threshold() const {
        if (index::is_in_no_train_list(index_type_)) {
            return 0;
        } else {
            return build_threshold ;
        }
    }

    knowhere::IndexType
    get_index_type() const {
        return index_type_;
    }

    knowhere::MetricType
    get_metric_type() const {
        return metric_type_;
    }

 //private:
    bool enable_segment_index_ = false;
    knowhere::IndexType index_type_;
    knowhere::MetricType  metric_type_;
    int64_t build_threshold = 100000;
    int64_t chunk_rows_ = 32 * 1024;
    int64_t segment_size_ = 1024;
    int64_t nlist_ = 100;
    int64_t nprobe_ = 4;
};

}  // namespace milvus::segcore
