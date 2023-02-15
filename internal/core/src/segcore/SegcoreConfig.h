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
#include "exceptions/EasyAssert.h"
#include "utils/Json.h"
#include "log/Log.h"

namespace milvus::segcore {

struct SmallIndexConf {
    std::string index_type;
    nlohmann::json build_params;
    nlohmann::json search_params;
};

class SegcoreConfig {
 private:
    SegcoreConfig() {
        // hard code configurations for small index
        SmallIndexConf sub_conf;
        table_[knowhere::metric::L2] = sub_conf;
        table_[knowhere::metric::IP] = sub_conf;
    }

 public:
    static SegcoreConfig&
    default_config() {
        // TODO: remove this when go side is ready
        static SegcoreConfig config;
        return config;
    }

    static SegcoreConfig
    gen_index_config(const std::string index_type) {
        SegcoreConfig config;
        if (index_type == "hnsw") {
            config.set_index_type(knowhere::IndexEnum::INDEX_HNSW);
        } else if (index_type == "ivfflat") {
            config.set_index_type(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT);
            LOG_SEGCORE_DEBUG_ << "set config type : "<< knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;
        } else if (index_type == "ivfpqfastscan") {
            config.set_index_type(knowhere::IndexEnum::INDEX_FAISS_IVFPQFASTSCAN);
        }
        return config;
    }

    const char*
    get_index_type() const {
        return vec_index_type_;
    }

    void
    parse_from(const std::string& string_path);

    const SmallIndexConf&
    at(const MetricType& metric_type) const {
        Assert(table_.count(metric_type));
        return table_.at(metric_type);
    }

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

    void
    set_small_index_config(const MetricType& metric_type, const SmallIndexConf& small_index_conf) {
        table_[metric_type] = small_index_conf;
    }

    void
    set_index_type(const char * index_type) {
        vec_index_type_ = index_type;
    }

 private:
    int64_t chunk_rows_ = 32 * 1024;
    int64_t nlist_ = 100;
    int64_t nprobe_ = 4;
    std::map<knowhere::MetricType, SmallIndexConf> table_;
    const char* vec_index_type_;
};

}  // namespace milvus::segcore
