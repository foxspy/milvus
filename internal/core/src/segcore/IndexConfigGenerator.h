#pragma once

#include "common/Types.h"
#include "common/IndexMeta.h"
#include "knowhere/config.h"
#include "SegcoreConfig.h"
#include "common/QueryInfo.h"

namespace milvus::segcore {

enum class IndexConfigLevel {
    UNKNOWN = 0,
    SUPPORT = 1,
    COMPATIBLE = 2,
    SYSTEM_ASSIGN = 3
};

class VecIndexConfig {
    inline static const std::vector<std::string> support_index_types = {knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC};

    inline static const std::map<std::string, double> index_build_ratio = {{knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, 0.1}};

 public:
    VecIndexConfig(const int64_t max_index_row_count, const FieldIndexMeta& index_meta_, const SegcoreConfig& config);

    int64_t getBuildThreshold() const noexcept;

    knowhere::IndexType getIndexType() noexcept;

    knowhere::MetricType getMetricType() noexcept;

    knowhere::Json getBuildBaseParams();

    SearchInfo getSearchConf(const SearchInfo& searchInfo);

 private:
    const SegcoreConfig& config_;

    int64_t max_index_row_count_;

    knowhere::IndexType origin_index_type_;

    knowhere::IndexType index_type_;

    knowhere::MetricType metric_type_;

    knowhere::Json build_params_;

    knowhere::Json search_params_;
};
}  //namespace cardinal