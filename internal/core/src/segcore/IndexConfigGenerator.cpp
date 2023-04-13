#include "IndexConfigGenerator.h"
#include "log/Log.h"

namespace milvus::segcore {

VecIndexConfig::VecIndexConfig(const int64_t max_index_row_cout, const FieldIndexMeta& index_meta_, const SegcoreConfig& config) : max_index_row_count_(max_index_row_cout), config_(config) {
    origin_index_type_ = index_meta_.get_index_type();
    metric_type_ = index_meta_.get_metric_type();

    index_type_ = support_index_types[0];
    build_params_[knowhere::meta::METRIC_TYPE] = metric_type_;
    build_params_[knowhere::indexparam::NLIST] = std::to_string(config_.get_nlist());
    build_params_[knowhere::indexparam::SSIZE] = std::to_string(std::max((int)(config_.get_chunk_rows() / config_.get_nlist()), 48));
    search_params_[knowhere::indexparam::NPROBE] = std::to_string(config_.get_nprobe());
    LOG_SEGCORE_INFO_ << "VecIndexConfig: "
                      <<"origin_index_type_:" << origin_index_type_
                      <<" index_type_: " << index_type_
                      <<" metric_type_: " << metric_type_;
}

int64_t
VecIndexConfig::getBuildThreshold() const noexcept  {
    assert(VecIndexConfig::index_build_ratio.count(index_type_));
    auto ratio = VecIndexConfig::index_build_ratio.at(index_type_);
    assert(ratio >= 0.0 && ratio < 1.0);
    return max_index_row_count_ * ratio;
}

knowhere::IndexType
VecIndexConfig::getIndexType() noexcept {
    return index_type_;
}


knowhere::MetricType
VecIndexConfig::getMetricType() noexcept {
    return metric_type_;
}

knowhere::Json
VecIndexConfig::getBuildBaseParams() {
    return build_params_;
}

SearchInfo
VecIndexConfig::getSearchConf(const SearchInfo& searchInfo) {
    SearchInfo searchParam(searchInfo);
    searchParam.metric_type_ = metric_type_;
    searchParam.search_params_ = search_params_;
    return searchParam;
}

}