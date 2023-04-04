#pragma once

#include <unordered_map>
#include <iostream>
#include <string>
#include <atomic>
#include <chrono>
#include <mutex>
#include "log/Log.h"

#define ENABLE_PROFILING

namespace milvus::segcore {

static std::string
rangeStr(int64_t beg, int64_t size) {
    return "[" + std::to_string(beg) + "," + std::to_string(beg + size) + "]";
}

class TimeProfiler {
 public:
    TimeProfiler(__attribute__((unused)) std::string name) {
#ifdef ENABLE_PROFILING
       enter_time_ =  std::chrono::steady_clock::now();
       entry_name_ = name;
#endif
    }

    virtual ~TimeProfiler() {
    }

    void
    report() {
#ifdef ENABLE_PROFILING
       auto now =  std::chrono::steady_clock::now();
       LOG_SEGCORE_INFO_<<"Segment Profiling :"<< entry_name_ << " cost "<< std::chrono::duration_cast<std::chrono::microseconds>(now - enter_time_).count()/ 1000000.0 << "[s]";
#endif
    }

    void
    reportRate(int64_t size) {
#ifdef ENABLE_PROFILING
       auto now =  std::chrono::steady_clock::now();
       double consume = std::chrono::duration_cast<std::chrono::microseconds>(now - enter_time_).count()/ 1000000.0;
       LOG_SEGCORE_INFO_<<"Segment Profiling :"<< entry_name_ << " cost "<< consume << "[s], rate: "<< size / consume << " per second";
#endif
    }
 private:
#ifdef ENABLE_PROFILING
    std::string entry_name_;
    std::chrono::steady_clock::time_point enter_time_;
#endif
};
}  // namespace cardinal
