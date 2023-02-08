#pragma once

#include <unordered_map>
#include <iostream>
#include <string>
#include <atomic>
#include <chrono>
#include <mutex>

#define ENABLE_PROFILING

namespace milvus::segcore {
class TimeProfiler {
 public:
    TimeProfiler(__attribute__((unused)) std::string name) {
#ifdef ENABLE_PROFILING
        if (time_table_.find(name) == time_table_.end()) {
            std::lock_guard<std::mutex> lock(time_table_mutex);
            if (time_table_.find(name) == time_table_.end()) {
                time_table_[name] = 0;
                cnt_table_[name] = 0;
            }
        }
        entry_name_ = name;
        enter_time_ = std::chrono::steady_clock::now();
        if (enter_time_ < global_enter_time_) {
            global_enter_time_ = enter_time_;
        }
#endif
    }

    virtual ~TimeProfiler() {
#ifdef ENABLE_PROFILING
        auto exit_time_ = std::chrono::steady_clock::now();
        global_exit_time_ = exit_time_;
        time_table_[entry_name_] += std::chrono::duration_cast<std::chrono::nanoseconds>(exit_time_ - enter_time_).count();
        cnt_table_[entry_name_]++;
#endif
    }

    static std::string
    report() {
#ifdef ENABLE_PROFILING
        struct Stat{
            std::string entry_name;
            long long time;
            long long cnt;
        };
        long long all_cost = std::chrono::duration_cast<std::chrono::nanoseconds>(global_exit_time_ - global_enter_time_).count();
        std::vector<struct Stat> time_lists;
        {
            std::lock_guard<std::mutex> lock(time_table_mutex);
            for (auto& entry : time_table_) {
                time_lists.push_back({entry.first, entry.second, cnt_table_[entry.first]});
            }
        }
        std::stringstream str;
        std::sort(time_lists.begin(), time_lists.end(), [](const Stat& a, const Stat& b){ return a.time > b.time; });
        str<<" ALL : "<<all_cost / 1000000.0 <<"[ms]"<<std::endl;
        for (auto& entry : time_lists) {
            str<<entry.time / 1000000.0<<"[ms]"<<"("<< (double)entry.time/all_cost * 100.0 <<"%)"<<" : " <<entry.entry_name<<" ("<<entry.cnt<<") "<<std::endl;
        }
        return str.str();
#endif
    }

    static void clear() {
#ifdef ENABLE_PROFILING
        std::lock_guard<std::mutex> lock(time_table_mutex);
        time_table_.clear();
        cnt_table_.clear();
        global_enter_time_ = std::chrono::steady_clock::time_point::max();
#endif
    }
 private:
#ifdef ENABLE_PROFILING
    std::string entry_name_;
    std::chrono::steady_clock::time_point enter_time_;
    inline static std::chrono::steady_clock::time_point global_enter_time_ = std::chrono::steady_clock::time_point::max();
    inline static std::chrono::steady_clock::time_point global_exit_time_;
    inline static std::mutex time_table_mutex;
    inline static std::unordered_map<std::string, long long> cnt_table_;
    inline static std::unordered_map<std::string, long long> time_table_;
#endif
};
#ifdef ENABLE_PROFILING
#define PROFILING(X) cardinal::TimeProfiler profile(X)
#define PROFILING_SCOPE(X) cardinal::TimeProfiler profile_scope(X)
#else
#define PROFILING(X) ;
#define PROFILING_SCOPE(X) ;
#endif
}  // namespace cardinal
