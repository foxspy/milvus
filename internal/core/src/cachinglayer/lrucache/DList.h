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

#include <atomic>
#include <mutex>

#include <folly/futures/Future.h>
#include <folly/futures/SharedPromise.h>

#include "cachinglayer/lrucache/ListNode.h"

namespace milvus::cachinglayer::internal {

class DList {
 public:
    // Touch a node means to move it to the head of the list, which requires locking the entire list.
    // Use TouchConfig to reduce the frequency of touching and reduce contention.
    struct TouchConfig {
        std::chrono::seconds refresh_window = std::chrono::seconds(10);
    };

    DList(size_t max_memory, TouchConfig touch_config)
        : max_memory_(max_memory), touch_config_(touch_config) {
    }

 private:
    ListNode* head_ = nullptr;
    ListNode* tail_ = nullptr;
    mutable std::mutex list_mtx_;
    std::atomic<size_t> used_memory_{0};
    const size_t max_memory_;
    const TouchConfig touch_config_;

    // items are evicted because they are not used for a while, thus it should be ok to lock them
    // a little bit longer.
    bool
    reserveMemory(size_t size);
    // used only when load failed, ListNode is not in the list, thus it's safe to substract on atomic without lock.
    // this will only cause used_memory_ to decrease, which will not affect the correctness of concurrent
    // reserveMemory().
    void
    releaseMemoryWhenLoadFailed(size_t size);
    // caller must guarantee that the current thread holds the lock of list_node->mtx_.
    void
    touchItem(ListNode* list_node);
    // must be called under the lock of list_mtx_ and list_node->mtx_
    // ListNode is guaranteed to be not in the list.
    void
    pushHead(ListNode* list_node);
    // must be called under the lock of list_mtx_ and list_node->mtx_,
    // if ListNode is not in the list, this function does nothing.
    void
    popItem(ListNode* list_node);

    friend class ListNode;
};

}  // namespace milvus::cachinglayer::internal
