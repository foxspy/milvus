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
#include <chrono>
#include <memory>

#include <folly/ExceptionWrapper.h>
#include <folly/futures/Future.h>
#include <folly/futures/SharedPromise.h>

#include "cachinglayer/Utils.h"

namespace milvus::cachinglayer::internal {

class DList;

// ListNode is not movable/copyable.
class ListNode {
 public:
    // RAII class to unpin the node.
    class Pin {
     public:
        // Pin is movable but not copyable.
        Pin(Pin&&);
        Pin&
        operator=(Pin&&);
        Pin(const Pin&) = delete;
        Pin&
        operator=(const Pin&) = delete;
        ~Pin();

     private:
        Pin(ListNode* node);
        friend class ListNode;
        ListNode* node_;
    };
    ListNode() = default;
    ListNode(DList* dlist);
    virtual ~ListNode();

    // ListNode is not movable/copyable because it contains a shared_mutex.
    // ListNode also should not be movable/copyable because that would make
    // all Pins::node_ dangling pointers.

    folly::SemiFuture<Pin>
    pin();

 protected:
    // NOT_LOADED ---> LOADING ---> ERROR
    //      ^            |
    //      |            v
    //      |------- LOADED
    enum class State { NOT_LOADED, LOADING, LOADED, ERROR };

    // subclasses can assume load()/unload() will not be called concurrently,
    // thus when called it can modify their private members without additional locking.
    virtual folly::SemiFuture<folly::Unit>
    load() = 0;
    // implementation should release all resources, will be called during eviction.
    virtual void
    unload() = 0;
    // resource usage size.
    virtual size_t
    size() = 0;
    virtual const std::string&
    key() const = 0;
    virtual const cid_t&
    cid() const = 0;

    void
    mark_loaded(std::function<void()>&& cb, bool requesting_thread);

    State state_ = State::NOT_LOADED;

    static std::string
    state_to_string(State state);

 private:
    friend class DList;
    friend class Pin;

    // called by DList during eviction.
    // must be called under the lock of mtx_.
    void
    clear_data();
    void
    unpin();
    // Touch the node if it has not been touched for a while.
    // Will insert this node into the head of DList if not already in the list.
    // When do we call this method?
    // 1. When this node is unpinned.
    // 2. When this node is marked loaded by a non-requesting thread.
    // must be called under the lock of mtx_.
    void
    touch();

    mutable std::shared_mutex mtx_;
    std::chrono::steady_clock::time_point last_touch_;
    DList* dlist_;
    ListNode* prev_ = nullptr;
    ListNode* next_ = nullptr;

    // When updating state under mtx_, ensure the correctness of the following
    // members:
    std::atomic<int> pin_count_{0};
    std::unique_ptr<folly::SharedPromise<folly::Unit>> load_promise_{nullptr};
    folly::exception_wrapper error_;
};

}  // namespace milvus::cachinglayer::internal
