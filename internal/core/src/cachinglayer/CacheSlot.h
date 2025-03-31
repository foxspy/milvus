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

#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <vector>

#include <folly/futures/Future.h>
#include <folly/futures/SharedPromise.h>
#include <folly/Synchronized.h>

#include "cachinglayer/EvictionManager.h"
#include "cachinglayer/lrucache/ListNode.h"
#include "cachinglayer/Translator.h"
#include "cachinglayer/Utils.h"
#include "common/Types.h"
#include "log/Log.h"

namespace milvus::cachinglayer {

template <typename CellT>
class CellAccessor;

// - The action of pinning cells is not started until the returned SemiFuture is scheduled on an executor.
// - Once the future is scheduled, CacheSlot must live until both the future is ready and the returned CellAccessor is
//   destroyed. In other words, ~CacheSlot() must not be called when there are pending futures or live CellAccessors.
template <typename CellT>
class CacheSlot final {
 public:
    static_assert(
        std::is_same_v<size_t, decltype(std::declval<CellT>().DataByteSize())>,
        "CellT must have a DataByteSize() method that returns a size_t "
        "representing the memory consumption of the cell");

    CacheSlot(std::unique_ptr<Translator<CellT>> translator,
              EvictionManager* eviction_manager)
        : cells_(translator->num_cells()),
          translator_(std::move(translator)),
          em_(eviction_manager),
          load_delay_ms_(10) {
        // em_->register_slot(slot_id(), cells_.size());
        for (cid_t i = 0; i < translator_->num_cells(); ++i) {
            new (&cells_[i])
                CacheCell(this, i, translator_->estimated_byte_size_of_cell(i));
        }
    }

    CacheSlot(const CacheSlot&) = delete;
    CacheSlot&
    operator=(const CacheSlot&) = delete;
    CacheSlot(CacheSlot&&) = delete;
    CacheSlot&
    operator=(CacheSlot&&) = delete;

    // `uids` must be valid until the returned future is ready. PinCells does not copy `uids`.
    // TODO: may want to change to std::shared_ptr of uids?
    folly::SemiFuture<std::unique_ptr<CellAccessor<CellT>>>
    PinCells(const uid_t* uids, size_t count) {
        return folly::makeSemiFuture().deferValue([this, uids, count](auto&&) {
            BitsetType bitset(cells_.size());
            std::vector<cid_t> involved_cids;
            involved_cids.reserve(cells_.size());
            for (size_t i = 0; i < count; ++i) {
                auto uid = uids[i];
                auto cid = translator_->cell_id_of(uid);
                if (cid >= cells_.size()) {
                    return folly::makeSemiFuture<
                        std::unique_ptr<CellAccessor<CellT>>>(
                        folly::make_exception_wrapper<std::invalid_argument>(
                            fmt::format(
                                "CacheSlot {}: translator returned cell_id {} "
                                "for uid {} which is out of range",
                                translator_->key(),
                                cid,
                                uid)));
                }
                if (!bitset[cid]) {
                    bitset[cid] = true;
                    involved_cids.push_back(cid);
                }
            }
            std::vector<folly::SemiFuture<internal::ListNode::Pin>> futures;
            futures.reserve(involved_cids.size());
            for (auto cid : involved_cids) {
                futures.push_back(cells_[cid].pin());
            }
            return folly::collect(futures).deferValue(
                [this](auto&& pins) mutable {
                    // or else unpinning is done in CellAccessor's destructor.
                    return std::make_unique<CellAccessor<CellT>>(
                        this, std::move(pins));
                });
        });
    }

    // CacheSlot is created/destroyed along with segment load/release.
    // CacheSlot must out live all returned CellAccessors.
    ~CacheSlot() {
        // em_->unregister_slot(slot_id(), cells_.size());
    }

 private:
    friend class CellAccessor<CellT>;
    friend class EvictionManager;

    cid_t
    cell_id_of(uid_t uid) const {
        return translator_->cell_id_of(uid);
    }

    // When called, RunLoad uses translator to load all NOT_LOADED cells in load_queue_.
    folly::SemiFuture<folly::Unit>
    RunLoad() {
        return folly::makeSemiFuture().deferValue([this](auto&&)
                                                      -> folly::SemiFuture<
                                                          folly::Unit> {
            // TODO: should use folly::SemiFuture::delayed(std::chrono::milliseconds(load_delay_ms_)),
            // but this requires folly::Init, thus using std::this_thread::sleep_for for now.
            std::this_thread::sleep_for(
                std::chrono::milliseconds(load_delay_ms_));
            std::unique_ptr<folly::SharedPromise<folly::Unit>>
                batch_load_promise_ptr{nullptr};
            std::vector<cid_t> copy_load_queue;
            BitsetType bitset(cells_.size());

            {
                auto load_queue = load_queue_.wlock();
                AssertInfo(!load_queue->empty(), "Load queue is empty");
                copy_load_queue.assign(load_queue->begin(), load_queue->end());
                load_queue->clear();
                batch_load_promise_ptr = std::move(batch_load_promise_);
            }
            for (auto cid : copy_load_queue) {
                bitset[cid] = true;
            }
            // after unlock, other load requests can be scheduled.
            try {
                auto results = translator_->get_cells(copy_load_queue);
                for (auto& result : results) {
                    cells_[result.first].set_cell(std::move(result.second), bitset[result.first]);
                }
                batch_load_promise_ptr->setValue(folly::Unit());
            } catch (std::exception& e) {
                LOG_ERROR(
                    fmt::format("CacheSlot {}: Error loading cells, reason: {}",
                                translator_->key(),
                                e.what()));
                batch_load_promise_ptr->setException(e);
                return folly::makeSemiFuture<folly::Unit>(e);
            }
            return folly::Unit();
        });
    }

    struct CacheCell : internal::ListNode {
     public:
        CacheCell() = default;
        CacheCell(CacheSlot<CellT>* slot, cid_t cid, size_t size)
            : internal::ListNode(slot->em_->dlist()),
              slot_(slot),
              cid_(cid),
              size_(size) {
        }
        ~CacheCell() {
            if (state_ == State::LOADING) {
                LOG_ERROR("CacheSlot {} Cell id {} destroyed while loading",
                          slot_->translator_->key(),
                          cid_);
            }
        }

        CellT*
        cell() {
            return cell_.get();
        }

        // Be careful that even though only a single thread can request loading a cell,
        // it is still possible that multiple threads call set_cell() concurrently.
        // For example, 2 RunLoad() calls tries to download cell 4 and 6, and both decided
        // to also download cell 5, if they finished at the same time, they will call set_cell()
        // of cell 5 concurrently.
        void
        set_cell(std::unique_ptr<CellT> cell, bool requesting_thread) {
            // translator may return more than requested, thus we need insert those
            // extra cells into the dlist and mark them as LOADED.
            auto shared_cell = std::make_shared<std::unique_ptr<CellT>>(std::move(cell));
            mark_loaded([this, shared_cell]() {
                cell_ = std::move(*shared_cell);
            }, requesting_thread);
        }

     protected:
        folly::SemiFuture<folly::Unit>
        load() override {
            auto load_queue = slot_->load_queue_.wlock();
            bool is_first = load_queue->empty();
            load_queue->push_back(cid_);
            if (!is_first) {
                return slot_->batch_load_promise_->getSemiFuture();
            }
            slot_->batch_load_promise_ =
                std::make_unique<folly::SharedPromise<folly::Unit>>();
            return slot_->RunLoad();
        }
        size_t
        size() override {
            return size_;
        }
        void
        unload() override {
            cell_ = nullptr;
        }
        const std::string&
        key() const override {
            return slot_->translator_->key();
        }
        const cid_t&
        cid() const override {
            return cid_;
        }

     private:
        CacheSlot<CellT>* slot_{nullptr};
        cid_t cid_{0};
        std::unique_ptr<CellT> cell_{nullptr};
        size_t size_{0};
    };
    // Each CacheCell's cid_t is its index in vector
    // Once initialized, cells_ should never be resized.
    std::vector<CacheCell> cells_;

    const std::unique_ptr<Translator<CellT>> translator_;

    // The first thread that adds a cid to load_queue_ will schedule a load in load_delay_ms_ and create a
    // batch_load_promise_. Access of batch_load_promise_ must be protected by wlock of load_queue_.
    folly::Synchronized<std::vector<cid_t>> load_queue_;
    const size_t load_delay_ms_;
    // Fulfilled when a batch load is finished.
    std::unique_ptr<folly::SharedPromise<folly::Unit>> batch_load_promise_{
        nullptr};

    // - CacheSlot is registered when created, and unregistered when destroyed.
    // - Each CellT is pinned when attempting to be loaded(no matter already loaded or not), and
    //   unpinned when: 1. load failed; 2. load succeeded and CellAccessor destroyed.
    // - Each CellT is marked as inserted when load succeeds.
    EvictionManager* em_;
};

// - A thin wrapper for accessing cells in a CacheSlot, does not own any logic of loading/eviction, etc.
// - When this class is created, the cells are loaded and pinned. Destroying this class will unpin the cells.
// - Accessing cells through this class does not incur any lock overhead.
// - Accessing cells that are not pinned by this CellAccessor is undefined behavior.
template <typename CellT>
class CellAccessor {
 public:
    CellAccessor(CacheSlot<CellT>* slot,
                 std::vector<internal::ListNode::Pin> pins)
        : slot_(slot), pins_(std::move(pins)) {
    }

    CellT*
    get_cell_of(uid_t uid) {
        auto cid = slot_->cell_id_of(uid);
        return slot_->cells_[cid].cell();
    }

 private:
    std::vector<internal::ListNode::Pin> pins_;
    CacheSlot<CellT>* slot_;
};
}  // namespace milvus::cachinglayer
