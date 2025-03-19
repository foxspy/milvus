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
#include "cachinglayer/Translator.h"
#include "cachinglayer/Utils.h"
#include "common/Types.h"
#include "log/Log.h"

namespace milvus::cachinglayer {

template <typename CellT>
class CellAccessor;

// - The action of pinning cells is not started until the returned SemiFuture is scheduled on an executor.
// - `uids` must be valid until the returned future is ready.
// - once the future is scheduled, CacheSlot must live until both the future is ready and the returned CellAccessor is
//   destroyed.
// - If the user decides to destroy a CacheSlot, and there are still pending pinning/loading of cells, ~CacheSlot()
//   must not be called until the pending operations are finished.
template <typename CellT>
class CacheSlot {
 public:
    static_assert(
        std::is_same_v<size_t, decltype(std::declval<CellT>().DataByteSize())>,
        "CellT must have a DataByteSize() method that returns size_t");
    CacheSlot(std::unique_ptr<Translator<CellT>> translator,
              EvictionManager* eviction_manager)
        : translator_(std::move(translator)),
          cells_(translator_->num_cells()),
          em_(eviction_manager) {
        em_->register_slot(slot_id(), cells_.size());
    }

    // we use the address of a CacheSlot as its unique identifier, thus disallowing copy/move.
    CacheSlot(const CacheSlot&) = delete;
    CacheSlot&
    operator=(const CacheSlot&) = delete;
    CacheSlot(CacheSlot&&) = delete;
    CacheSlot&
    operator=(CacheSlot&&) = delete;

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
                    throw std::invalid_argument(fmt::format(
                        "CacheSlot {}: translator returned cell_id {} "
                        "for uid {} which is out of range",
                        translator_->key(),
                        cid,
                        uid));
                }
                if (!bitset[cid]) {
                    bitset[cid] = true;
                    involved_cids.push_back(cid);
                }
            }
            std::vector<folly::SemiFuture<folly::Unit>> futures;
            futures.reserve(involved_cids.size());
            for (auto cid : involved_cids) {
                futures.push_back(PinCell(cid));
            }
            return folly::collect(futures).deferValue(
                [this, involved_cids = std::move(involved_cids)](
                    auto&&) -> std::unique_ptr<CellAccessor<CellT>> {
                    return std::make_unique<CellAccessor<CellT>>(
                        this, std::move(involved_cids));
                });
        });
    }

    // CacheSlot is created/destroyed along with segment load/release.
    // CacheSlot must out live all returned CellAccessors.
    ~CacheSlot() {
        for (cid_t cid = 0; cid < cells_.size(); ++cid) {
            auto& cell_internal = cells_[cid];
            std::unique_lock<std::shared_mutex> lock(cell_internal.mutex_);
            if (cell_internal.state_ == CellState::LOADED) {
                auto key = GlobalCellKey(slot_id(), cid);
                em_->notify_cell_evicted(key);
                cell_internal.cell_ = nullptr;
                cell_internal.state_ = CellState::NOT_LOADED;
            } else if (cell_internal.state_ == CellState::LOADING) {
                LOG_ERROR(fmt::format(
                    "CacheSlot {} destroyed while cell {} is still loading",
                    translator_->key(),
                    cid));
            }
        }
        em_->unregister_slot(slot_id(), cells_.size());
    }

 private:
    friend class CellAccessor<CellT>;
    // called only by CellAccessor.
    void
    UnpinCells(std::vector<cid_t>&& cids) {
        for (auto cid : cids) {
            auto key = GlobalCellKey(slot_id(), cid);
            em_->notify_cell_unpinned(key);
        }
    }
    cid_t
    cell_id_of(uid_t uid) const {
        return translator_->cell_id_of(uid);
    }

    uint64_t
    slot_id() const {
        return (uint64_t)this;
    }

    // returns a future that will be ready when the cell is pinned.
    // if the cell is already pinned, the returned future will be ready immediately.
    folly::SemiFuture<folly::Unit>
    PinCell(cid_t cid) {
        return folly::makeSemiFuture().deferValue([this, cid](auto&&) {
            auto& cell_internal = cells_[cid];
            // must be called with lock acquired, and state must not be NOT_LOADED.
            auto read_op = [this,
                            &cell_internal,
                            cid]() -> folly::SemiFuture<folly::Unit> {
                if (cell_internal.state_ == CellState::LOADED) {
                    auto key = GlobalCellKey(slot_id(), cid);
                    em_->notify_cell_pinned(key);
                    return folly::Unit();
                }
                if (cell_internal.state_ == CellState::LOADING) {
                    return cell_internal.load_promise_->getSemiFuture()
                        .deferValue([this, cid](auto&&) {
                            auto key = GlobalCellKey(slot_id(), cid);
                            em_->notify_cell_pinned(key);
                            return folly::Unit();
                        });
                }
                // state_ is ERROR:
                throw cell_internal.error_;
            };
            {
                std::shared_lock<std::shared_mutex> lock(cell_internal.mutex_);
                if (cell_internal.state_ != CellState::NOT_LOADED) {
                    return read_op();
                }
            }
            std::unique_lock<std::shared_mutex> lock(cell_internal.mutex_);
            if (cell_internal.state_ != CellState::NOT_LOADED) {
                return read_op();
            }
            cell_internal.load_promise_ =
                std::make_unique<folly::SharedPromise<folly::Unit>>();
            cell_internal.state_ = CellState::LOADING;

            auto load_queue = load_queue_.wlock();
            bool is_first = load_queue->empty();
            load_queue->push_back(cid);
            if (!is_first) {
                return batch_load_promise_->getSemiFuture();
            }
            batch_load_promise_ =
                std::make_unique<folly::SharedPromise<folly::Unit>>();
            // pin happens in RunLoad().
            return RunLoad();
        });
    }

    // When called, RunLoad uses translator to load all NOT_LOADED cells in load_queue_.
    folly::SemiFuture<folly::Unit>
    RunLoad() {
        return folly::makeSemiFuture()
            // .delayed(std::chrono::milliseconds(10))
            .deferValue([this](auto&&) {
                std::unique_ptr<folly::SharedPromise<folly::Unit>>
                    batch_load_promise_ptr{nullptr};
                std::vector<cid_t> copy_load_queue;

                {
                    auto load_queue = load_queue_.wlock();
                    AssertInfo(!load_queue->empty(), "Load queue is empty");
                    copy_load_queue.assign(load_queue->begin(),
                                           load_queue->end());
                    load_queue->clear();
                    batch_load_promise_ptr = std::move(batch_load_promise_);
                }
                try {
                    auto results = translator_->get_cells(copy_load_queue);
                    for (auto& result : results) {
                        auto cid = result.first;
                        auto& cell_internal = cells_[cid];
                        std::unique_lock<std::shared_mutex> lock(
                            cell_internal.mutex_);
                        cell_internal.cell_ = std::move(result.second);
                        cell_internal.state_ = CellState::LOADED;
                        auto key = GlobalCellKey(slot_id(), cid);
                        em_->notify_cell_inserted(
                            key, cell_internal.cell_->DataByteSize());
                        // translator may return more cells than requested, those are not pinned so they don't have a
                        // load_promise_.
                        if (cell_internal.load_promise_) {
                            cell_internal.load_promise_->setValue(
                                folly::Unit());
                            cell_internal.load_promise_ = nullptr;
                            em_->notify_cell_pinned(key);
                        }
                    }
                    batch_load_promise_ptr->setValue(folly::Unit());
                } catch (std::exception& e) {
                    LOG_ERROR(fmt::format(
                        "CacheSlot {}: Error loading cells, reason: {}",
                        translator_->key(),
                        e.what()));
                    for (auto cid : copy_load_queue) {
                        auto& cell_internal = cells_[cid];
                        std::unique_lock<std::shared_mutex> lock(
                            cell_internal.mutex_);
                        cell_internal.state_ = CellState::ERROR;
                        if (cell_internal.load_promise_) {
                            cell_internal.load_promise_->setException(e);
                            cell_internal.load_promise_ = nullptr;
                        }
                        cell_internal.error_ = e;
                    }
                    batch_load_promise_ptr->setException(e);
                    throw e;
                }
                return folly::Unit();
            });
    }

    // NOT_LOADED ---> LOADING ---> ERROR
    //      ^            |
    //      |            v
    //      |------- LOADED
    enum class CellState {
        NOT_LOADED,  // cell is not loaded.
        LOADING,     // cell is being loaded.
        LOADED,      // cell is loaded.
        ERROR,       // cell loading failed.
    };
    struct CellInternal {
        CellState state_{CellState::NOT_LOADED};
        std::unique_ptr<CellT> cell_{nullptr};
        // fulfilled when the cell is loaded.
        std::unique_ptr<folly::SharedPromise<folly::Unit>> load_promise_{
            nullptr};
        std::shared_mutex mutex_;
        std::exception error_;
    };
    std::unique_ptr<Translator<CellT>> translator_;
    // Each CacheCell's cid_t is its index in vector
    // Once initialized, cells_ should never be resized.
    std::vector<CellInternal> cells_;

    // the first thread that adds a cid to load_queue_ will schedule a load in 10ms and create a batch_load_promise_.
    // access of batch_load_promise_ must be protected by wlock of load_queue_.
    folly::Synchronized<std::vector<cid_t>> load_queue_;
    // fulfilled when a batch load is finished.
    std::unique_ptr<folly::SharedPromise<folly::Unit>> batch_load_promise_{
        nullptr};

    EvictionManager* em_;
};

// - A thin wrapper for accessing cells in a CacheSlot, does not own any logic of loading/eviction, etc.
// - When this class is created, the cells are loaded and pinned. Destroying this class will unpin the cells.
// - Accessing cells through this class does not incur any lock overhead.
template <typename CellT>
class CellAccessor {
 public:
    CellAccessor(CacheSlot<CellT>* slot, std::vector<cid_t> cids)
        : slot_(slot), cids_(std::move(cids)) {
    }

    CellT*
    get_cell_of(uid_t uid) {
        auto cid = slot_->cell_id_of(uid);
        return slot_->cells_[cid].cell_.get();
    }

    ~CellAccessor() {
        slot_->UnpinCells(std::move(cids_));
    }

 private:
    std::vector<cid_t> cids_;
    CacheSlot<CellT>* slot_;
    ;
};

}  // namespace milvus::cachinglayer
