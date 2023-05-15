// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <memory>
#include <sys/mman.h>
#include <cstring>
#include <stdexcept>

namespace milvus {
using namespace std;

template<typename T>
class mmap_alloctor :  public std::allocator<T> {
    T* allocate(size_t n, const void *hint=0) {
        T *addr = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error("mmap allocate failed.");
        }
        return addr;
    }

    void deallocate(T* p, size_t n) {
        if (p != NULL) {
            munmap(p, n);
        }
    }
};
}

