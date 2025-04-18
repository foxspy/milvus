#-------------------------------------------------------------------------------
# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.
#-------------------------------------------------------------------------------

if (DEFINED ENV{MILVUS_MIMALLOC_URL})
    set(MIMALLOC_SOURCE_URL "$ENV{MILVUS_MIMALLOC_URL}")
else ()
    set(MIMALLOC_SOURCE_URL "https://github.com/microsoft/mimalloc/archive/refs/tags/v2.2.3.tar.gz")
endif ()

message(STATUS "Building mimalloc from source in Debug mode with MI_GUARDED=ON")

set(MIMALLOC_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(MIMALLOC_LIB_DIR "${MIMALLOC_PREFIX}/lib")
set(MIMALLOC_STATIC_LIB "${MIMALLOC_LIB_DIR}/libmimalloc${CMAKE_STATIC_LIBRARY_SUFFIX}")

# 保存当前的 CMAKE_BUILD_TYPE
set(SAVED_BUILD_TYPE ${CMAKE_BUILD_TYPE})

ExternalProject_Add(
    mimalloc_ep
    PREFIX ${CMAKE_BINARY_DIR}/3rdparty_download/mimalloc-subbuild
    URL ${MIMALLOC_SOURCE_URL}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${MIMALLOC_PREFIX}
        -DCMAKE_BUILD_TYPE=Debug  # 强制使用 Debug 模式
        -DMI_BUILD_SHARED=ON
        -DMI_BUILD_STATIC=OFF
        -DMI_BUILD_TESTS=OFF
        -DMI_BUILD_OBJECT=OFF
        -DMI_DEBUG_FULL=ON        # 启用完整的调试功能
        -DMI_SECURE=ON           # 启用安全检查
        -DMI_SECURE_FULL=ON      # 启用完整的安全检查
        -DMI_GUARDED=ON          # 启用 guard pages 保护
    BUILD_BYPRODUCTS "${MIMALLOC_STATIC_LIB}"
)

add_library(mimalloc SHARED IMPORTED)
set_target_properties(mimalloc
    PROPERTIES 
    IMPORTED_LOCATION "${MIMALLOC_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libmimalloc-secure-debug.so"
    INTERFACE_INCLUDE_DIRECTORIES "${MIMALLOC_PREFIX}/include")
add_dependencies(mimalloc mimalloc_ep)

# 恢复原来的 CMAKE_BUILD_TYPE
set(CMAKE_BUILD_TYPE ${SAVED_BUILD_TYPE})
