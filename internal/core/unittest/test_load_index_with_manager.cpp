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

#include <boost/filesystem/operations.hpp>
#include <chrono>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>
#include <unistd.h>

#include "common/EasyAssert.h"
#include "common/FieldDataInterface.h"
#include "common/Slice.h"
#include "common/Common.h"
#include "common/Types.h"
#include "storage/ChunkManager.h"
#include "storage/DataCodec.h"
#include "storage/InsertData.h"
#include "storage/ThreadPool.h"
#include "storage/Types.h"
#include "storage/Util.h"
#include "storage/DiskFileManagerImpl.h"
#include "storage/LocalChunkManagerSingleton.h"



#include "test_utils/storage_test_utils.h"

using namespace std;
using namespace milvus;
using namespace milvus::storage;
using namespace knowhere;

class DiskAnnFileManagerLoadTest : public testing::Test {
 public:
    DiskAnnFileManagerLoadTest() {
    }
    ~DiskAnnFileManagerLoadTest() {
    }

    virtual void
    SetUp() {
        cm_ = storage::CreateChunkManager(get_default_local_storage_config());
    }

 protected:
    ChunkManagerPtr cm_;
};

namespace fs = std::filesystem;

std::vector<std::string> getAllFiles(const fs::path& directory) {
    std::vector<std::string> filePaths;

    // Iterate through directory and subdirectories
    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (fs::is_regular_file(entry.path())) {
            filePaths.push_back(entry.path().string());
        }
    }

    return filePaths;
}

TEST_F(DiskAnnFileManagerLoadTest, LoadWithManager) {
    auto lcm = LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    std::string indexFilePath = "/tmp/diskann/index_files/1000/index";
    auto exist = lcm->Exist(indexFilePath);
    EXPECT_EQ(exist, false);
    uint64_t index_size = 50 << 20;
    lcm->CreateFile(indexFilePath);
    std::vector<uint8_t> data(index_size);
    lcm->Write(indexFilePath, data.data(), index_size);

    // collection_id: 1, partition_id: 2, segment_id: 3
    // field_id: 100, index_build_id: 1000, index_version: 1
    FieldDataMeta filed_data_meta = {1, 2, 3, 100};
    IndexMeta index_meta = {3, 100, 1000, 1, "index"};

    int64_t slice_size = milvus::FILE_SLICE_SIZE;
    auto diskAnnFileManager = std::make_shared<DiskFileManagerImpl>(
        storage::FileManagerContext(filed_data_meta, index_meta, cm_));
    auto ok = diskAnnFileManager->AddFile(indexFilePath);
    EXPECT_EQ(ok, true);

    auto remote_files_to_size = diskAnnFileManager->GetRemotePathsToFileSize();
    auto num_slice = index_size / slice_size;
    EXPECT_EQ(remote_files_to_size.size(),
              index_size % slice_size == 0 ? num_slice : num_slice + 1);

    std::vector<std::string> remote_files;
    for (auto& file2size : remote_files_to_size) {
        std::cout << file2size.first << std::endl;
        remote_files.emplace_back(file2size.first);
    }
    diskAnnFileManager->CacheIndexToDisk(remote_files);
    auto local_files = diskAnnFileManager->GetLocalFilePaths();

    for (auto file : local_files) {
        cm_->Remove(file);
    }
}

TEST_F(DiskAnnFileManagerLoadTest, LoadRawData) {
    auto lcm = LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    uint64_t index_size = 50 << 20;

    // collection_id: 1, partition_id: 2, segment_id: 3
    // field_id: 100, index_build_id: 1000, index_version: 1
    FieldDataMeta filed_data_meta = {1, 2, 3, 100};
    IndexMeta index_meta = {3, 100, 1000, 1, "index"};

    int64_t slice_size = milvus::FILE_SLICE_SIZE;
    auto diskAnnFileManager = std::make_shared<DiskFileManagerImpl>(
        storage::FileManagerContext(filed_data_meta, index_meta, cm_));

    std::vector<std::string> remote_files = getAllFiles("/home/fox/Downloads/in01-0e3f674f63dfafd_101/101/");
    for (auto file : remote_files) {
        std::cout<<" file "<< file<<std::endl;
    }

    diskAnnFileManager->CacheRawDataToDisk<float>(remote_files);
    auto local_files = diskAnnFileManager->GetLocalFilePaths();

    for (auto file : local_files) {
        cm_->Remove(file);
    }
}
