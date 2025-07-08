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

#include <gtest/gtest.h>
#include "storage/MmapManager.h"

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

#include "folly/init/Init.h"
#include "test_utils/Constants.h"
#include "storage/LocalChunkManagerSingleton.h"
#include "storage/RemoteChunkManagerSingleton.h"
#include "test_utils/storage_test_utils.h"
/*
checking register function of mmap chunk manager
*/
TEST(MmapChunkManager, Register) {
    auto mcm =
        milvus::storage::MmapManager::GetInstance().GetMmapChunkManager();
    auto get_descriptor =
        [](int64_t seg_id,
           SegmentType seg_type) -> milvus::storage::MmapChunkDescriptorPtr {
        return std::shared_ptr<milvus::storage::MmapChunkDescriptor>(
            new milvus::storage::MmapChunkDescriptor({seg_id, seg_type}));
    };
    int64_t segment_id = 0x0000456789ABCDEF;
    int64_t flow_segment_id = 0x8000456789ABCDEF;
    mcm->Register(get_descriptor(segment_id, SegmentType::Growing));
    ASSERT_TRUE(
        mcm->HasRegister(get_descriptor(segment_id, SegmentType::Growing)));
    ASSERT_FALSE(
        mcm->HasRegister(get_descriptor(segment_id, SegmentType::Sealed)));
    mcm->Register(get_descriptor(segment_id, SegmentType::Sealed));
    ASSERT_FALSE(mcm->HasRegister(
        get_descriptor(flow_segment_id, SegmentType::Growing)));
    ASSERT_FALSE(
        mcm->HasRegister(get_descriptor(flow_segment_id, SegmentType::Sealed)));

    mcm->UnRegister(get_descriptor(segment_id, SegmentType::Sealed));
    ASSERT_TRUE(
        mcm->HasRegister(get_descriptor(segment_id, SegmentType::Growing)));
    ASSERT_FALSE(
        mcm->HasRegister(get_descriptor(segment_id, SegmentType::Sealed)));
    mcm->UnRegister(get_descriptor(segment_id, SegmentType::Growing));
}

using namespace std;
using namespace milvus;
using namespace milvus::storage;
using namespace knowhere;
namespace fs = std::filesystem;

class DiskAnnFileManagerLoadTest : public testing::Test {
 public:
    DiskAnnFileManagerLoadTest() {
    }
    ~DiskAnnFileManagerLoadTest() {
    }

    virtual void
    SetUp() {
        cm_ = milvus::storage::CreateChunkManager(get_default_local_storage_config());
    }

 protected:
    ChunkManagerPtr cm_;
};

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

    // collection_id: 1, partition_id: 2, segment_id: 3
    // field_id: 100, index_build_id: 1000, index_version: 1
    FieldDataMeta filed_data_meta = {1, 2, 3, 100};
    IndexMeta index_meta = {3, 100, 1000, 1, "index"};

    int64_t slice_size = milvus::FILE_SLICE_SIZE;
    auto diskAnnFileManager = std::make_shared<DiskFileManagerImpl>(
        storage::FileManagerContext(filed_data_meta, index_meta, cm_));

    std::vector<std::string> remote_files = getAllFiles(indexFilePath);
    diskAnnFileManager->CacheIndexToDisk(remote_files);
    auto local_files = diskAnnFileManager->GetLocalFilePaths();

    for (auto file : local_files) {
        std::cout << "local_file: " << file << std::endl;
    }

    std::this_thread::sleep_for(std::chrono::seconds(300));

    for (auto file : local_files) {
        cm_->Remove(file);
    }
}