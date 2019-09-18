// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>

#include "config/ConfigMgr.h"
#include "server/ServerConfig.h"
#include "utils/CommonUtil.h"
#include "utils/ValidationUtil.h"

using namespace zilliz::milvus;

namespace {

static const char* CONFIG_FILE_PATH = "./milvus/conf/server_config.yaml";
static const char* LOG_FILE_PATH = "./milvus/conf/log_config.conf";

static constexpr uint64_t KB = 1024;
static constexpr uint64_t MB = KB*1024;
static constexpr uint64_t GB = MB*1024;

}

TEST(ConfigTest, CONFIG_TEST) {
    server::ConfigMgr* config_mgr = server::ConfigMgr::GetInstance();

    ErrorCode err = config_mgr->LoadConfigFile("");
    ASSERT_EQ(err, SERVER_UNEXPECTED_ERROR);

    err = config_mgr->LoadConfigFile(LOG_FILE_PATH);
    ASSERT_EQ(err, SERVER_UNEXPECTED_ERROR);

    err = config_mgr->LoadConfigFile(CONFIG_FILE_PATH);
    ASSERT_EQ(err, SERVER_SUCCESS);

    config_mgr->Print();

    server::ConfigNode& root_config = config_mgr->GetRootNode();
    server::ConfigNode& server_config = root_config.GetChild("server_config");
    server::ConfigNode& db_config = root_config.GetChild("db_config");
    server::ConfigNode& metric_config = root_config.GetChild("metric_config");
    server::ConfigNode& cache_config = root_config.GetChild("cache_config");
    server::ConfigNode invalid_config = root_config.GetChild("invalid_config");
    auto valus = invalid_config.GetSequence("not_exist");
    float ff = invalid_config.GetFloatValue("not_exist", 3.0);
    ASSERT_EQ(ff, 3.0);

    std::string address = server_config.GetValue("address");
    ASSERT_TRUE(!address.empty());
    int64_t port = server_config.GetInt64Value("port");
    ASSERT_TRUE(port != 0);

    server_config.SetValue("test", "2.5");
    double test = server_config.GetDoubleValue("test");
    ASSERT_EQ(test, 2.5);

    server::ConfigNode fake;
    server_config.AddChild("fake", fake);
    fake = server_config.GetChild("fake");
    server::ConfigNodeArr arr;
    server_config.GetChildren(arr);
    ASSERT_EQ(arr.size(), 1UL);

    server_config.ClearChildren();
    auto children = server_config.GetChildren();
    ASSERT_TRUE(children.empty());

    server_config.ClearConfig();
    auto configs = server_config.GetConfig();
    ASSERT_TRUE(configs.empty());

    server_config.AddSequenceItem("seq", "aaa");
    server_config.AddSequenceItem("seq", "bbb");
    auto seq = server_config.GetSequence("seq");
    ASSERT_EQ(seq.size(), 2UL);

    server::ConfigNode combine;
    combine.Combine(server_config);

    combine.PrintAll();
    std::string all = combine.DumpString();
    ASSERT_TRUE(!all.empty());

    server_config.ClearSequences();
    auto seqs = server_config.GetSequences();
    ASSERT_TRUE(seqs.empty());

    const server::ConfigNode const_node = root_config.GetChild("cache_config");
    float flt = const_node.GetFloatValue("cpu_cache_capacity");
    ASSERT_GT(flt, 0.0);
}

TEST(ConfigTest, SERVER_CONFIG_TEST) {
    server::ServerConfig& config = server::ServerConfig::GetInstance();
    auto status = config.LoadConfigFile(CONFIG_FILE_PATH);
    ASSERT_TRUE(status.ok());

    status = server::ServerConfig::GetInstance().ValidateConfig();
    ASSERT_TRUE(status.ok());

    const server::ServerConfig& config_const = config;
    server::ConfigNode node1 = config_const.GetConfig("server_config");
    server::ConfigNode& node2 = config.GetConfig("cache_config");
    node1.Combine(node2);

    int32_t cap = node1.GetInt32Value("cpu_cache_capacity");
    ASSERT_GT(cap, 0);

    node1.SetValue("bool", "true");
    bool bt = node1.GetBoolValue("bool");
    ASSERT_TRUE(bt);

    config.PrintAll();

    unsigned long total_mem = 0, free_mem = 0;
    server::CommonUtil::GetSystemMemInfo(total_mem, free_mem);

    size_t gpu_mem = 0;
    server::ValidationUtil::GetGpuMemory(0, gpu_mem);

    server::ConfigNode& server_config = config.GetConfig("server_config");
    server::ConfigNode& db_config = config.GetConfig("db_config");
    server::ConfigNode& cache_config = config.GetConfig(server::CONFIG_CACHE);
    cache_config.SetValue(server::CACHE_FREE_PERCENT, "2.0");
    status = config.ValidateConfig();
    ASSERT_FALSE(status.ok());

    size_t cache_cap = 16;
    size_t insert_buffer_size = (total_mem - cache_cap*GB + 1*GB)/GB;
    db_config.SetValue(server::CONFIG_DB_INSERT_BUFFER_SIZE, std::to_string(insert_buffer_size));
    cache_config.SetValue(server::CONFIG_CPU_CACHE_CAPACITY, std::to_string(cache_cap));
    status = config.ValidateConfig();
    ASSERT_FALSE(status.ok());

    cache_cap = total_mem/GB + 2;
    cache_config.SetValue(server::CONFIG_CPU_CACHE_CAPACITY, std::to_string(cache_cap));
    status = config.ValidateConfig();
    ASSERT_FALSE(status.ok());

    insert_buffer_size = total_mem/GB + 2;
    db_config.SetValue(server::CONFIG_DB_INSERT_BUFFER_SIZE, std::to_string(insert_buffer_size));
    status = config.ValidateConfig();
    ASSERT_FALSE(status.ok());
}