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

#include "cachinglayer/Translator.h"
#include "cachinglayer/Utils.h"
#include "common/Types.h"
#include "common/FieldMeta.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index.h"

namespace milvus::segcore::storagev1translator {

class DiskVecIndexTranslator : public milvus::cachinglayer::Translator<
                                   knowhere::Index<knowhere::IndexNode>> {};

}  // namespace milvus::segcore::storagev1translator