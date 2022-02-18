// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "ppl/nn/models/onnx/onnx_runtime_builder_factory.h"
#include "ppl/nn/engines/cuda/cuda_engine_options.h"
#include "ppl/nn/engines/cuda/cuda_options.h"
#include "ppl/nn/engines/cuda/engine_factory.h"
#include "ppl/nn/common/logger.h"
#include <set>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

using namespace std;

namespace triton { namespace backend { namespace openppl {

#define RESPOND_ALL_AND_RETURN_IF_ERROR(                          \
    REQUESTS, REQUEST_COUNT, RESPONSES, S)                        \
  do {                                                            \
    TRITONSERVER_Error* raarie_err__ = (S);                       \
    if (raarie_err__ != nullptr) {                                \
      for (uint32_t r = 0; r < REQUEST_COUNT; ++r) {              \
        TRITONBACKEND_Response* response = (*RESPONSES)[r];       \
        if (response != nullptr) {                                \
          LOG_IF_ERROR(                                           \
              TRITONBACKEND_ResponseSend(                         \
                  response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                  raarie_err__),                                  \
              "failed to send OpenPPL backend response");     \
          response = nullptr;                                     \
        }                                                         \
        LOG_IF_ERROR(                                             \
            TRITONBACKEND_RequestRelease(                         \
                REQUESTS[r], TRITONSERVER_REQUEST_RELEASE_ALL),   \
            "failed releasing request");                          \
        REQUESTS[r] = nullptr;                                    \
      }                                                           \
      TRITONSERVER_ErrorDelete(raarie_err__);                     \
      return;                                                     \
    }                                                             \
  } while (false) 

extern const unique_ptr<ppl::nn::Runtime> runtime;

ppl::common::RetCode ReadFileContent(const char* fname, string* buf);

const char* MemMem(const char* haystack, unsigned int haystack_len, const char* needle, unsigned int needle_len);

void SplitString(const char* str, unsigned int len, const char* delim, unsigned int delim_len,
                        const function<bool(const char* s, unsigned int l)>& f);

bool ParseInputShapes(const string& shape_str, vector<vector<int64_t>>* input_shapes);

TRITONSERVER_DataType ConvertFromOpenPPLDataType(ppl::common::datatype_t data_type);

ppl::common::datatype_t ConvertToOpenPPLDataType(TRITONSERVER_DataType data_type);

}}}  // namespace triton::backend::openppl
