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

#include "install/include/ppl/nn/models/onnx/onnx_runtime_builder_factory.h"
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

using namespace std;

namespace triton { namespace backend { namespace openppl {

extern const unique_ptr<ppl::nn::Runtime> runtime;

#define RETURN_IF_ORT_ERROR(S, message)                                           \
  do {                                                                            \
    auto status__ = (S);                                                          \
    if (status__ != ppl::common::RC_SUCCESS) {                                    \
      return TRITONSERVER_ErrorNew(                                               \
          TRITONSERVER_ERROR_INTERNAL, (Message + " " + GetRetCodeStr(status)));  \
    }                                                                             \
  } while (false)

static bool ParseInputShapes(const string& shape_str, vector<vector<int64_t>>* input_shapes) {
    bool ok = true;

    vector<string> input_shape_list;
    SplitString(shape_str.data(), shape_str.size(), ",", 1,
                [&ok, &input_shape_list](const char* s, unsigned int l) -> bool {
                    if (l > 0) {
                        input_shape_list.emplace_back(s, l);
                        return true;
                    }
                    LOG(ERROR) << "empty shape in option '--input-shapes'";
                    ok = false;
                    return false;
                });
    if (!ok) {
        return false;
    }

    for (auto x = input_shape_list.begin(); x != input_shape_list.end(); ++x) {
        ok = true;
        vector<int64_t> shape;
        SplitString(x->data(), x->size(), "_", 1, [&ok, &shape](const char* s, unsigned int l) -> bool {
            if (l > 0) {
                int64_t dim = atol(string(s, l).c_str());
                shape.push_back(dim);
                return true;
            }
            LOG(ERROR) << "illegal dim format.";
            ok = false;
            return false;
        });
        if (!ok) {
            return false;
        }

        input_shapes->push_back(shape);
    }

    return true;
}

}}}  // namespace triton::backend::openppl
