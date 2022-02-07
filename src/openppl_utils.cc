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

#include "openppl_utils.h"

namespace triton { namespace backend { namespace openppl {

ppl::common::RetCode ReadFileContent(const char* fname, string* buf) {
    ifstream ifile;

    ifile.open(fname, ios_base::in);
    if (!ifile.is_open()) {
        LOG(ERROR) << "open file[" << fname << "] failed.";
        return ppl::common::RC_NOT_FOUND;
    }

    stringstream ss;
    ss << ifile.rdbuf();
    *buf = ss.str();

    ifile.close();
    return ppl::common::RC_SUCCESS;
}

const char* MemMem(const char* haystack, unsigned int haystack_len, const char* needle,
                          unsigned int needle_len) {
    if (!haystack || haystack_len == 0 || !needle || needle_len == 0) {
        return nullptr;
    }

    for (auto h = haystack; haystack_len >= needle_len; ++h, --haystack_len) {
        if (memcmp(h, needle, needle_len) == 0) {
            return h;
        }
    }
    return nullptr;
}

void SplitString(const char* str, unsigned int len, const char* delim, unsigned int delim_len,
                        const function<bool(const char* s, unsigned int l)>& f) {
    const char* end = str + len;

    while (str < end) {
        auto cursor = MemMem(str, len, delim, delim_len);
        if (!cursor) {
            f(str, end - str);
            return;
        }

        if (!f(str, cursor - str)) {
            return;
        }

        cursor += delim_len;
        str = cursor;
        len = end - cursor;
    }

    f("", 0); // the last empty field
}

bool ParseInputShapes(const string& shape_str, vector<vector<int64_t>>* input_shapes) {
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


TRITONSERVER_DataType ConvertFromOpenPPLDataType(
    ppl::common::datatype_t data_type) {
  switch (data_type) {
    case ppl::common::DATATYPE_UINT8:
		return TRITONSERVER_TYPE_UINT8;
    case ppl::common::DATATYPE_UINT16:
		return TRITONSERVER_TYPE_UINT16;
    case ppl::common::DATATYPE_UINT32:
		return TRITONSERVER_TYPE_UINT32;
    case ppl::common::DATATYPE_UINT64:
		return TRITONSERVER_TYPE_UINT64;
    case ppl::common::DATATYPE_INT8:
		return TRITONSERVER_TYPE_INT8;
    case ppl::common::DATATYPE_INT16:
		return TRITONSERVER_TYPE_INT16;
    case ppl::common::DATATYPE_INT32:
		return TRITONSERVER_TYPE_INT32;
    case ppl::common::DATATYPE_INT64:
		return TRITONSERVER_TYPE_INT64;
    case ppl::common::DATATYPE_FLOAT16:
		return TRITONSERVER_TYPE_FP16;
    case ppl::common::DATATYPE_FLOAT32:
		return TRITONSERVER_TYPE_FP32;
    case ppl::common::DATATYPE_FLOAT64:
		return TRITONSERVER_TYPE_FP64;
    case ppl::common::DATATYPE_BOOL:
		return TRITONSERVER_TYPE_BOOL;
    default:
      break;
  }

  return TRITONSERVER_TYPE_INVALID;
}

ppl::common::datatype_t ConvertToOpenPPLDataType(
    TRITONSERVER_DataType data_type) {
  switch (data_type) {
    case TRITONSERVER_TYPE_UINT8:
      return ppl::common::DATATYPE_UINT8;
    case TRITONSERVER_TYPE_UINT16:
      return ppl::common::DATATYPE_UINT16;
    case TRITONSERVER_TYPE_UINT32:
      return ppl::common::DATATYPE_UINT32;
    case TRITONSERVER_TYPE_UINT64:
      return ppl::common::DATATYPE_UINT64;
    case TRITONSERVER_TYPE_INT8:
      return ppl::common::DATATYPE_INT8;
    case TRITONSERVER_TYPE_INT16:
      return ppl::common::DATATYPE_INT16;
    case TRITONSERVER_TYPE_INT32:
      return ppl::common::DATATYPE_INT32;
    case TRITONSERVER_TYPE_INT64:
      return ppl::common::DATATYPE_INT64;
    case TRITONSERVER_TYPE_FP16:
      return ppl::common::DATATYPE_FLOAT16;
    case TRITONSERVER_TYPE_FP32:
      return ppl::common::DATATYPE_FLOAT32;
    case TRITONSERVER_TYPE_FP64:
      return ppl::common::DATATYPE_FLOAT64;
    case TRITONSERVER_TYPE_BOOL:
      return ppl::common::DATATYPE_BOOL;
    default:
      break;
  }

  return ppl::common::DATATYPE_UNKNOWN;
}

}}}  // namespace triton::backend::openppl
