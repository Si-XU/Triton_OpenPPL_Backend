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
		return TRITONSERVER_TYPE_FLOAT16;
    case ppl::common::DATATYPE_FLOAT32:
		return TRITONSERVER_TYPE_FLOAT32;
    case ppl::common::DATATYPE_FLOAT64:
		return TRITONSERVER_TYPE_FLOAT64;
    case ppl::common::DATATYPE_BOOL:
		return TRITONSERVER_TYPE_BOOL;
    default:
      break;
  }

  return ppl::common::DATATYPE_UNKNOWN;
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
