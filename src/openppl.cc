// Copyright 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdint.h>
#include <mutex>
#include <vector>
#include <string>
#include <map>
#include "openppl_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

using namespace ppl::nn;
using namespace ppl::common;
using namespace std;

//
// Openppl Backend that implements the TRITONBACKEND API.
//
namespace triton { namespace backend { namespace openppl {

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  AutoCompleteConfig()
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** model_state)
{
  try {
    *model_state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }
  
  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*model_state)->AutoCompleteConfig());

    triton::common::TritonJson::WriteBuffer json_buffer;
    (*model_state)->ModelConfig().Write(&json_buffer);

    TRITONSERVER_Message* message;
    RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(
        &message, json_buffer.Base(), json_buffer.Size()));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(
        triton_model, 1 /* config_version */, message));
  }
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{
}


//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  RetCode RegisterCudaEngine(vector<unique_ptr<Engine>>* engines);
  TRITONSERVER_Error* SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      std::vector<const char*>* input_names);
  TRITONSERVER_Error* ReadOutputTensors(
      size_t total_batch_size, const std::vector<const char*>& output_names,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  ModelState* model_state_;
  std::string model_path_;

  // Store input and output openppl tensers for all requests
  uint32_t input_count_ = 0;
  uint32_t output_count_ = 0;
  std::vector<std::map<std::string, Tensor*>> input_tensors_;
  std::vector<std::map<std::string, Tensor*>> output_tensors_;
  unique_ptr<Runtime> runtime_;

};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state)
{
  vector<unique_ptr<Engine>> engines;
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    try {
      RegisterCudaEngine(engines);
    }
    catch (const BackendModelInstanceException& ex) {
      throw TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, ("Register cuda engine fail."));
    }
  } else { // Only support GPU right now
    throw TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, ("Unsupport engine type."));
  }

  triton::common::TritonJson::Value param;
  if (!sequence_batching.Find("params", &param)) {
    throw TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, ("Read params fail."));
  }

  string g_flag_onnx_model;
  param.MemberAsString("onnx-model", &g_flag_onnx_model));
  if (!g_flag_onnx_model.empty()) {
      vector<Engine*> engine_ptrs(engines.size());
      for (uint32_t i = 0; i < engines.size(); ++i) {
          engine_ptrs[i] = engines[i].get();
      }
      auto builder = unique_ptr<RuntimeBuilder>(
          OnnxRuntimeBuilderFactory::Create(g_flag_onnx_model.c_str(), engine_ptrs.data(), engine_ptrs.size()));
      if (!builder) {
          throw TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, ("Read params fail."));
      }

      runtime_.reset(builder->CreateRuntime());
  }

  if (!runtime_) {
    throw TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL, ("Reset runtime fail."));
  }
}

TRITONSERVER_Error*
RegisterCudaEngine(vector<unique_ptr<Engine>>* engines) {
  CudaEngineOptions options;
  options.device_id = g_flag_device_id;

  triton::common::TritonJson::Value param;
  if (sequence_batching.Find("params", &param)) {
    string policy;
    param.MemberAsString("mm-policy", &policy));
    if (policy == "perf") {
        options.mm_policy = CUDA_MM_BEST_FIT;
    } else if (policy == "mem") {
        options.mm_policy = CUDA_MM_COMPACT;
    }
  }

  auto cuda_engine = CudaEngineFactory::Create(options);
  if (!cuda_engine) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, ("invalid cuda engine"));
  }
  cuda_engine->Configure(ppl::nn::CUDA_CONF_USE_DEFAULT_ALGORITHMS, g_flag_quick_select);

  string g_flag_kernel_type;
  param.MemberAsString("kerne-type", &g_flag_kernel_type));
  if (!g_flag_kernel_type.empty()) {
      string kernel_type_str(g_flag_kernel_type);
      std::transform(g_flag_kernel_type.begin(), g_flag_kernel_type.end(),
                      kernel_type_str.begin(), ::toupper);

      datatype_t kernel_type = DATATYPE_UNKNOWN;
      for (datatype_t i = DATATYPE_UNKNOWN; i < DATATYPE_MAX; i++) {
          if (GetDataTypeStr(i) == kernel_type_str) {
              kernel_type = i;
              break;
          }
      }

      if (kernel_type != DATATYPE_UNKNOWN) {
        cuda_engine->Configure(ppl::nn::CUDA_CONF_SET_KERNEL_TYPE, kernel_type);
      } else {
        return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, ("invalid kernel type[" + g_flag_kernel_type + "]. valid values: int8/16/32/64,float16/32."));
      }
  }

  string g_flag_quant_file;
  param.MemberAsString("quant-file", &g_flag_quant_file));
  if (!g_flag_quant_file.empty()) {
      string file_content;
      auto status = ReadFileContent(g_flag_quant_file.c_str(), &file_content);
      if (status != RC_SUCCESS) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL, ("invalid quant file path[" + g_flag_kernel_type + "]."));
      }
      cuda_engine->Configure(ppl::nn::CUDA_CONF_SET_QUANT_INFO, file_content.c_str());
  }

  string g_flag_export_algo_file;
  param.MemberAsString("export-algo-file", &g_flag_export_algo_file));
  if (!g_flag_export_algo_file.empty()) {
      cuda_engine->Configure(ppl::nn::CUDA_CONF_EXPORT_ALGORITHMS, g_flag_export_algo_file.c_str());
  }

  string g_flag_import_algo_file;
  param.MemberAsString("import-algo-file", &g_flag_import_algo_file));  
  if (!g_flag_import_algo_file.empty()) {
      // import and export from the same file
      if (g_flag_import_algo_file == g_flag_export_algo_file) {
          // try to create this file first
          ofstream ofs(g_flag_export_algo_file, ios_base::app);
          if (!ofs.is_open()) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INTERNAL, ("invalid algo file path[" + g_flag_import_algo_file + "]."));
          }
          ofs.close();
      }
      cuda_engine->Configure(ppl::nn::CUDA_CONF_IMPORT_ALGORITHMS, g_flag_import_algo_file.c_str());
  }

  // pass input shapes to cuda engine for further optimizations
  string g_flag_input_shapes;
  param.MemberAsString("input-shapes", &g_flag_input_shapes));  
  if (!g_flag_input_shapes.empty()) {
      vector<vector<int64_t>> input_shapes;
      if (!ParseInputShapes(g_flag_input_shapes, &input_shapes)) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL, ("invalid input shapes[" + g_flag_input_shapes + "]."));
      }

      vector<utils::Array<int64_t>> dims(input_shapes.size());
      for (uint32_t i = 0; i < input_shapes.size(); ++i) {
          auto& arr = dims[i];
          arr.base = input_shapes[i].data();
          arr.size = input_shapes[i].size();
      }
      cuda_engine->Configure(ppl::nn::CUDA_CONF_SET_INPUT_DIMS, dims.data(), dims.size());
  }

  engines->emplace_back(unique_ptr<Engine>(cuda_engine));
  LOG(INFO) << "***** register CudaEngine *****";
  return nullptr;
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = model_state_->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to ONNX Runtime backend for '" + Name() +
                  "'")
                  .c_str()));
      return;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                Name() + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
    return;
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  bool all_response_failed = false;

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }


  std::vector<const char*> input_names;
  bool cuda_copy = false;
  BackendInputCollector collector(
      requests, request_count, &responses, model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedInput(), CudaStream(), nullptr, nullptr, 0,
      HostPolicyName().c_str());

  RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
      responses, request_count, all_response_failed,
      SetInputTensors(
          total_batch_size, requests, request_count, &responses, &collector,
          &input_names, &cuda_copy));

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(CudaStream());
  }
#endif

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  if (!all_response_failed) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        RunOneByOne(&responses, request_count));
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  if (!all_response_failed) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ReadOutputTensors(
            total_batch_size, requests, request_count, &responses));
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send onnxruntime backend response");
    }
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  if (!all_response_failed) {
    // Report the entire batch statistics.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportBatchStatistics(
            TritonModelInstance(), total_batch_size, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting batch request statistics");
  }
}

TRITONSERVER_Error*
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    bool* cuda_copy)
{
  const int max_batch_size = model_state_->MaxBatchSize();

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));
  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape, &input_dims_count,
        nullptr, nullptr));

    input_names->emplace_back(input_name);
    input_tensors_.emplace_back(nullptr);

    std::vector<int64_t> batchn_shape;
    // For a ragged input tensor, the tensor shape should be
    // the flatten shape of the whole batch
    if (StateForModel()->IsInputRagged(input_name)) {
      batchn_shape = std::vector<int64_t>{0};
      for (size_t idx = 0; idx < request_count; idx++) {
        TRITONBACKEND_Input* input;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_RequestInput(requests[idx], input_name, &input));
        const int64_t* input_shape;
        uint32_t input_dims_count;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]), TRITONBACKEND_InputProperties(
                                      input, nullptr, nullptr, &input_shape,
                                      &input_dims_count, nullptr, nullptr));

        batchn_shape[0] += GetElementCount(input_shape, input_dims_count);
      }
    }
    // The shape for the entire input batch, [total_batch_size, ...]
    else {
      batchn_shape =
          std::vector<int64_t>(input_shape, input_shape + input_dims_count);
      if (max_batch_size != 0) {
        batchn_shape[0] = total_batch_size;
      }
    }

    std::vector<int64_t> input_dims = batchn_shape;
    for (size_t i = 0; i < input_dims_count; i++) {
      input_dims.emplace(input_shape[i]);
    }

    // The input must be in contiguous CPU memory. Use appropriate
    // allocator info to bind inputs to the right device. .i.e bind inputs
    // to GPU if they are being provided on GPU.
    const char* input_buffer;
    size_t batchn_byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>>
        allowed_input_types;
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
      allowed_input_types = {{TRITONSERVER_MEMORY_GPU, DeviceId()},
                              {TRITONSERVER_MEMORY_CPU_PINNED, 0},
                              {TRITONSERVER_MEMORY_CPU, 0}};
    } else {
      allowed_input_types = {{TRITONSERVER_MEMORY_CPU_PINNED, 0},
                              {TRITONSERVER_MEMORY_CPU, 0}};
    }

    RETURN_IF_ERROR(collector->ProcessTensor(
        input_name, nullptr, 0, allowed_input_types, &input_buffer,
        &batchn_byte_size, &memory_type, &memory_type_id));

    // Alloc OpenPPL Tensor
    auto ppl_tensor = runtime_->GetInputTensor(input_idx);
    ppl_tensor->GetShape()->Reshape(input_dims);
    ppl_tensor->GetShape()->SetDataType(ConvertToOpenPPLDataType(input_datatype));
    ppl_tensor->GetShape()->SetDataFormat(DATAFORMAT_NDARRAY);
    ppl_tensor->ConvertFromHost(input_buffer, *ppl_tensor->GetShape());
  }

  // Finalize...
  *cuda_copy |= collector->Finalize();
  return nullptr;
}


/////////////////////////////////////////////////////////////////////////////

TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->TritonMemoryManager(),
      model_state_->MaxBatchSize() > 0, model_state_->EnablePinnedInput(),
      CudaStream());

  // Use to hold string output contents
  bool cuda_copy = false;
  std::pair<TRITONSERVER_MemoryType, int64_t> alloc_perference = {
      TRITONSERVER_MEMORY_CPU, 0};
  auto& model_outputs = StateForModel()->ModelOutputs();

  for (uint32_t i = 0; i < runtime_->GetOutputCount(); i++) {
    auto ppl_tensor = runtime_->GetOutputTensor(i);
    auto ppl_shape = ppl_tensor->GetShape();
    auto name = ppl_tensor->GetName();
    const BatchOutput* batch_output = StateForModel()->FindBatchOutput(name);

    TRITONSERVER_DataType dtype = ConvertFromOpenPPLDataType(ppl_shape->GetDataType());

    std::vector<int64_t> batchn_shape;
    for (uint32_t j = 0; j < ppl_shape->GetDimCount(); i++)
      batchn_shape.emplace(ppl_shape->GetDim(j));
    
    responder.ProcessTensor(
              name, dtype, batchn_shape, reinterpret_cast<char*>(ppl_tensor->GetBufferPtr()),
              alloc_perference.first, alloc_perference.second);
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRITON_ENABLE_GPU
  return nullptr;
}

}}}  // namespace triton::backend::openppl
