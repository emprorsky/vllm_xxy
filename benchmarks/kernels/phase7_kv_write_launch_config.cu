// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

__global__ void kv_write_bf16_nhd_kernel(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping, int num_heads, int head_size,
    int block_size) {
  int token_idx = blockIdx.x;
  int64_t slot = slot_mapping[token_idx];
  if (slot < 0) {
    return;
  }

  constexpr int kVecElements = 8;
  int num_elements = num_heads * head_size;
  int num_vectors = num_elements / kVecElements;
  int64_t block_idx = slot / block_size;
  int64_t block_offset = slot % block_size;
  int64_t src_start = static_cast<int64_t>(token_idx) * num_elements;
  int64_t dst_start =
      (block_idx * block_size + block_offset) * num_elements;

  auto* key_src = reinterpret_cast<const uint4*>(key + src_start);
  auto* value_src = reinterpret_cast<const uint4*>(value + src_start);
  auto* key_dst = reinterpret_cast<uint4*>(key_cache + dst_start);
  auto* value_dst = reinterpret_cast<uint4*>(value_cache + dst_start);
  for (int i = threadIdx.x; i < num_vectors; i += blockDim.x) {
    key_dst[i] = key_src[i];
    value_dst[i] = value_src[i];
  }
}

void kv_write_bf16_nhd(torch::Tensor key, torch::Tensor value,
                       torch::Tensor key_cache, torch::Tensor value_cache,
                       torch::Tensor slot_mapping, int64_t block_threads) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda() && key_cache.is_cuda() &&
                  value_cache.is_cuda() && slot_mapping.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(key.scalar_type() == torch::kBFloat16 &&
                  value.scalar_type() == torch::kBFloat16 &&
                  key_cache.scalar_type() == torch::kBFloat16 &&
                  value_cache.scalar_type() == torch::kBFloat16,
              "the launch probe supports bfloat16 only");
  TORCH_CHECK(key.is_contiguous() && value.is_contiguous() &&
                  key_cache.is_contiguous() && value_cache.is_contiguous(),
              "the launch probe requires contiguous NHD tensors");
  TORCH_CHECK(key.size(1) * key.size(2) % 8 == 0,
              "the element count must be divisible by eight");
  TORCH_CHECK(block_threads == 64 || block_threads == 128 ||
                  block_threads == 256 || block_threads == 512,
              "block_threads must be 64, 128, 256, or 512");

  c10::cuda::CUDAGuard device_guard(key.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(key.get_device());
  kv_write_bf16_nhd_kernel<<<slot_mapping.numel(), block_threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(key_cache.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr()),
      slot_mapping.data_ptr<int64_t>(), key.size(1), key.size(2),
      key_cache.size(1));
}

std::vector<int64_t> occupancy(int64_t block_threads) {
  int active_blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kv_write_bf16_nhd_kernel, block_threads, 0);
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);
  return {active_blocks, properties.maxThreadsPerMultiProcessor,
          properties.maxBlocksPerMultiProcessor,
          static_cast<int64_t>(properties.sharedMemPerMultiprocessor)};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kv_write_bf16_nhd", &kv_write_bf16_nhd);
  m.def("occupancy", &occupancy);
}
