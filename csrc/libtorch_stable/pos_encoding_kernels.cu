#include "torch_utils.h"

#include "../cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t, typename cache_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr, const cache_t* __restrict__ cos_ptr,
    const cache_t* __restrict__ sin_ptr, int rot_offset, int embed_dim,
    const bool inverse) {
  int x_index, y_index;
  float cos_f, sin_f;
  if (IS_NEOX) {
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos_f = static_cast<float>(VLLM_LDG(cos_ptr + x_index));
    sin_f = static_cast<float>(VLLM_LDG(sin_ptr + x_index));
  } else {
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos_f = static_cast<float>(VLLM_LDG(cos_ptr + x_index / 2));
    sin_f = static_cast<float>(VLLM_LDG(sin_ptr + x_index / 2));
  }
  if (inverse) {
    sin_f = -sin_f;
  }
  const float x_f = static_cast<float>(arr[x_index]);
  const float y_f = static_cast<float>(arr[y_index]);
  arr[x_index] = static_cast<scalar_t>(x_f * cos_f - y_f * sin_f);
  arr[y_index] = static_cast<scalar_t>(y_f * cos_f + x_f * sin_f);
}

template <typename scalar_t, typename cache_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,    // nullptr or
                                   // [batch_size, seq_len, num_kv_heads,
                                   // head_size] or [num_tokens, num_kv_heads,
                                   // head_size]
    const cache_t* cache_ptr, const int head_size, const int num_heads,
    const int num_kv_heads, const int rot_dim, const int token_idx,
    const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int64_t rope_dim_offset,
    const bool inverse) {
  const int embed_dim = rot_dim / 2;
  const cache_t* cos_ptr = cache_ptr;
  const cache_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        token_idx * query_stride + head_idx * head_stride + rope_dim_offset;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, cache_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim, inverse);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int64_t token_head =
          token_idx * key_stride + head_idx * head_stride + rope_dim_offset;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, cache_t, IS_NEOX>(
          key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim, inverse);
    }
  }
}

template <typename T, int N>
struct alignas(16) AlignedVector {
  T data[N];
};

template <typename scalar_t, int VEC_SIZE>
inline __device__ void apply_neox_rotary_embedding_vector(
    scalar_t* __restrict__ head, const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr, const int vector_offset,
    const bool inverse) {
  constexpr int kHalfRotaryDim = 64;
  using vector_t = AlignedVector<scalar_t, VEC_SIZE>;
  const int pair_offset = vector_offset < kHalfRotaryDim
                              ? vector_offset + kHalfRotaryDim
                              : vector_offset - kHalfRotaryDim;
  const int cache_offset = vector_offset % kHalfRotaryDim;
  const vector_t values =
      *reinterpret_cast<const vector_t*>(head + vector_offset);
  const vector_t paired =
      *reinterpret_cast<const vector_t*>(head + pair_offset);
  const vector_t cos_values =
      *reinterpret_cast<const vector_t*>(cos_ptr + cache_offset);
  const vector_t sin_values =
      *reinterpret_cast<const vector_t*>(sin_ptr + cache_offset);
  __syncwarp(__activemask());

  vector_t output;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    const float value_f = static_cast<float>(values.data[i]);
    const float paired_f = static_cast<float>(paired.data[i]);
    const float cos_f = static_cast<float>(cos_values.data[i]);
    float sin_f = static_cast<float>(sin_values.data[i]);
    if (inverse) {
      sin_f = -sin_f;
    }
    const float signed_sin =
        vector_offset < kHalfRotaryDim ? -sin_f : sin_f;
    output.data[i] =
        static_cast<scalar_t>(value_f * cos_f + paired_f * signed_sin);
  }
  *reinterpret_cast<vector_t*>(head + vector_offset) = output;
}

template <typename scalar_t, int VEC_SIZE>
__global__ void neox_rotary_embedding_head_parallel_kernel(
    const int64_t* __restrict__ positions, scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* __restrict__ cos_sin_cache, const int num_tokens,
    const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const bool inverse) {
  constexpr int kRotaryDim = 128;
  const int token_idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (token_idx >= num_tokens) {
    return;
  }
  const int head_idx = blockIdx.y;
  scalar_t* head;
  if (head_idx < num_heads) {
    head = query + token_idx * query_stride + head_idx * head_stride;
  } else {
    head = key + token_idx * key_stride +
           (head_idx - num_heads) * head_stride;
  }
  const int64_t position = positions[token_idx];
  const scalar_t* cache = cos_sin_cache + position * kRotaryDim;
  const int vector_offset = threadIdx.x * VEC_SIZE;
  apply_neox_rotary_embedding_vector<scalar_t, VEC_SIZE>(
      head, cache, cache + kRotaryDim / 2, vector_offset, inverse);
}

template <typename scalar_t, int VEC_SIZE>
__global__ void neox_rotary_embedding_token_parallel_kernel(
    const int64_t* __restrict__ positions, scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* __restrict__ cos_sin_cache, const int num_tokens,
    const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const bool inverse) {
  constexpr int kRotaryDim = 128;
  const int token_idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (token_idx >= num_tokens) {
    return;
  }
  const int64_t position = positions[token_idx];
  const scalar_t* cache = cos_sin_cache + position * kRotaryDim;
  const int vector_offset = threadIdx.x * VEC_SIZE;
  const scalar_t* cos_ptr = cache;
  const scalar_t* sin_ptr = cache + kRotaryDim / 2;

  for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
    scalar_t* head =
        query + token_idx * query_stride + head_idx * head_stride;
    apply_neox_rotary_embedding_vector<scalar_t, VEC_SIZE>(
        head, cos_ptr, sin_ptr, vector_offset, inverse);
  }
  for (int head_idx = 0; head_idx < num_kv_heads; ++head_idx) {
    scalar_t* head = key + token_idx * key_stride + head_idx * head_stride;
    apply_neox_rotary_embedding_vector<scalar_t, VEC_SIZE>(
        head, cos_ptr, sin_ptr, vector_offset, inverse);
  }
}

template <typename scalar_t, typename cache_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    scalar_t* __restrict__ query,           // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,  // nullptr or
                                 // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const cache_t* __restrict__ cos_sin_cache,  // [max_position, rot_dim]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const int head_size, const int64_t rope_dim_offset, const bool inverse) {
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const cache_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<scalar_t, cache_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx, query_stride, key_stride, head_stride, rope_dim_offset,
      inverse);
}

}  // namespace vllm

void rotary_embedding(
    torch::stable::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::stable::Tensor&
        query,  // [batch_size, seq_len, num_heads * head_size] or
                // [num_tokens, num_heads * head_size] or
                // [batch_size, seq_len, num_heads, head_size] or
                // [num_tokens, num_heads, head_size]
    std::optional<torch::stable::Tensor> key,
    // null or
    // [batch_size, seq_len, num_kv_heads * head_size] or
    // [num_tokens, num_kv_heads * head_size] or
    // [batch_size, seq_len, num_heads, head_size] or
    // [num_tokens, num_heads, head_size]
    int64_t head_size,
    torch::stable::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox, int64_t rope_dim_offset, bool inverse) {
  // num_tokens = batch_size * seq_len
  int64_t num_tokens = positions.numel();
  int positions_ndim = positions.dim();

  // Make sure num_tokens dim is consistent across positions, query, and key
  STD_TORCH_CHECK(
      positions_ndim == 1 || positions_ndim == 2,
      "positions must have shape [num_tokens] or [batch_size, seq_len]");
  if (positions_ndim == 1) {
    STD_TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (!key.has_value() || key->size(0) == positions.size(0)),
        "query, key and positions must have the same number of tokens");
  }
  if (positions_ndim == 2) {
    STD_TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (!key.has_value() || key->size(0) == positions.size(0)) &&
            query.size(1) == positions.size(1) &&
            (!key.has_value() || key->size(1) == positions.size(1)),
        "query, key and positions must have the same batch_size and seq_len");
  }

  // Make sure head_size is valid for query and key
  // hidden_size = num_heads * head_size
  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  STD_TORCH_CHECK(query_hidden_size % head_size == 0);
  STD_TORCH_CHECK(key_hidden_size % head_size == 0);

  // Make sure query and key have consistent number of heads
  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key.has_value() ? key_hidden_size / head_size : num_heads;
  STD_TORCH_CHECK(num_heads % num_kv_heads == 0);

  int rot_dim = cos_sin_cache.size(1);
  int seq_dim_idx = positions_ndim - 1;
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;

  STD_TORCH_CHECK((rot_dim + rope_dim_offset) <= head_size);
  // Determine head stride: for [*, heads, head_size] use stride of last dim;
  // for flat [*, heads*head_size], heads blocks are contiguous of size
  // head_size
  int query_ndim = query.dim();
  int64_t head_stride =
      (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  constexpr int vector_width = 8;
  constexpr int vector_alignment = vector_width * 2;
  const bool can_vectorize =
      key.has_value() && positions_ndim == 1 && is_neox && head_size == 128 &&
      rot_dim == 128 && rope_dim_offset == 0 &&
      query.scalar_type() == key->scalar_type() &&
      query.scalar_type() == cos_sin_cache.scalar_type() &&
      query.element_size() == 2 && positions.stride(0) == 1 &&
      cos_sin_cache.stride(1) == 1 && query_stride % vector_width == 0 &&
      key_stride % vector_width == 0 && head_stride % vector_width == 0 &&
      reinterpret_cast<uintptr_t>(query.mutable_data_ptr()) % vector_alignment ==
          0 &&
      reinterpret_cast<uintptr_t>(key->mutable_data_ptr()) % vector_alignment ==
          0 &&
      reinterpret_cast<uintptr_t>(cos_sin_cache.const_data_ptr()) %
              vector_alignment ==
          0;
  if (can_vectorize) {
    constexpr int threads_x = 128 / vector_width;
    constexpr int threads_y = 8;
    dim3 block(threads_x, threads_y);
    dim3 grid((num_tokens + threads_y - 1) / threads_y);
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(
        query.scalar_type(), "rotary_embedding_vector", [&] {
          using query_t = scalar_t;
          if (num_tokens < 4096) {
            grid.y = num_heads + num_kv_heads;
            vllm::neox_rotary_embedding_head_parallel_kernel<query_t,
                                                              vector_width>
                <<<grid, block, 0, stream>>>(
                    positions.const_data_ptr<int64_t>(),
                    query.mutable_data_ptr<query_t>(),
                    key->mutable_data_ptr<query_t>(),
                    cos_sin_cache.const_data_ptr<query_t>(), num_tokens,
                    query_stride, key_stride, head_stride, num_heads, inverse);
          } else {
            vllm::neox_rotary_embedding_token_parallel_kernel<query_t,
                                                               vector_width>
                <<<grid, block, 0, stream>>>(
                    positions.const_data_ptr<int64_t>(),
                    query.mutable_data_ptr<query_t>(),
                    key->mutable_data_ptr<query_t>(),
                    cos_sin_cache.const_data_ptr<query_t>(), num_tokens,
                    query_stride, key_stride, head_stride, num_heads,
                    num_kv_heads, inverse);
          }
        });
    return;
  }

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "rotary_embedding", [&] {
        using query_t = scalar_t;
        VLLM_STABLE_DISPATCH_FLOATING_TYPES(
            cos_sin_cache.scalar_type(), "rotary_embedding_cache", [&] {
              using cache_t = scalar_t;
              if (is_neox) {
                vllm::rotary_embedding_kernel<query_t, cache_t, true>
                    <<<grid, block, 0, stream>>>(
                        positions.const_data_ptr<int64_t>(),
                        query.mutable_data_ptr<query_t>(),
                        key.has_value() ? key->mutable_data_ptr<query_t>()
                                        : nullptr,
                        cos_sin_cache.const_data_ptr<cache_t>(), rot_dim,
                        query_stride, key_stride, head_stride, num_heads,
                        num_kv_heads, head_size, rope_dim_offset, inverse);
              } else {
                vllm::rotary_embedding_kernel<query_t, cache_t, false>
                    <<<grid, block, 0, stream>>>(
                        positions.const_data_ptr<int64_t>(),
                        query.mutable_data_ptr<query_t>(),
                        key.has_value() ? key->mutable_data_ptr<query_t>()
                                        : nullptr,
                        cos_sin_cache.const_data_ptr<cache_t>(), rot_dim,
                        query_stride, key_stride, head_stride, num_heads,
                        num_kv_heads, head_size, rope_dim_offset, inverse);
              }
            });
      });
}
