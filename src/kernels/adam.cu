#include "layers.h"

#include "cuda_check.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::size_t kMaximumBlocks = 65535;

struct BufferRange {
  const void* pointer;
  std::size_t bytes;
};

void ValidateFloatCount(std::size_t count, const char* operation) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error(std::string(operation) +
                              " count overflows byte extent");
  }
}

void ValidateBuffers(std::initializer_list<BufferRange> buffers,
                     const char* operation) {
  for (const BufferRange& buffer : buffers) {
    if (buffer.pointer == nullptr) {
      throw std::invalid_argument(std::string(operation) +
                                  " pointer must be non-null");
    }
  }
  for (auto left = buffers.begin(); left != buffers.end(); ++left) {
    const std::uintptr_t left_begin =
        reinterpret_cast<std::uintptr_t>(left->pointer);
    if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left->bytes) {
      throw std::invalid_argument(std::string(operation) +
                                  " pointer range is invalid");
    }
    const std::uintptr_t left_end = left_begin + left->bytes;
    for (auto right = left + 1; right != buffers.end(); ++right) {
      const std::uintptr_t right_begin =
          reinterpret_cast<std::uintptr_t>(right->pointer);
      if (right_begin >
          std::numeric_limits<std::uintptr_t>::max() - right->bytes) {
        throw std::invalid_argument(std::string(operation) +
                                    " pointer range is invalid");
      }
      const std::uintptr_t right_end = right_begin + right->bytes;
      if (left_begin < right_end && right_begin < left_end) {
        throw std::invalid_argument(std::string(operation) +
                                    " buffers overlap");
      }
    }
  }
}

unsigned int BlockCount(std::size_t count) {
  const std::size_t needed =
      1 + (count - 1) / static_cast<std::size_t>(kThreadsPerBlock);
  return static_cast<unsigned int>(std::min(needed, kMaximumBlocks));
}

void ValidateAdamScalars(float learning_rate, float beta1, float beta2,
                         float epsilon, float inverse_bias_correction1,
                         float inverse_bias_correction2, float weight_decay) {
  if (!std::isfinite(learning_rate) || learning_rate < 0.0F) {
    throw std::invalid_argument(
        "AdamW learning_rate must be finite and nonnegative");
  }
  if (!std::isfinite(beta1) || beta1 < 0.0F || beta1 >= 1.0F ||
      !std::isfinite(beta2) || beta2 < 0.0F || beta2 >= 1.0F) {
    throw std::invalid_argument("AdamW beta values must be finite in [0,1)");
  }
  if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
    throw std::invalid_argument("AdamW epsilon must be finite and positive");
  }
  if (!std::isfinite(inverse_bias_correction1) ||
      inverse_bias_correction1 < 1.0F ||
      !std::isfinite(inverse_bias_correction2) ||
      inverse_bias_correction2 < 1.0F) {
    throw std::invalid_argument(
        "AdamW inverse bias corrections must be finite and at least one");
  }
  if (!std::isfinite(weight_decay) || weight_decay < 0.0F) {
    throw std::invalid_argument(
        "AdamW weight_decay must be finite and nonnegative");
  }
}

// A thread grid-strides over parameter indices; every index is visited by
// exactly one thread and all four arrays are disjoint, so no writes race and no
// shared memory, barriers, atomics, or synchronization are needed. For each i,
// m'=beta1*m+(1-beta1)*g and v'=beta2*v+(1-beta2)*g^2, followed by
// p'=p-lr*((m'*inverse_bias1)/(sqrt(v'*inverse_bias2)+epsilon)+decay*p).
// parameter captures pre-update p, so decoupled decay never observes p'. The
// host supplies FP32 factors converted from double 1/(1-beta^t), including t=1;
// a bias caller supplies decay=0. FP32 products/sqrtf have normal rounding, and
// finite inputs with nonnegative v make the denominator positive.
__global__ void AdamWKernel(
    float* parameters, const float* gradients, float* first_moments,
    float* second_moments, std::size_t count, float learning_rate, float beta1,
    float beta2, float epsilon, float inverse_bias_correction1,
    float inverse_bias_correction2, float weight_decay) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = start; index < count; index += stride) {
    const float parameter = parameters[index];
    const float gradient = gradients[index];
    const float first =
        beta1 * first_moments[index] + (1.0F - beta1) * gradient;
    const float second = beta2 * second_moments[index] +
                         (1.0F - beta2) * gradient * gradient;
    first_moments[index] = first;
    second_moments[index] = second;
    const float corrected_first = first * inverse_bias_correction1;
    const float corrected_second = second * inverse_bias_correction2;
    parameters[index] =
        parameter -
        learning_rate *
            (corrected_first / (sqrtf(corrected_second) + epsilon) +
             weight_decay * parameter);
  }
}

// Exactly one thread writes result[0]=count before scanning starts. count has
// already been constrained to INT_MAX, so the sentinel conversion is exact.
// There is no shared memory, barrier, atomic, race, arithmetic instability, or
// boundary loop: one scalar output has one writer. Same-stream launch ordering
// makes this initialization visible to the following scan without host sync.
__global__ void InitializeFirstBadIndexKernel(int* result, int count) {
  result[0] = count;
}

// Threads grid-stride over [0,count), including non-divisible tails such as
// 257. Finite values perform no write; each NaN or +/-infinity atomically lowers
// result[0] with its representable int index. atomicMin is the sole shared write
// and is order-independent, so concurrent bad values deterministically leave
// the smallest index without a barrier or shared memory. The preceding
// same-stream initializer supplies count when no atomic executes; isfinite is a
// CUDA device math classification and performs no numerically unstable math.
__global__ void FindFirstNonFiniteKernel(const float* values, int count,
                                         int* result) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = start; index < static_cast<std::size_t>(count);
       index += stride) {
    if (!isfinite(values[index])) {
      atomicMin(result, static_cast<int>(index));
    }
  }
}

}  // namespace

void LaunchAdamW(float* parameters, const float* gradients,
                 float* first_moments, float* second_moments,
                 std::size_t count, float learning_rate, float beta1,
                 float beta2, float epsilon, float inverse_bias_correction1,
                 float inverse_bias_correction2, float weight_decay,
                 cudaStream_t stream) {
  ValidateFloatCount(count, "AdamW");
  ValidateAdamScalars(learning_rate, beta1, beta2, epsilon,
                      inverse_bias_correction1, inverse_bias_correction2,
                      weight_decay);
  if (count == 0) {
    return;
  }
  const std::size_t bytes = count * sizeof(float);
  ValidateBuffers({{parameters, bytes},
                   {gradients, bytes},
                   {first_moments, bytes},
                   {second_moments, bytes}},
                  "AdamW");
  AdamWKernel<<<BlockCount(count), kThreadsPerBlock, 0, stream>>>(
      parameters, gradients, first_moments, second_moments, count,
      learning_rate, beta1, beta2, epsilon, inverse_bias_correction1,
      inverse_bias_correction2, weight_decay);
  CUDA_KERNEL_CHECK();
}

void LaunchFindFirstNonFinite(const float* values, std::size_t count,
                              int* first_bad_index, cudaStream_t stream) {
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("finite scan count exceeds INT_MAX");
  }
  ValidateFloatCount(count, "finite scan");
  if (first_bad_index == nullptr) {
    throw std::invalid_argument("finite scan result must be non-null");
  }
  const std::uintptr_t result_begin =
      reinterpret_cast<std::uintptr_t>(first_bad_index);
  if (result_begin >
      std::numeric_limits<std::uintptr_t>::max() - sizeof(int)) {
    throw std::invalid_argument("finite scan result pointer range is invalid");
  }
  if (count > 0) {
    if (values == nullptr) {
      throw std::invalid_argument("finite scan values must be non-null");
    }
    ValidateBuffers({{values, count * sizeof(float)},
                     {first_bad_index, sizeof(int)}},
                    "finite scan");
  }
  InitializeFirstBadIndexKernel<<<1, 1, 0, stream>>>(
      first_bad_index, static_cast<int>(count));
  CUDA_KERNEL_CHECK();
  if (count == 0) {
    return;
  }
  FindFirstNonFiniteKernel<<<BlockCount(count), kThreadsPerBlock, 0, stream>>>(
      values, static_cast<int>(count), first_bad_index);
  CUDA_KERNEL_CHECK();
}
