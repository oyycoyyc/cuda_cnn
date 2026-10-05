#include "layers.h"

#include "cuda_check.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

// Fixed launch geometry: 256 threads per block and a 65535-block grid cap.
constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::size_t kMaximumBlocks = 65535;

// Rejects float counts whose byte extent would overflow std::size_t.
void ValidateFloatCount(std::size_t count, const char* operation) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error(std::string(operation) +
                              " count overflows byte extent");
  }
}

// Returns a capped grid-stride block count sufficient to cover count elements.
unsigned int BlockCount(std::size_t count) {
  const std::size_t needed =
      1 + (count - 1) / static_cast<std::size_t>(kThreadsPerBlock);
  return static_cast<unsigned int>(std::min(needed, kMaximumBlocks));
}

// Writes exact positive FP32 zero to every values[i]. A thread starts at
// blockIdx.x*blockDim.x+threadIdx.x and loops by blockDim.x*gridDim.x over
// linear contiguous indices. The index<count condition handles the tail and
// grid-stride exhaustion. Each element has one writer, so no races, barriers,
// synchronization, or atomics are needed. No arithmetic is performed and the
// stored IEEE-754 value is exactly +0.0F.
__global__ void ZeroKernel(float* values, std::size_t count) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = start; index < count; index += stride) {
    values[index] = 0.0F;
  }
}

// Computes output[i]=input[i]>0?input[i]:0. A thread starts at
// blockIdx.x*blockDim.x+threadIdx.x and loops by blockDim.x*gridDim.x over
// contiguous linear coordinates; index<count covers non-divisible tails and
// loop termination. Exact input/output aliasing is safe because each thread
// reads its element before writing the same element; otherwise elements have
// distinct writers, with no races, barriers, synchronization, or atomics.
// Positive finite values pass bit-for-bit, both signed zeros and negatives map
// to +0.0F, and unordered NaN comparisons also map to +0.0F.
__global__ void ReluForwardKernel(const float* input, float* output,
                                  std::size_t count) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = start; index < count; index += stride) {
    const float value = input[index];
    output[index] = value > 0.0F ? value : 0.0F;
  }
}

// Computes input_gradient[i]=(forward_input[i]>0?output_gradient[i]:0), the
// ReLU derivative with derivative zero at both signed zeros. A thread starts at
// blockIdx.x*blockDim.x+threadIdx.x and loops by blockDim.x*gridDim.x over
// linear coordinates; index<count handles tails and termination. Exact
// output_gradient/input_gradient aliasing is safe because the incoming value is
// read before its same-index write. forward_input is disjoint by contract and
// each output has one writer, so there are no races, barriers, synchronization,
// or atomics. Passing gradients are copied without arithmetic; negatives,
// signed zeros, and unordered NaNs in forward_input produce exact +0.0F.
__global__ void ReluBackwardKernel(const float* forward_input,
                                   const float* output_gradient,
                                   float* input_gradient, std::size_t count) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = start; index < count; index += stride) {
    const float incoming = output_gradient[index];
    input_gradient[index] =
        forward_input[index] > 0.0F ? incoming : 0.0F;
  }
}

}  // namespace

// Zeroes count floats with a 256-thread grid-stride launch; a zero count returns
// without launching.
void LaunchZero(float* values, std::size_t count, cudaStream_t stream) {
  ValidateFloatCount(count, "zero");
  if (count == 0) {
    return;
  }
  ZeroKernel<<<BlockCount(count), kThreadsPerBlock, 0, stream>>>(values, count);
  CUDA_KERNEL_CHECK();
}

// Applies elementwise ReLU with a 256-thread grid-stride launch; a zero count
// returns without launching.
void LaunchReluForward(const float* input, float* output, std::size_t count,
                       cudaStream_t stream) {
  ValidateFloatCount(count, "ReLU forward");
  if (count == 0) {
    return;
  }
  ReluForwardKernel<<<BlockCount(count), kThreadsPerBlock, 0, stream>>>(
      input, output, count);
  CUDA_KERNEL_CHECK();
}

// Applies the ReLU derivative using the forward-input mask; a zero count returns
// without launching.
void LaunchReluBackward(const float* forward_input,
                        const float* output_gradient, float* input_gradient,
                        std::size_t count, cudaStream_t stream) {
  ValidateFloatCount(count, "ReLU backward");
  if (count == 0) {
    return;
  }
  ReluBackwardKernel<<<BlockCount(count), kThreadsPerBlock, 0, stream>>>(
      forward_input, output_gradient, input_gradient, count);
  CUDA_KERNEL_CHECK();
}
