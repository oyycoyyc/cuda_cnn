#include "layers.h"

#include "cuda_check.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace {

// Fixed launch geometry: 256 threads per block and a 65535-block grid cap.
constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::size_t kMaximumBlocks = 65535;

// Validated linear geometry and the derived element counts.
struct LinearShape {
  int batch_size;
  int input_features;
  int output_features;
  std::size_t input_count;
  std::size_t weight_count;
  std::size_t output_count;
};

// A device buffer's base pointer and byte length used for overlap checking.
struct BufferRange {
  const void* pointer;
  std::size_t bytes;
};

// Multiplies positive dimensions, failing closed on non-positive values or
// overflow of the size_t element or byte extent.
std::size_t CheckedProduct(std::initializer_list<int> dimensions) {
  std::size_t product = 1;
  for (int dimension : dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument("linear dimensions must be positive");
    }
    const std::size_t value = static_cast<std::size_t>(dimension);
    if (product > std::numeric_limits<std::size_t>::max() / value) {
      throw std::overflow_error("linear dimensions overflow size_t");
    }
    product *= value;
  }
  if (product > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error("linear dimensions overflow byte extent");
  }
  return product;
}

// Validates linear dimensions and derives the input, weight, and output counts.
LinearShape ValidateLinearShape(int batch_size, int input_features,
                                int output_features) {
  const std::size_t input_count =
      CheckedProduct({batch_size, input_features});
  const std::size_t weight_count =
      CheckedProduct({output_features, input_features});
  const std::size_t output_count =
      CheckedProduct({batch_size, output_features});
  CheckedProduct({output_features});
  return LinearShape{batch_size, input_features, output_features, input_count,
                     weight_count, output_count};
}

// Rejects null pointers and overlapping byte ranges across the supplied buffers.
void ValidateBuffers(std::initializer_list<BufferRange> buffers) {
  for (const BufferRange& buffer : buffers) {
    if (buffer.pointer == nullptr) {
      throw std::invalid_argument("linear pointer must be non-null");
    }
  }
  for (auto left = buffers.begin(); left != buffers.end(); ++left) {
    const std::uintptr_t left_begin =
        reinterpret_cast<std::uintptr_t>(left->pointer);
    if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left->bytes) {
      throw std::invalid_argument("linear pointer range is invalid");
    }
    const std::uintptr_t left_end = left_begin + left->bytes;
    for (auto right = left + 1; right != buffers.end(); ++right) {
      const std::uintptr_t right_begin =
          reinterpret_cast<std::uintptr_t>(right->pointer);
      if (right_begin >
          std::numeric_limits<std::uintptr_t>::max() - right->bytes) {
        throw std::invalid_argument("linear pointer range is invalid");
      }
      const std::uintptr_t right_end = right_begin + right->bytes;
      if (left_begin < right_end && right_begin < left_end) {
        throw std::invalid_argument("linear buffers overlap");
      }
    }
  }
}

// Returns a capped grid-stride block count sufficient to cover count elements.
unsigned int BlockCount(std::size_t count) {
  const std::size_t needed =
      1 + (count - 1) / static_cast<std::size_t>(kThreadsPerBlock);
  return static_cast<unsigned int>(std::min(needed, kMaximumBlocks));
}

// Computes y[n,of]=bias[of]+sum_inf input[n,inf]*weight[of,inf]. A
// thread starts at blockIdx.x*blockDim.x+threadIdx.x and grid-strides over
// output_index=n*OF+of, where n=output_index/OF and of=output_index%OF;
// input_index=n*IF+inf and weight_index=of*IF+inf. The outer
// output_index<output_count condition handles excess threads and large grids,
// while inf runs upward from zero exactly as the CPU reference. Every output
// element has one owner and all other buffers are read-only, so there are no
// races, barriers, atomics, or synchronization. Each owner performs a serial
// FP32 accumulation and one bias addition; fixed loop order is deterministic,
// while ordinary FP32 rounding (including compiler contraction) applies.
__global__ void LinearForwardKernel(const float* input, const float* weight,
                                    const float* bias, float* output,
                                    std::size_t output_count,
                                    int input_features,
                                    int output_features) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t output_index = start; output_index < output_count;
       output_index += stride) {
    const std::size_t sample = output_index / output_features;
    const int output_feature =
        static_cast<int>(output_index % output_features);
    const std::size_t input_base = sample * input_features;
    const std::size_t weight_base =
        static_cast<std::size_t>(output_feature) * input_features;
    float sum = 0.0F;
    for (int input_feature = 0; input_feature < input_features;
         ++input_feature) {
      sum += input[input_base + input_feature] *
             weight[weight_base + input_feature];
    }
    output[output_index] = sum + bias[output_feature];
  }
}

// Computes dInput[n,inf]=sum_of dOutput[n,of]*weight[of,inf], the derivative
// of y=input*weight^T+bias with respect to one input. Threads grid-stride over
// input_index=n*IF+inf, decoded by n=input_index/IF and inf=input_index%IF;
// dOutput uses n*OF+of and weight uses of*IF+inf. input_index<input_count
// handles tails, and of increases from zero in CPU-reference order. One thread
// owns each dInput and only reads upstream/weight, so no races, barriers,
// atomics, or synchronization occur. The serial FP32 sum is deterministic for
// a fixed build/device and has normal accumulation and contraction rounding.
__global__ void LinearInputGradientKernel(
    const float* weight, const float* output_gradient, float* input_gradient,
    std::size_t input_count, int input_features, int output_features) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t input_index = start; input_index < input_count;
       input_index += stride) {
    const std::size_t sample = input_index / input_features;
    const int input_feature =
        static_cast<int>(input_index % input_features);
    const std::size_t output_base = sample * output_features;
    float sum = 0.0F;
    for (int output_feature = 0; output_feature < output_features;
         ++output_feature) {
      sum += output_gradient[output_base + output_feature] *
             weight[static_cast<std::size_t>(output_feature) * input_features +
                    input_feature];
    }
    input_gradient[input_index] = sum;
  }
}

// Computes dWeight[of,inf]=sum_n input[n,inf]*dOutput[n,of]. Threads
// grid-stride over weight_index=of*IF+inf, decoded by division/remainder; input
// uses n*IF+inf and dOutput uses n*OF+of. weight_index<weight_count covers
// launch tails and n increases from zero to batch_size-1 in CPU-reference
// order. Each weight gradient has one gather owner, so writes never race and no
// barriers, atomics, or synchronization are needed. The sum is serial FP32 and
// deterministic in loop order, with ordinary accumulation/contraction error.
__global__ void LinearWeightGradientKernel(
    const float* input, const float* output_gradient, float* weight_gradient,
    std::size_t weight_count, int batch_size, int input_features,
    int output_features) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t weight_index = start; weight_index < weight_count;
       weight_index += stride) {
    const int output_feature =
        static_cast<int>(weight_index / input_features);
    const int input_feature =
        static_cast<int>(weight_index % input_features);
    float sum = 0.0F;
    for (int sample = 0; sample < batch_size; ++sample) {
      sum += input[static_cast<std::size_t>(sample) * input_features +
                   input_feature] *
             output_gradient[static_cast<std::size_t>(sample) *
                                 output_features +
                             output_feature];
    }
    weight_gradient[weight_index] = sum;
  }
}

// Computes dBias[of]=sum_n dOutput[n,of], the derivative of adding one bias to
// every sample. Threads grid-stride over of; of<output_features handles tails,
// and dOutput's row-major index is n*OF+of with n accumulated upward exactly as
// in the CPU reference. One thread owns each bias gradient and reads disjoint or
// shared immutable upstream values, so no races, barriers, atomics, or
// synchronization occur. Serial FP32 addition has deterministic order and the
// expected rounding error of a batch reduction.
__global__ void LinearBiasGradientKernel(const float* output_gradient,
                                         float* bias_gradient, int batch_size,
                                         int output_features) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t output_feature = start;
       output_feature < static_cast<std::size_t>(output_features);
       output_feature += stride) {
    float sum = 0.0F;
    for (int sample = 0; sample < batch_size; ++sample) {
      sum += output_gradient[static_cast<std::size_t>(sample) *
                                 output_features +
                             output_feature];
    }
    bias_gradient[output_feature] = sum;
  }
}

}  // namespace

// Launches the linear forward kernel over the validated output extent.
void LaunchLinearForward(const float* input, const float* weight,
                         const float* bias, float* output, int batch_size,
                         int input_features, int output_features,
                         cudaStream_t stream) {
  const LinearShape shape =
      ValidateLinearShape(batch_size, input_features, output_features);
  ValidateBuffers({{input, shape.input_count * sizeof(float)},
                   {weight, shape.weight_count * sizeof(float)},
                   {bias, static_cast<std::size_t>(output_features) *
                              sizeof(float)},
                   {output, shape.output_count * sizeof(float)}});
  LinearForwardKernel<<<BlockCount(shape.output_count), kThreadsPerBlock, 0,
                        stream>>>(input, weight, bias, output,
                                  shape.output_count, shape.input_features,
                                  shape.output_features);
  CUDA_KERNEL_CHECK();
}

// Launches the ordered input, weight, then bias gradient kernels on one stream.
void LaunchLinearBackward(const float* input, const float* weight,
                          const float* output_gradient, float* input_gradient,
                          float* weight_gradient, float* bias_gradient,
                          int batch_size, int input_features,
                          int output_features, cudaStream_t stream) {
  const LinearShape shape =
      ValidateLinearShape(batch_size, input_features, output_features);
  ValidateBuffers({{input, shape.input_count * sizeof(float)},
                   {weight, shape.weight_count * sizeof(float)},
                   {output_gradient, shape.output_count * sizeof(float)},
                   {input_gradient, shape.input_count * sizeof(float)},
                   {weight_gradient, shape.weight_count * sizeof(float)},
                   {bias_gradient, static_cast<std::size_t>(output_features) *
                                       sizeof(float)}});
  LinearInputGradientKernel<<<BlockCount(shape.input_count), kThreadsPerBlock,
                              0, stream>>>(
      weight, output_gradient, input_gradient, shape.input_count,
      shape.input_features, shape.output_features);
  CUDA_KERNEL_CHECK();
  LinearWeightGradientKernel<<<BlockCount(shape.weight_count), kThreadsPerBlock,
                               0, stream>>>(
      input, output_gradient, weight_gradient, shape.weight_count,
      shape.batch_size, shape.input_features, shape.output_features);
  CUDA_KERNEL_CHECK();
  LinearBiasGradientKernel<<<BlockCount(shape.output_features),
                             kThreadsPerBlock, 0, stream>>>(
      output_gradient, bias_gradient, shape.batch_size, shape.output_features);
  CUDA_KERNEL_CHECK();
}
