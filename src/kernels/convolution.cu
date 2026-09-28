#include "layers.h"

#include "cuda_check.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::size_t kMaximumBlocks = 65535;

struct ConvolutionShape {
  int batch_size;
  int input_channels;
  int input_height;
  int input_width;
  int output_channels;
  int kernel_height;
  int kernel_width;
  int output_height;
  int output_width;
  std::size_t input_count;
  std::size_t weight_count;
  std::size_t output_count;
};

struct BufferRange {
  const void* pointer;
  std::size_t bytes;
};

std::size_t CheckedProduct(std::initializer_list<int> dimensions) {
  std::size_t product = 1;
  for (int dimension : dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument("convolution dimensions must be positive");
    }
    const std::size_t value = static_cast<std::size_t>(dimension);
    if (product > std::numeric_limits<std::size_t>::max() / value) {
      throw std::overflow_error("convolution dimensions overflow size_t");
    }
    product *= value;
  }
  if (product > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error("convolution dimensions overflow byte extent");
  }
  return product;
}

ConvolutionShape ValidateConvolutionShape(
    int batch_size, int input_channels, int input_height, int input_width,
    int output_channels, int kernel_height, int kernel_width) {
  if (batch_size <= 0 || input_channels <= 0 || input_height <= 0 ||
      input_width <= 0 || output_channels <= 0 || kernel_height <= 0 ||
      kernel_width <= 0) {
    throw std::invalid_argument("convolution dimensions must be positive");
  }
  if (kernel_height > input_height || kernel_width > input_width) {
    throw std::invalid_argument("convolution kernel must fit input");
  }
  const int output_height = input_height - kernel_height + 1;
  const int output_width = input_width - kernel_width + 1;
  const std::size_t input_count = CheckedProduct(
      {batch_size, input_channels, input_height, input_width});
  const std::size_t weight_count = CheckedProduct(
      {output_channels, input_channels, kernel_height, kernel_width});
  const std::size_t output_count = CheckedProduct(
      {batch_size, output_channels, output_height, output_width});
  CheckedProduct({output_channels});
  return ConvolutionShape{
      batch_size,      input_channels, input_height, input_width,
      output_channels, kernel_height,  kernel_width, output_height,
      output_width,    input_count,    weight_count, output_count};
}

void ValidateBuffers(std::initializer_list<BufferRange> buffers) {
  for (const BufferRange& buffer : buffers) {
    if (buffer.pointer == nullptr) {
      throw std::invalid_argument("convolution pointer must be non-null");
    }
  }
  for (auto left = buffers.begin(); left != buffers.end(); ++left) {
    const std::uintptr_t left_begin =
        reinterpret_cast<std::uintptr_t>(left->pointer);
    if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left->bytes) {
      throw std::invalid_argument("convolution pointer range is invalid");
    }
    const std::uintptr_t left_end = left_begin + left->bytes;
    for (auto right = left + 1; right != buffers.end(); ++right) {
      const std::uintptr_t right_begin =
          reinterpret_cast<std::uintptr_t>(right->pointer);
      if (right_begin >
          std::numeric_limits<std::uintptr_t>::max() - right->bytes) {
        throw std::invalid_argument("convolution pointer range is invalid");
      }
      const std::uintptr_t right_end = right_begin + right->bytes;
      if (left_begin < right_end && right_begin < left_end) {
        throw std::invalid_argument("convolution buffers overlap");
      }
    }
  }
}

unsigned int BlockCount(std::size_t count) {
  const std::size_t needed =
      1 + (count - 1) / static_cast<std::size_t>(kThreadsPerBlock);
  return static_cast<unsigned int>(std::min(needed, kMaximumBlocks));
}

// Computes valid stride-one y[n,oc,orow,ocol]=bias[oc]+sum_ic,kr,kc
// input[n,ic,orow+kr,ocol+kc]*weight[oc,ic,kr,kc]. Threads grid-stride over
// output_index=((n*OC+oc)*OH+orow)*OW+ocol; division/remainder by OH*OW and OW
// recover those coordinates. Input and OIHW weight indices use the same NCHW
// formula with IH/IW and KH/KW. output_index<output_count handles launch tails;
// ic,kr,kc increase in CPU-reference order. One thread owns each output and all
// operands are read-only, so there are no races, barriers, atomic operations,
// or synchronization. Accumulation is serial FP32 followed by one bias add,
// deterministic in loop order with ordinary FP32/contraction rounding.
__global__ void ConvolutionForwardKernel(
    const float* input, const float* weight, const float* bias, float* output,
    std::size_t output_count, int input_channels, int input_height,
    int input_width, int output_channels, int kernel_height, int kernel_width,
    int output_height, int output_width) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t output_plane =
      static_cast<std::size_t>(output_height) * output_width;
  const std::size_t output_sample =
      static_cast<std::size_t>(output_channels) * output_plane;
  const std::size_t input_plane =
      static_cast<std::size_t>(input_height) * input_width;
  const std::size_t input_sample =
      static_cast<std::size_t>(input_channels) * input_plane;
  const std::size_t kernel_plane =
      static_cast<std::size_t>(kernel_height) * kernel_width;
  for (std::size_t output_index = start; output_index < output_count;
       output_index += stride) {
    const std::size_t sample = output_index / output_sample;
    const std::size_t within_sample = output_index % output_sample;
    const int output_channel =
        static_cast<int>(within_sample / output_plane);
    const std::size_t spatial = within_sample % output_plane;
    const int output_row = static_cast<int>(spatial / output_width);
    const int output_column = static_cast<int>(spatial % output_width);
    float sum = 0.0F;
    for (int input_channel = 0; input_channel < input_channels;
         ++input_channel) {
      const std::size_t input_channel_base =
          sample * input_sample +
          static_cast<std::size_t>(input_channel) * input_plane;
      const std::size_t weight_channel_base =
          (static_cast<std::size_t>(output_channel) * input_channels +
           input_channel) *
          kernel_plane;
      for (int kernel_row = 0; kernel_row < kernel_height; ++kernel_row) {
        const std::size_t input_row_base =
            input_channel_base +
            static_cast<std::size_t>(output_row + kernel_row) * input_width;
        const std::size_t weight_row_base =
            weight_channel_base +
            static_cast<std::size_t>(kernel_row) * kernel_width;
        for (int kernel_column = 0; kernel_column < kernel_width;
             ++kernel_column) {
          sum += input[input_row_base + output_column + kernel_column] *
                 weight[weight_row_base + kernel_column];
        }
      }
    }
    output[output_index] = sum + bias[output_channel];
  }
}

// Computes dInput[n,ic,ir,icol] by gathering every dOutput[n,oc,orow,ocol]
// whose valid receptive field covers that coordinate: orow=ir-kr and
// ocol=icol-kc. Threads grid-stride over
// input_index=((n*IC+ic)*IH+ir)*IW+icol and decode with IC*IH*IW, IH*IW, and
// IW. Bounds 0<=orow<OH and 0<=ocol<OW reject non-covering kernel positions;
// oc,kr,kc loop upward in CPU-reference order and input_index<input_count
// handles launch tails. Each dInput has one gather owner, so no writes race and
// no barriers, atomic operations, or synchronization are needed. The serial
// FP32 reduction is deterministic in order with normal accumulation rounding.
__global__ void ConvolutionInputGradientKernel(
    const float* weight, const float* output_gradient, float* input_gradient,
    std::size_t input_count, int input_channels, int input_height,
    int input_width, int output_channels, int kernel_height, int kernel_width,
    int output_height, int output_width) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t input_plane =
      static_cast<std::size_t>(input_height) * input_width;
  const std::size_t input_sample =
      static_cast<std::size_t>(input_channels) * input_plane;
  const std::size_t output_plane =
      static_cast<std::size_t>(output_height) * output_width;
  const std::size_t output_sample =
      static_cast<std::size_t>(output_channels) * output_plane;
  const std::size_t kernel_plane =
      static_cast<std::size_t>(kernel_height) * kernel_width;
  for (std::size_t input_index = start; input_index < input_count;
       input_index += stride) {
    const std::size_t sample = input_index / input_sample;
    const std::size_t within_sample = input_index % input_sample;
    const int input_channel = static_cast<int>(within_sample / input_plane);
    const std::size_t spatial = within_sample % input_plane;
    const int input_row = static_cast<int>(spatial / input_width);
    const int input_column = static_cast<int>(spatial % input_width);
    float sum = 0.0F;
    for (int output_channel = 0; output_channel < output_channels;
         ++output_channel) {
      const std::size_t output_channel_base =
          sample * output_sample +
          static_cast<std::size_t>(output_channel) * output_plane;
      const std::size_t weight_channel_base =
          (static_cast<std::size_t>(output_channel) * input_channels +
           input_channel) *
          kernel_plane;
      for (int kernel_row = 0; kernel_row < kernel_height; ++kernel_row) {
        const int output_row = input_row - kernel_row;
        if (output_row < 0 || output_row >= output_height) {
          continue;
        }
        const std::size_t weight_row_base =
            weight_channel_base +
            static_cast<std::size_t>(kernel_row) * kernel_width;
        for (int kernel_column = 0; kernel_column < kernel_width;
             ++kernel_column) {
          const int output_column = input_column - kernel_column;
          if (output_column < 0 || output_column >= output_width) {
            continue;
          }
          sum += output_gradient[
                     output_channel_base +
                     static_cast<std::size_t>(output_row) * output_width +
                     output_column] *
                 weight[weight_row_base + kernel_column];
        }
      }
    }
    input_gradient[input_index] = sum;
  }
}

// Computes dWeight[oc,ic,kr,kc]=sum_n,orow,ocol
// input[n,ic,orow+kr,ocol+kc]*dOutput[n,oc,orow,ocol]. Threads grid-stride
// over weight_index=((oc*IC+ic)*KH+kr)*KW+kc and decode by IC*KH*KW,
// KH*KW, and KW; each sample's output stride is OC*OH*OW. The explicit OC
// parameter is the validated output-channel extent supplied by the launcher.
// weight_index<weight_count handles tails; n,orow,ocol increase in CPU-reference
// order across the complete valid output domain. Each weight gradient has one
// gather owner, eliminating write races without barriers, atomic operations, or
// synchronization. Its serial FP32 reduction is deterministic in loop order
// and subject only to standard accumulation and contraction rounding.
__global__ void ConvolutionWeightGradientKernel(
    const float* input, const float* output_gradient, float* weight_gradient,
    std::size_t weight_count, int batch_size, int input_channels,
    int input_height, int input_width, int output_channels, int kernel_height,
    int kernel_width, int output_height, int output_width) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t input_plane =
      static_cast<std::size_t>(input_height) * input_width;
  const std::size_t input_sample =
      static_cast<std::size_t>(input_channels) * input_plane;
  const std::size_t output_plane =
      static_cast<std::size_t>(output_height) * output_width;
  const std::size_t output_sample =
      static_cast<std::size_t>(output_channels) * output_plane;
  const std::size_t kernel_plane =
      static_cast<std::size_t>(kernel_height) * kernel_width;
  const std::size_t output_channel_weights =
      static_cast<std::size_t>(input_channels) * kernel_plane;
  for (std::size_t weight_index = start; weight_index < weight_count;
       weight_index += stride) {
    const int output_channel =
        static_cast<int>(weight_index / output_channel_weights);
    const std::size_t within_output_channel =
        weight_index % output_channel_weights;
    const int input_channel =
        static_cast<int>(within_output_channel / kernel_plane);
    const std::size_t kernel_spatial =
        within_output_channel % kernel_plane;
    const int kernel_row =
        static_cast<int>(kernel_spatial / kernel_width);
    const int kernel_column =
        static_cast<int>(kernel_spatial % kernel_width);
    float sum = 0.0F;
    for (int sample = 0; sample < batch_size; ++sample) {
      const std::size_t input_channel_base =
          static_cast<std::size_t>(sample) * input_sample +
          static_cast<std::size_t>(input_channel) * input_plane;
      const std::size_t output_channel_base =
          static_cast<std::size_t>(sample) * output_sample +
          static_cast<std::size_t>(output_channel) * output_plane;
      for (int output_row = 0; output_row < output_height; ++output_row) {
        const std::size_t input_row_base =
            input_channel_base +
            static_cast<std::size_t>(output_row + kernel_row) * input_width;
        const std::size_t output_row_base =
            output_channel_base +
            static_cast<std::size_t>(output_row) * output_width;
        for (int output_column = 0; output_column < output_width;
             ++output_column) {
          sum += input[input_row_base + output_column + kernel_column] *
                 output_gradient[output_row_base + output_column];
        }
      }
    }
    weight_gradient[weight_index] = sum;
  }
}

// Computes dBias[oc]=sum_n,orow,ocol dOutput[n,oc,orow,ocol]. Threads
// grid-stride over oc; oc<output_channels handles launch tails, and the NCHW
// upstream index is (n*OC+oc)*OH*OW+orow*OW+ocol. n,orow,ocol increase in
// CPU-reference order. One thread owns each channel reduction, so there are no
// write races, barriers, atomic operations, or synchronization. Serial FP32
// addition is deterministic in order and has the expected reduction rounding.
__global__ void ConvolutionBiasGradientKernel(
    const float* output_gradient, float* bias_gradient, int batch_size,
    int output_channels, int output_height, int output_width) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t output_plane =
      static_cast<std::size_t>(output_height) * output_width;
  const std::size_t output_sample =
      static_cast<std::size_t>(output_channels) * output_plane;
  for (std::size_t output_channel = start;
       output_channel < static_cast<std::size_t>(output_channels);
       output_channel += stride) {
    float sum = 0.0F;
    for (int sample = 0; sample < batch_size; ++sample) {
      const std::size_t channel_base =
          static_cast<std::size_t>(sample) * output_sample +
          output_channel * output_plane;
      for (int output_row = 0; output_row < output_height; ++output_row) {
        const std::size_t row_base =
            channel_base +
            static_cast<std::size_t>(output_row) * output_width;
        for (int output_column = 0; output_column < output_width;
             ++output_column) {
          sum += output_gradient[row_base + output_column];
        }
      }
    }
    bias_gradient[output_channel] = sum;
  }
}

}  // namespace

void LaunchConvolutionForward(
    const float* input, const float* weight, const float* bias, float* output,
    int batch_size, int input_channels, int input_height, int input_width,
    int output_channels, int kernel_height, int kernel_width,
    cudaStream_t stream) {
  const ConvolutionShape shape = ValidateConvolutionShape(
      batch_size, input_channels, input_height, input_width, output_channels,
      kernel_height, kernel_width);
  ValidateBuffers({{input, shape.input_count * sizeof(float)},
                   {weight, shape.weight_count * sizeof(float)},
                   {bias, static_cast<std::size_t>(output_channels) *
                              sizeof(float)},
                   {output, shape.output_count * sizeof(float)}});
  ConvolutionForwardKernel<<<BlockCount(shape.output_count), kThreadsPerBlock,
                             0, stream>>>(
      input, weight, bias, output, shape.output_count, shape.input_channels,
      shape.input_height, shape.input_width, shape.output_channels,
      shape.kernel_height, shape.kernel_width, shape.output_height,
      shape.output_width);
  CUDA_KERNEL_CHECK();
}

void LaunchConvolutionBackward(
    const float* input, const float* weight, const float* output_gradient,
    float* input_gradient, float* weight_gradient, float* bias_gradient,
    int batch_size, int input_channels, int input_height, int input_width,
    int output_channels, int kernel_height, int kernel_width,
    cudaStream_t stream) {
  const ConvolutionShape shape = ValidateConvolutionShape(
      batch_size, input_channels, input_height, input_width, output_channels,
      kernel_height, kernel_width);
  ValidateBuffers({{input, shape.input_count * sizeof(float)},
                   {weight, shape.weight_count * sizeof(float)},
                   {output_gradient, shape.output_count * sizeof(float)},
                   {input_gradient, shape.input_count * sizeof(float)},
                   {weight_gradient, shape.weight_count * sizeof(float)},
                   {bias_gradient, static_cast<std::size_t>(output_channels) *
                                       sizeof(float)}});
  ConvolutionInputGradientKernel<<<BlockCount(shape.input_count),
                                   kThreadsPerBlock, 0, stream>>>(
      weight, output_gradient, input_gradient, shape.input_count,
      shape.input_channels, shape.input_height, shape.input_width,
      shape.output_channels, shape.kernel_height, shape.kernel_width,
      shape.output_height, shape.output_width);
  CUDA_KERNEL_CHECK();
  ConvolutionWeightGradientKernel<<<BlockCount(shape.weight_count),
                                    kThreadsPerBlock, 0, stream>>>(
      input, output_gradient, weight_gradient, shape.weight_count,
      shape.batch_size, shape.input_channels, shape.input_height,
      shape.input_width, shape.output_channels, shape.kernel_height,
      shape.kernel_width, shape.output_height, shape.output_width);
  CUDA_KERNEL_CHECK();
  ConvolutionBiasGradientKernel<<<BlockCount(shape.output_channels),
                                  kThreadsPerBlock, 0, stream>>>(
      output_gradient, bias_gradient, shape.batch_size, shape.output_channels,
      shape.output_height, shape.output_width);
  CUDA_KERNEL_CHECK();
}
