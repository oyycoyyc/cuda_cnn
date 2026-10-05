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

// Validated pooling geometry and the derived element counts.
struct PoolShape {
  int input_height;
  int input_width;
  int output_height;
  int output_width;
  std::size_t input_count;
  std::size_t output_count;
};

// Multiplies dimensions, failing closed while detecting size_t overflow.
std::size_t CheckedProduct(std::initializer_list<int> dimensions) {
  std::size_t product = 1;
  for (int dimension : dimensions) {
    const std::size_t value = static_cast<std::size_t>(dimension);
    if (product > std::numeric_limits<std::size_t>::max() / value) {
      throw std::overflow_error("max pool dimensions overflow size_t");
    }
    product *= value;
  }
  return product;
}

// Validates positive even spatial sizes of at least two and derives the output
// shape and counts.
PoolShape ValidatePoolShape(int batch_size, int channels, int input_height,
                            int input_width) {
  if (batch_size <= 0 || channels <= 0 || input_height < 2 ||
      input_width < 2 || input_height % 2 != 0 || input_width % 2 != 0) {
    throw std::invalid_argument(
        "max pool dimensions must be positive with even spatial sizes >= 2");
  }
  const int output_height = input_height / 2;
  const int output_width = input_width / 2;
  const std::size_t input_count =
      CheckedProduct({batch_size, channels, input_height, input_width});
  const std::size_t output_count =
      CheckedProduct({batch_size, channels, output_height, output_width});
  if (input_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      output_count >
          std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error("max pool dimensions overflow byte extent");
  }
  return PoolShape{input_height, input_width, output_height, output_width,
                   input_count, output_count};
}

// Returns a capped grid-stride block count sufficient to cover count elements.
unsigned int BlockCount(std::size_t count) {
  const std::size_t needed =
      1 + (count - 1) / static_cast<std::size_t>(kThreadsPerBlock);
  return static_cast<unsigned int>(std::min(needed, kMaximumBlocks));
}

// Computes 2x2 stride-two NCHW max pooling and records a row-major uint8 winner
// offset. A thread starts at blockIdx.x*blockDim.x+threadIdx.x and loops by
// blockDim.x*gridDim.x over output linear indices. For output plane size
// OH*OW, group=linear/(OH*OW), oy=(linear%(OH*OW))/OW, ox=linear%OW, and the
// input base is group*(IH*IW)+(2*oy)*IW+2*ox. Offsets 0,1,2,3 map by
// row=offset/2,column=offset%2 and are compared in that order using strict >,
// so ties retain the smallest offset. index<output_count handles all tails and
// loop termination. Every output/value-offset pair has one writer and windows
// only read input, so there are no races, barriers, synchronization, or atomics.
// Values are copied without arithmetic; strict comparisons make ties stable,
// while NaN behavior follows ordinary unordered FP32 comparisons.
__global__ void MaxPoolForwardKernel(const float* input, float* output,
                                     std::uint8_t* winner_offsets,
                                     std::size_t output_count,
                                     int input_height, int input_width,
                                     int output_height, int output_width) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t output_plane =
      static_cast<std::size_t>(output_height) * output_width;
  const std::size_t input_plane =
      static_cast<std::size_t>(input_height) * input_width;
  for (std::size_t output_index = start; output_index < output_count;
       output_index += stride) {
    const std::size_t group = output_index / output_plane;
    const std::size_t spatial = output_index % output_plane;
    const int output_y = static_cast<int>(spatial / output_width);
    const int output_x = static_cast<int>(spatial % output_width);
    const std::size_t input_base =
        group * input_plane + static_cast<std::size_t>(output_y * 2) *
                                  input_width +
        output_x * 2;
    float maximum = input[input_base];
    std::uint8_t winner = 0;
    for (std::uint8_t offset = 1; offset < 4; ++offset) {
      const float candidate =
          input[input_base + static_cast<std::size_t>(offset / 2) *
                                 input_width +
                offset % 2];
      if (candidate > maximum) {
        maximum = candidate;
        winner = offset;
      }
    }
    output[output_index] = maximum;
    winner_offsets[output_index] = winner;
  }
}

// Scatters each NCHW output gradient to its recorded 2x2 winner after the
// launcher has queued zeroing of the full input gradient on the same stream. A
// thread starts at blockIdx.x*blockDim.x+threadIdx.x and loops by
// blockDim.x*gridDim.x over output linear indices. group, oy, ox, and input_base
// use the same formulas as forward; winner row=offset/2 and column=offset%2.
// index<output_count handles tails and loop termination. Valid offsets and
// non-overlapping stride-two windows guarantee one scatter writer per input
// location, so no atomics, barriers, or intra-kernel synchronization are used;
// stream ordering makes zeroing happen first. Gradients are copied without
// arithmetic, and all non-winner positions remain exact +0.0F.
__global__ void MaxPoolBackwardKernel(
    const float* output_gradient, const std::uint8_t* winner_offsets,
    float* input_gradient, std::size_t output_count, int input_height,
    int input_width, int output_height, int output_width) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t output_plane =
      static_cast<std::size_t>(output_height) * output_width;
  const std::size_t input_plane =
      static_cast<std::size_t>(input_height) * input_width;
  for (std::size_t output_index = start; output_index < output_count;
       output_index += stride) {
    const std::size_t group = output_index / output_plane;
    const std::size_t spatial = output_index % output_plane;
    const int output_y = static_cast<int>(spatial / output_width);
    const int output_x = static_cast<int>(spatial % output_width);
    const std::uint8_t offset = winner_offsets[output_index];
    const std::size_t input_index =
        group * input_plane + static_cast<std::size_t>(output_y * 2) *
                                  input_width +
        output_x * 2 + static_cast<std::size_t>(offset / 2) * input_width +
        offset % 2;
    input_gradient[input_index] = output_gradient[output_index];
  }
}

}  // namespace

// Launches 2x2 stride-two max pooling over the validated output extent.
void LaunchMaxPoolForward(const float* input, float* output,
                          std::uint8_t* winner_offsets, int batch_size,
                          int channels, int input_height, int input_width,
                          cudaStream_t stream) {
  const PoolShape shape =
      ValidatePoolShape(batch_size, channels, input_height, input_width);
  MaxPoolForwardKernel<<<BlockCount(shape.output_count), kThreadsPerBlock, 0,
                         stream>>>(input, output, winner_offsets,
                                   shape.output_count, shape.input_height,
                                   shape.input_width, shape.output_height,
                                   shape.output_width);
  CUDA_KERNEL_CHECK();
}

// Zeroes the input gradient on the stream, then scatters each output gradient to
// its recorded winner.
void LaunchMaxPoolBackward(const float* output_gradient,
                           const std::uint8_t* winner_offsets,
                           float* input_gradient, int batch_size, int channels,
                           int input_height, int input_width,
                           cudaStream_t stream) {
  const PoolShape shape =
      ValidatePoolShape(batch_size, channels, input_height, input_width);
  LaunchZero(input_gradient, shape.input_count, stream);
  MaxPoolBackwardKernel<<<BlockCount(shape.output_count), kThreadsPerBlock, 0,
                          stream>>>(output_gradient, winner_offsets,
                                    input_gradient, shape.output_count,
                                    shape.input_height, shape.input_width,
                                    shape.output_height, shape.output_width);
  CUDA_KERNEL_CHECK();
}
