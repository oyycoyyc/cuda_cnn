#include "layers.h"

#include "cuda_check.h"
#include "random.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

constexpr int kImageSide = 28;
constexpr int kImagePixels = kImageSide * kImageSide;
constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::size_t kMaximumGridX = 2147483647ULL;

// Converts one packed uint8 image element to one NCHW FP32 element using
// (pixel/255-0.1307)/0.3081. A thread's linear output coordinate is
// blockIdx.x*blockDim.x+threadIdx.x; sample=linear/784,
// pixel=linear%784, y=pixel/28, and x=pixel%28. There are no loops: excess
// threads return at element_count, and valid threads read exactly one source
// pixel and write exactly one output. During augmentation, positive dx/dy moves
// content right/down, so output (x,y) samples source (x-dx,y-dy); out-of-bounds
// source coordinates supply uint8 zero before normalization. Offsets come from
// TranslationOffsetForSample(seed, epoch, original_indices[sample]). Distinct
// threads write distinct elements, so there are no races, synchronization, or
// atomics. Arithmetic is FP32 and deterministic per element; pixel-zero padding
// therefore becomes the normalized value of zero rather than FP32 zero.
__global__ void NormalizeTranslateKernel(
    const std::uint8_t* images, const std::uint32_t* original_indices,
    float* output, std::size_t element_count, std::uint64_t seed,
    std::uint32_t one_based_epoch, bool augment) {
  const std::size_t output_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= element_count) {
    return;
  }

  const std::size_t sample = output_index / kImagePixels;
  const int pixel_index = static_cast<int>(output_index % kImagePixels);
  const int output_y = pixel_index / kImageSide;
  const int output_x = pixel_index % kImageSide;
  TranslationOffset offset{0, 0};
  if (augment) {
    offset = TranslationOffsetForSample(seed, one_based_epoch,
                                        original_indices[sample]);
  }
  const int source_x = output_x - offset.dx;
  const int source_y = output_y - offset.dy;
  std::uint8_t pixel = 0;
  if (source_x >= 0 && source_x < kImageSide && source_y >= 0 &&
      source_y < kImageSide) {
    pixel = images[sample * kImagePixels + source_y * kImageSide + source_x];
  }
  output[output_index] =
      (static_cast<float>(pixel) / 255.0F - 0.1307F) / 0.3081F;
}

std::size_t InputElementCount(int batch_size) {
  if (batch_size <= 0) {
    throw std::invalid_argument("input batch_size must be positive");
  }
  const std::size_t batch = static_cast<std::size_t>(batch_size);
  if (batch > std::numeric_limits<std::size_t>::max() / kImagePixels) {
    throw std::overflow_error("input dimensions overflow size_t");
  }
  const std::size_t count = batch * kImagePixels;
  const std::size_t blocks =
      1 + (count - 1) / static_cast<std::size_t>(kThreadsPerBlock);
  if (blocks > kMaximumGridX) {
    throw std::overflow_error("input dimensions exceed CUDA grid capacity");
  }
  return count;
}

}  // namespace

void LaunchNormalizeTranslate(const std::uint8_t* images,
                              const std::uint32_t* original_indices,
                              float* output, int batch_size,
                              std::uint64_t seed,
                              std::uint32_t one_based_epoch, bool augment,
                              cudaStream_t stream) {
  const std::size_t count = InputElementCount(batch_size);
  if (augment && one_based_epoch == 0) {
    throw std::invalid_argument(
        "input one_based_epoch must be positive when augmenting");
  }
  const unsigned int blocks = static_cast<unsigned int>(
      1 + (count - 1) / static_cast<std::size_t>(kThreadsPerBlock));
  NormalizeTranslateKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      images, original_indices, output, count, seed, one_based_epoch, augment);
  CUDA_KERNEL_CHECK();
}
