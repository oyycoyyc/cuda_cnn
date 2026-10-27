// Implements the course GPU helpers: FP32-only input preparation with the
// original deterministic translation, and a label-free int32 argmax with a
// sticky non-finite diagnostic flag.
#include "course/gpu_ops.h"

#include "cuda_check.h"
#include "random.h"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

// Fixed course geometry, launch shape, and grid capacity.
constexpr int kImageSide = 28;
constexpr int kImagePixels = kImageSide * kImageSide;
constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::size_t kMaximumGridX = 2147483647ULL;

// Fixed normalization constants from the model contract.
constexpr float kNormalizationMean = 0.1307F;
constexpr float kNormalizationStddev = 0.3081F;

// kUnit transform evaluated with the exact documented FP32 operation order.
__device__ float NormalizeUnit(float value) {
  return (value - kNormalizationMean) / kNormalizationStddev;
}

// kByte transform evaluated with the exact documented FP32 operation order;
// no uint8 conversion is performed at any point.
__device__ float NormalizeByte(float value) {
  return (value / 255.0F - kNormalizationMean) / kNormalizationStddev;
}

// Translation padding is the normalized form of a black (zero) pixel in every
// input mode, including kNormalized.
__device__ float NormalizedBlackPixel() { return NormalizeUnit(0.0F); }

// Device finiteness test without relying on device standard-library support.
__device__ bool IsFinite(float value) {
  return value == value && value <= FLT_MAX && value >= -FLT_MAX;
}

// Converts one packed image element per thread. Sample and pixel geometry
// match the original input kernel; translation uses the original helper and
// positive dx/dy move content right/down. No mode writes through a uint8 or any
// other quantized representation.
__global__ void PrepareInputKernel(
    const float* packed, const std::uint32_t* original_indices,
    float* normalized, std::size_t element_count, course::InputMode mode,
    std::uint64_t seed, std::uint32_t epoch, bool augment) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }

  const std::size_t sample = index / kImagePixels;
  const int pixel_index = static_cast<int>(index % kImagePixels);
  const int output_y = pixel_index / kImageSide;
  const int output_x = pixel_index % kImageSide;
  int source_x = output_x;
  int source_y = output_y;
  if (augment) {
    const TranslationOffset offset = TranslationOffsetForSample(
        seed, epoch, original_indices[sample]);
    source_x -= offset.dx;
    source_y -= offset.dy;
  }
  if (source_x < 0 || source_x >= kImageSide || source_y < 0 ||
      source_y >= kImageSide) {
    normalized[index] = NormalizedBlackPixel();
    return;
  }

  const float source =
      packed[sample * kImagePixels + static_cast<std::size_t>(source_y) *
                                         kImageSide +
             static_cast<std::size_t>(source_x)];
  float result = source;
  if (mode == course::InputMode::kUnit) {
    result = NormalizeUnit(source);
  } else if (mode == course::InputMode::kByte) {
    result = NormalizeByte(source);
  }
  normalized[index] = result;
}

// Writes one smallest-index argmax per row, scans every logit of every row for
// NaN/infinity, and sets (never clears) the sticky scalar flag.
__global__ void ArgmaxKernel(const float* logits, std::int32_t* predictions,
                             int* bad_output, int batch, int classes) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= batch) {
    return;
  }

  const float* const row_logits =
      logits + static_cast<std::size_t>(row) * static_cast<std::size_t>(classes);
  float best = row_logits[0];
  int best_index = 0;
  bool nonfinite = !IsFinite(best);
  for (int class_index = 1; class_index < classes; ++class_index) {
    const float value = row_logits[class_index];
    if (!IsFinite(value)) {
      nonfinite = true;
    }
    // Strict comparison keeps the smallest index on ties.
    if (value > best) {
      best = value;
      best_index = class_index;
    }
  }
  predictions[row] = static_cast<std::int32_t>(best_index);
  if (nonfinite) {
    // Rows are independent threads, so several may update this shared word in
    // one launch. The exchange is atomic: a defined concurrent same-address
    // update instead of a plain-store data race. Every writer stores the same
    // value and nothing here ever clears it, preserving the caller-zeroed
    // sticky contract.
    atomicExch(bad_output, 1);
  }
}

// Adds two sizes, failing closed instead of wrapping.
std::size_t CheckedAdd(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error("course GPU extent overflows size_t");
  }
  return left + right;
}

// Multiplies two sizes, rejecting overflow before it can wrap.
std::size_t CheckedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("course GPU extent overflows size_t");
  }
  return left * right;
}

// Validates the positive batch and returns the packed element count, rejecting
// images that do not fit a single CUDA x-grid.
std::size_t PreparedElementCount(int batch) {
  if (batch <= 0) {
    throw std::invalid_argument("prepare input batch must be positive");
  }
  const std::size_t count = CheckedMultiply(
      static_cast<std::size_t>(batch), static_cast<std::size_t>(kImagePixels));
  const std::size_t blocks =
      1 + (count - 1) / static_cast<std::size_t>(kThreadsPerBlock);
  if (blocks > kMaximumGridX) {
    throw std::overflow_error("prepare input exceeds CUDA grid capacity");
  }
  return count;
}

// Returns whether mode is one of the three supported enumerators.
bool IsSupportedMode(course::InputMode mode) {
  return mode == course::InputMode::kUnit ||
         mode == course::InputMode::kByte ||
         mode == course::InputMode::kNormalized;
}

// Returns whether [a, a+a_bytes) and [b, b+b_bytes) intersect. All extents are
// nonzero at each call site.
bool RangesOverlap(const void* a, std::size_t a_bytes, const void* b,
                   std::size_t b_bytes) {
  const std::uintptr_t a_begin = reinterpret_cast<std::uintptr_t>(a);
  const std::uintptr_t b_begin = reinterpret_cast<std::uintptr_t>(b);
  return a_begin < CheckedAdd(b_begin, b_bytes) &&
         b_begin < CheckedAdd(a_begin, a_bytes);
}

}  // namespace

// Validates the host contract and enqueues one element-per-thread conversion.
void course::LaunchPrepareInput(const float* packed,
                                const std::uint32_t* original_indices,
                                float* normalized, int batch,
                                course::InputMode mode, std::uint64_t seed,
                                std::uint32_t epoch, bool augment,
                                cudaStream_t stream) {
  const std::size_t element_count = PreparedElementCount(batch);
  if (!IsSupportedMode(mode)) {
    throw std::invalid_argument("prepare input mode is not supported");
  }
  if (packed == nullptr) {
    throw std::invalid_argument("prepare input packed must be non-null");
  }
  if (normalized == nullptr) {
    throw std::invalid_argument("prepare input normalized must be non-null");
  }
  if (augment && original_indices == nullptr) {
    throw std::invalid_argument(
        "prepare input original_indices must be non-null when augmenting");
  }
  if (augment && epoch == 0) {
    throw std::invalid_argument(
        "prepare input epoch must be positive when augmenting");
  }
  const std::size_t element_bytes =
      CheckedMultiply(element_count, sizeof(float));
  if (RangesOverlap(packed, element_bytes, normalized, element_bytes)) {
    throw std::invalid_argument(
        "prepare input packed and normalized must not overlap");
  }

  const unsigned int blocks = static_cast<unsigned int>(
      1 + (element_count - 1) / static_cast<std::size_t>(kThreadsPerBlock));
  PrepareInputKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      packed, original_indices, normalized, element_count, mode, seed, epoch,
      augment);
  CUDA_KERNEL_CHECK();
}

// Validates the host contract and enqueues one argmax thread per row.
void course::LaunchArgmax(const float* logits, std::int32_t* predictions,
                          int* bad_output, int batch, int classes,
                          cudaStream_t stream) {
  if (batch <= 0) {
    throw std::invalid_argument("argmax batch must be positive");
  }
  if (classes < 1 || classes > 256) {
    throw std::invalid_argument("argmax classes must be in [1, 256]");
  }
  if (logits == nullptr) {
    throw std::invalid_argument("argmax logits must be non-null");
  }
  if (predictions == nullptr) {
    throw std::invalid_argument("argmax predictions must be non-null");
  }
  if (bad_output == nullptr) {
    throw std::invalid_argument("argmax bad_output must be non-null");
  }

  const std::size_t batch_count = static_cast<std::size_t>(batch);
  const std::size_t logit_bytes = CheckedMultiply(
      CheckedMultiply(batch_count, static_cast<std::size_t>(classes)),
      sizeof(float));
  const std::size_t prediction_bytes =
      CheckedMultiply(batch_count, sizeof(std::int32_t));
  const std::size_t flag_bytes = sizeof(int);
  if (RangesOverlap(logits, logit_bytes, predictions, prediction_bytes) ||
      RangesOverlap(logits, logit_bytes, bad_output, flag_bytes) ||
      RangesOverlap(predictions, prediction_bytes, bad_output, flag_bytes)) {
    throw std::invalid_argument(
        "argmax logits, predictions, and bad_output must be disjoint");
  }

  const unsigned int blocks = static_cast<unsigned int>(
      1 + (batch_count - 1) / static_cast<std::size_t>(kThreadsPerBlock));
  ArgmaxKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      logits, predictions, bad_output, batch, classes);
  CUDA_KERNEL_CHECK();
}
