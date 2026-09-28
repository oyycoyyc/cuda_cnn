#include "layers.h"

#include "cuda_check.h"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace {

constexpr int kThreadsPerBlock = 256;

struct MetricsShape {
  int batch_size;
  int class_count;
  std::size_t logits_count;
};

struct BufferRange {
  const void* pointer;
  std::size_t bytes;
};

MetricsShape ValidateMetricsShape(int batch_size, int class_count) {
  if (batch_size <= 0 || class_count <= 0) {
    throw std::invalid_argument("metrics dimensions must be positive");
  }
  if (class_count > kThreadsPerBlock) {
    throw std::invalid_argument("metrics class_count must not exceed 256");
  }
  const std::size_t batch = static_cast<std::size_t>(batch_size);
  const std::size_t classes = static_cast<std::size_t>(class_count);
  if (batch > std::numeric_limits<std::size_t>::max() / classes) {
    throw std::overflow_error("metrics dimensions overflow size_t");
  }
  const std::size_t logits_count = batch * classes;
  if (logits_count >
          std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      batch > std::numeric_limits<std::size_t>::max() / sizeof(int)) {
    throw std::overflow_error("metrics dimensions overflow byte extent");
  }
  return MetricsShape{batch_size, class_count, logits_count};
}

void ValidateBuffers(std::initializer_list<BufferRange> buffers) {
  for (const BufferRange& buffer : buffers) {
    if (buffer.pointer == nullptr) {
      throw std::invalid_argument("metrics pointer must be non-null");
    }
  }
  for (auto left = buffers.begin(); left != buffers.end(); ++left) {
    const std::uintptr_t left_begin =
        reinterpret_cast<std::uintptr_t>(left->pointer);
    if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left->bytes) {
      throw std::invalid_argument("metrics pointer range is invalid");
    }
    const std::uintptr_t left_end = left_begin + left->bytes;
    for (auto right = left + 1; right != buffers.end(); ++right) {
      const std::uintptr_t right_begin =
          reinterpret_cast<std::uintptr_t>(right->pointer);
      if (right_begin >
          std::numeric_limits<std::uintptr_t>::max() - right->bytes) {
        throw std::invalid_argument("metrics pointer range is invalid");
      }
      const std::uintptr_t right_end = right_begin + right->bytes;
      if (left_begin < right_end && right_begin < left_end) {
        throw std::invalid_argument("metrics buffers overlap");
      }
    }
  }
}

// One block owns sample=blockIdx.x. shared_values[0..255] and
// shared_indices[0..255] are parallel candidate arrays; inactive lanes use
// (-FLT_MAX,class_count), whose out-of-range index loses ties to every valid
// class. Each stride-128..1 comparison keeps the larger value, or the smaller
// index for equal values, and __syncthreads after initialization and every stage
// prevents shared read/write races. Lane zero alone writes the uint8 prediction
// and exact 0/1 correct flag. Rows are disjoint and inputs immutable, so no
// atomics or cross-block synchronization occur; comparison performs no rounding.
__global__ void ArgmaxFlagsKernel(
    const float* logits, const std::uint8_t* labels,
    std::uint8_t* predictions, int* correct_flags, int class_count) {
  __shared__ float shared_values[kThreadsPerBlock];
  __shared__ int shared_indices[kThreadsPerBlock];
  const int sample = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const bool valid_class = lane < class_count;
  const std::size_t row =
      static_cast<std::size_t>(sample) * class_count;
  shared_values[lane] = valid_class ? logits[row + lane] : -FLT_MAX;
  shared_indices[lane] = valid_class ? lane : class_count;
  __syncthreads();
  for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
    if (lane < stride) {
      const float right_value = shared_values[lane + stride];
      const int right_index = shared_indices[lane + stride];
      const float left_value = shared_values[lane];
      const int left_index = shared_indices[lane];
      if (right_value > left_value ||
          (right_value == left_value && right_index < left_index)) {
        shared_values[lane] = right_value;
        shared_indices[lane] = right_index;
      }
    }
    __syncthreads();
  }
  if (lane == 0) {
    const int prediction = shared_indices[0];
    predictions[sample] = static_cast<std::uint8_t>(prediction);
    correct_flags[sample] =
        prediction == static_cast<int>(labels[sample]) ? 1 : 0;
  }
}

// The only thread reads correct_flags in ascending sample order and writes the
// scalar count once. No shared memory, barriers, atomics, races, or inter-block
// ordering exist because this fixed-order project-owned reduction is exactly
// one block and one thread. Each input is 0/1 and batch_size<=INT_MAX, so the
// integer sum is exact and cannot overflow; no floating-point error is involved.
__global__ void CorrectCountKernel(const int* correct_flags,
                                   int* correct_count, int batch_size) {
  int sum = 0;
  for (int sample = 0; sample < batch_size; ++sample) {
    sum += correct_flags[sample];
  }
  correct_count[0] = sum;
}

}  // namespace

void LaunchArgmaxAndCountCorrect(
    const float* logits, const std::uint8_t* labels,
    std::uint8_t* predictions, int* correct_flags, int* correct_count,
    int batch_size, int class_count, cudaStream_t stream) {
  const MetricsShape shape = ValidateMetricsShape(batch_size, class_count);
  const std::size_t batch = static_cast<std::size_t>(batch_size);
  ValidateBuffers({{logits, shape.logits_count * sizeof(float)},
                   {labels, batch},
                   {predictions, batch},
                   {correct_flags, batch * sizeof(int)},
                   {correct_count, sizeof(int)}});
  ArgmaxFlagsKernel<<<static_cast<unsigned int>(batch_size), kThreadsPerBlock,
                      0, stream>>>(logits, labels, predictions, correct_flags,
                                  class_count);
  CUDA_KERNEL_CHECK();
  CorrectCountKernel<<<1, 1, 0, stream>>>(correct_flags, correct_count,
                                          batch_size);
  CUDA_KERNEL_CHECK();
}
