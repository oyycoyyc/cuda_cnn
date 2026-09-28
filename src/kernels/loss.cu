#include "layers.h"

#include "cuda_check.h"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace {

constexpr int kThreadsPerBlock = 256;

struct LossShape {
  int batch_size;
  int class_count;
  std::size_t element_count;
};

struct BufferRange {
  const void* pointer;
  std::size_t bytes;
};

LossShape ValidateLossShape(int batch_size, int class_count) {
  if (batch_size <= 0 || class_count <= 0) {
    throw std::invalid_argument("softmax dimensions must be positive");
  }
  if (class_count > kThreadsPerBlock) {
    throw std::invalid_argument("softmax class_count must not exceed 256");
  }
  const std::size_t batch = static_cast<std::size_t>(batch_size);
  const std::size_t classes = static_cast<std::size_t>(class_count);
  if (batch > std::numeric_limits<std::size_t>::max() / classes) {
    throw std::overflow_error("softmax dimensions overflow size_t");
  }
  const std::size_t element_count = batch * classes;
  if (element_count >
          std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      batch > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error("softmax dimensions overflow byte extent");
  }
  return LossShape{batch_size, class_count, element_count};
}

void ValidateBuffers(std::initializer_list<BufferRange> buffers) {
  for (const BufferRange& buffer : buffers) {
    if (buffer.pointer == nullptr) {
      throw std::invalid_argument("softmax pointer must be non-null");
    }
  }
  for (auto left = buffers.begin(); left != buffers.end(); ++left) {
    const std::uintptr_t left_begin =
        reinterpret_cast<std::uintptr_t>(left->pointer);
    if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left->bytes) {
      throw std::invalid_argument("softmax pointer range is invalid");
    }
    const std::uintptr_t left_end = left_begin + left->bytes;
    for (auto right = left + 1; right != buffers.end(); ++right) {
      const std::uintptr_t right_begin =
          reinterpret_cast<std::uintptr_t>(right->pointer);
      if (right_begin >
          std::numeric_limits<std::uintptr_t>::max() - right->bytes) {
        throw std::invalid_argument("softmax pointer range is invalid");
      }
      const std::uintptr_t right_end = right_begin + right->bytes;
      if (left_begin < right_end && right_begin < left_end) {
        throw std::invalid_argument("softmax buffers overlap");
      }
    }
  }
}

// One block owns row sample=blockIdx.x. shared_values[0..255] first contains
// logits (unused lanes hold -FLT_MAX), then exponentials (unused lanes hold
// zero). After every write and every stride-128..1 tree step, __syncthreads
// makes the next readers safe; no lane exits early. The first tree finds max,
// so expf(logit-max) lies in [0,1], at least one term is one, and division by
// the positive sum produces finite probabilities even for logits near +/-1000.
// Each valid lane writes its probability and
// d(mean CE)/dlogit=(probability-one_hot)/batch_size exactly once. Lane zero
// writes one loss as log(sum)+max-logit[label], avoiding -log(0) when the label
// probability underflows. Blocks touch disjoint rows, so no atomics or races
// occur; the two trees have fixed pairing and ordinary deterministic FP32 error.
__global__ void SoftmaxCrossEntropyKernel(
    const float* logits, const std::uint8_t* labels, float* probabilities,
    float* per_sample_losses, float* logits_gradient, int class_count,
    float inverse_batch_size) {
  __shared__ float shared_values[kThreadsPerBlock];
  const int sample = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const std::size_t row =
      static_cast<std::size_t>(sample) * class_count;
  const bool valid_class = lane < class_count;
  const float logit = valid_class ? logits[row + lane] : -FLT_MAX;
  shared_values[lane] = logit;
  __syncthreads();
  for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
    if (lane < stride) {
      shared_values[lane] =
          fmaxf(shared_values[lane], shared_values[lane + stride]);
    }
    __syncthreads();
  }
  const float maximum = shared_values[0];
  const float exponential = valid_class ? expf(logit - maximum) : 0.0F;
  shared_values[lane] = exponential;
  __syncthreads();
  for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
    if (lane < stride) {
      shared_values[lane] += shared_values[lane + stride];
    }
    __syncthreads();
  }
  const float sum = shared_values[0];
  if (valid_class) {
    const float probability = exponential / sum;
    probabilities[row + lane] = probability;
    const float target = lane == static_cast<int>(labels[sample]) ? 1.0F : 0.0F;
    logits_gradient[row + lane] =
        (probability - target) * inverse_batch_size;
  }
  if (lane == 0) {
    per_sample_losses[sample] =
        logf(sum) + maximum - logits[row + labels[sample]];
  }
}

// One block owns one inference row. shared_values[0..255] is reused first for
// max reduction and then sum reduction; inactive class lanes contribute
// -FLT_MAX and zero respectively. Every stage is followed by __syncthreads, so
// no shared read races the preceding writes. Max subtraction bounds expf inputs
// above by zero and guarantees sum>=1 for finite logits, making every output
// finite. Each valid lane writes one class probability; rows are disjoint, so
// no atomics or cross-block synchronization are needed. There is deliberately
// no label input or label-dependent branch in this inference kernel.
__global__ void SoftmaxKernel(const float* logits, float* probabilities,
                              int class_count) {
  __shared__ float shared_values[kThreadsPerBlock];
  const int sample = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const std::size_t row =
      static_cast<std::size_t>(sample) * class_count;
  const bool valid_class = lane < class_count;
  const float logit = valid_class ? logits[row + lane] : -FLT_MAX;
  shared_values[lane] = logit;
  __syncthreads();
  for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
    if (lane < stride) {
      shared_values[lane] =
          fmaxf(shared_values[lane], shared_values[lane + stride]);
    }
    __syncthreads();
  }
  const float maximum = shared_values[0];
  const float exponential = valid_class ? expf(logit - maximum) : 0.0F;
  shared_values[lane] = exponential;
  __syncthreads();
  for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
    if (lane < stride) {
      shared_values[lane] += shared_values[lane + stride];
    }
    __syncthreads();
  }
  if (valid_class) {
    probabilities[row + lane] = exponential / shared_values[0];
  }
}

// The sole thread consumes losses[0..batch_size-1] in ascending order, writes
// only mean_loss[0], and divides exactly once after the fixed-order FP32 sum.
// There is no shared memory, barrier, atomic, race, or inter-block ordering: the
// one-thread/one-block boundary intentionally trades parallelism for a fully
// specified deterministic reduction. Inputs are finite by the preceding stable
// CE kernel, with ordinary serial FP32 accumulation and division rounding.
__global__ void MeanLossKernel(const float* per_sample_losses,
                               float* mean_loss, int batch_size) {
  float sum = 0.0F;
  for (int sample = 0; sample < batch_size; ++sample) {
    sum += per_sample_losses[sample];
  }
  mean_loss[0] = sum / static_cast<float>(batch_size);
}

}  // namespace

void LaunchSoftmaxCrossEntropy(
    const float* logits, const std::uint8_t* labels, float* probabilities,
    float* per_sample_losses, float* mean_loss, float* logits_gradient,
    int batch_size, int class_count, cudaStream_t stream) {
  const LossShape shape = ValidateLossShape(batch_size, class_count);
  const std::size_t matrix_bytes = shape.element_count * sizeof(float);
  const std::size_t batch_float_bytes =
      static_cast<std::size_t>(batch_size) * sizeof(float);
  ValidateBuffers({{logits, matrix_bytes},
                   {labels, static_cast<std::size_t>(batch_size)},
                   {probabilities, matrix_bytes},
                   {per_sample_losses, batch_float_bytes},
                   {mean_loss, sizeof(float)},
                   {logits_gradient, matrix_bytes}});
  const float inverse_batch_size = 1.0F / static_cast<float>(batch_size);
  SoftmaxCrossEntropyKernel<<<static_cast<unsigned int>(batch_size),
                              kThreadsPerBlock, 0, stream>>>(
      logits, labels, probabilities, per_sample_losses, logits_gradient,
      class_count, inverse_batch_size);
  CUDA_KERNEL_CHECK();
  MeanLossKernel<<<1, 1, 0, stream>>>(per_sample_losses, mean_loss, batch_size);
  CUDA_KERNEL_CHECK();
}

void LaunchSoftmax(const float* logits, float* probabilities, int batch_size,
                   int class_count, cudaStream_t stream) {
  const LossShape shape = ValidateLossShape(batch_size, class_count);
  const std::size_t matrix_bytes = shape.element_count * sizeof(float);
  ValidateBuffers(
      {{logits, matrix_bytes}, {probabilities, matrix_bytes}});
  SoftmaxKernel<<<static_cast<unsigned int>(batch_size), kThreadsPerBlock, 0,
                  stream>>>(logits, probabilities, class_count);
  CUDA_KERNEL_CHECK();
}
