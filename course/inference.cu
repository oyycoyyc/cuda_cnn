// Implements the forward-only course inference model: one checked, aligned
// fixed arena holding the ten supplied parameter payloads, forward activations,
// and pooling winners, plus the original LeNet forward launch chain.
#include "course/inference.h"

#include "cuda_check.h"
#include "layers.h"
#include "tensor.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Arena alignment keeps every float region naturally aligned; cudaMalloc's base
// pointer is already more strictly aligned.
constexpr std::size_t kAlignment = 16;

// Forward activation shapes used by the original LeNet chain. ReLU runs
// in-place on its convolution/linear output, which the original launcher
// contract explicitly permits, so only pre-ReLU storage is reserved.
constexpr std::size_t kConv1FloatsPerSample = 3456;
constexpr std::size_t kPool1FloatsPerSample = 864;
constexpr std::size_t kConv2FloatsPerSample = 1024;
constexpr std::size_t kPool2FloatsPerSample = 256;
constexpr std::size_t kFc1FloatsPerSample = 120;
constexpr std::size_t kFc2FloatsPerSample = 84;
constexpr std::size_t kLogitsFloatsPerSample = 10;
constexpr std::size_t kPool1WinnerBytesPerSample = 864;
constexpr std::size_t kPool2WinnerBytesPerSample = 256;

// Adds two sizes, failing closed instead of wrapping.
std::size_t CheckedAdd(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error("InferenceModel device bytes overflow size_t");
  }
  return left + right;
}

// Multiplies two sizes, rejecting overflow before it can wrap.
std::size_t CheckedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("InferenceModel device bytes overflow size_t");
  }
  return left * right;
}

// Aligns one arena offset to the required boundary.
std::size_t AlignUp(std::size_t offset) {
  const std::size_t remainder = offset % kAlignment;
  return remainder == 0 ? offset : CheckedAdd(offset, kAlignment - remainder);
}

// Returns the offset after count FP32 elements, with region alignment.
std::size_t AfterFloats(std::size_t offset, std::size_t count) {
  return CheckedAdd(AlignUp(offset), CheckedMultiply(count, sizeof(float)));
}

// Returns the offset after count bytes, with region alignment.
std::size_t AfterBytes(std::size_t offset, std::size_t count) {
  return CheckedAdd(AlignUp(offset), count);
}

// Converts a positive capacity and per-sample extent into a total count.
std::size_t Count(int maximum_batch, std::size_t per_sample) {
  return CheckedMultiply(static_cast<std::size_t>(maximum_batch), per_sample);
}

// Computes the exact arena size for a capacity, mirroring PlanArena's order.
std::size_t RequiredBytes(int maximum_batch) {
  if (maximum_batch <= 0) {
    throw std::invalid_argument(
        "InferenceModel maximum_batch must be positive");
  }
  std::size_t offset = 0;
  const auto& specs = LenetParameterSpecs();
  for (std::size_t index = 0; index < specs.size(); ++index) {
    offset = AfterFloats(
        offset, static_cast<std::size_t>(specs[index].element_count));
  }
  offset = AfterFloats(offset, Count(maximum_batch, kConv1FloatsPerSample));
  offset = AfterFloats(offset, Count(maximum_batch, kPool1FloatsPerSample));
  offset = AfterFloats(offset, Count(maximum_batch, kConv2FloatsPerSample));
  offset = AfterFloats(offset, Count(maximum_batch, kPool2FloatsPerSample));
  offset = AfterFloats(offset, Count(maximum_batch, kFc1FloatsPerSample));
  offset = AfterFloats(offset, Count(maximum_batch, kFc2FloatsPerSample));
  offset = AfterFloats(offset, Count(maximum_batch, kLogitsFloatsPerSample));
  offset = AfterBytes(offset, Count(maximum_batch, kPool1WinnerBytesPerSample));
  offset = AfterBytes(offset, Count(maximum_batch, kPool2WinnerBytesPerSample));
  return offset;
}

}  // namespace

namespace course {

class InferenceModel::Impl {
 public:
  // Validates capacity, schema, and finiteness before any device allocation,
  // then reserves one arena, lays out its regions, and performs the single
  // synchronized weight upload.
  Impl(int maximum_batch, const ParameterSet& parameters, cudaStream_t stream)
      : maximum_batch_(maximum_batch),
        stream_(stream),
        required_bytes_(RequiredBytes(maximum_batch)) {
    ValidateLenetParameters(parameters);
    ValidateFiniteParameters(parameters);

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(
        device_buffer_detail::QueryMemoryInfo(&free_bytes, &total_bytes));
    if (free_bytes < required_bytes_) {
      std::ostringstream message;
      message << "insufficient CUDA memory: required_bytes=" << required_bytes_
              << " free_bytes=" << free_bytes
              << " total_bytes=" << total_bytes;
      throw std::runtime_error(message.str());
    }

    // One allocation; the arena is never zeroed or reallocated.
    arena_ = DeviceBuffer<std::uint8_t>(required_bytes_);
    PlanArena();
    CopyParametersToDevice(parameters);
    // One synchronization makes the uploaded weights immediately usable.
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  // Enqueues the original forward-only chain and returns the stable logits.
  const float* Forward(const float* normalized, int batch) {
    if (batch <= 0) {
      throw std::invalid_argument("Forward batch must be positive");
    }
    if (batch > maximum_batch_) {
      throw std::invalid_argument("Forward batch exceeds maximum_batch");
    }
    if (normalized == nullptr) {
      throw std::invalid_argument("Forward normalized must be non-null");
    }

    // Chain order, shapes, and arithmetic exactly match the original reference
    // LeNet forward path. ReLU is in-place on its own output, which the
    // original launcher contract permits and leaves the values bit-identical.
    LaunchConvolutionForward(normalized, parameters_[0], parameters_[1],
                             conv1_, batch, 1, 28, 28, 6, 5, 5, stream_);
    LaunchReluForward(conv1_, conv1_, Count(batch, kConv1FloatsPerSample),
                      stream_);
    LaunchMaxPoolForward(conv1_, pool1_, pool1_winners_, batch, 6, 24, 24,
                         stream_);
    LaunchConvolutionForward(pool1_, parameters_[2], parameters_[3], conv2_,
                             batch, 6, 12, 12, 16, 5, 5, stream_);
    LaunchReluForward(conv2_, conv2_, Count(batch, kConv2FloatsPerSample),
                      stream_);
    LaunchMaxPoolForward(conv2_, pool2_, pool2_winners_, batch, 16, 8, 8,
                         stream_);
    // pool2_ is already contiguous N x 256; flattening is this pointer only.
    LaunchLinearForward(pool2_, parameters_[4], parameters_[5], fc1_, batch,
                        256, 120, stream_);
    LaunchReluForward(fc1_, fc1_, Count(batch, kFc1FloatsPerSample), stream_);
    LaunchLinearForward(fc1_, parameters_[6], parameters_[7], fc2_, batch, 120,
                        84, stream_);
    LaunchReluForward(fc2_, fc2_, Count(batch, kFc2FloatsPerSample), stream_);
    LaunchLinearForward(fc2_, parameters_[8], parameters_[9], logits_, batch,
                        84, 10, stream_);
    return logits_;
  }

  // Returns the exact arena size; no CUDA call or allocation occurs.
  std::size_t RequiredDeviceBytes() const { return required_bytes_; }

 private:
  // Rejects any non-finite supplied weight before device storage is touched.
  static void ValidateFiniteParameters(const ParameterSet& parameters) {
    for (std::size_t tensor = 0; tensor < parameters.size(); ++tensor) {
      const std::vector<float>& values = parameters[tensor].values;
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
          std::ostringstream message;
          message << "InferenceModel parameter " << parameters[tensor].name
                  << '[' << index << "] must be finite";
          throw std::invalid_argument(message.str());
        }
      }
    }
  }

  // Bump-allocates count aligned FP32 elements from the arena.
  float* TakeFloats(std::size_t count, std::size_t* offset) {
    *offset = AlignUp(*offset);
    float* result = reinterpret_cast<float*>(arena_.get() + *offset);
    *offset = AfterFloats(*offset, count);
    return result;
  }

  // Bump-allocates count aligned bytes from the arena.
  std::uint8_t* TakeBytes(std::size_t count, std::size_t* offset) {
    *offset = AlignUp(*offset);
    std::uint8_t* result = arena_.get() + *offset;
    *offset = AfterBytes(*offset, count);
    return result;
  }

  // Partitions the single arena in canonical parameter order followed by
  // forward activations and pooling winners.
  void PlanArena() {
    std::size_t offset = 0;
    const auto& specs = LenetParameterSpecs();
    for (std::size_t index = 0; index < specs.size(); ++index) {
      parameters_[index] = TakeFloats(
          static_cast<std::size_t>(specs[index].element_count), &offset);
    }
    conv1_ = TakeFloats(Count(maximum_batch_, kConv1FloatsPerSample), &offset);
    pool1_ = TakeFloats(Count(maximum_batch_, kPool1FloatsPerSample), &offset);
    conv2_ = TakeFloats(Count(maximum_batch_, kConv2FloatsPerSample), &offset);
    pool2_ = TakeFloats(Count(maximum_batch_, kPool2FloatsPerSample), &offset);
    fc1_ = TakeFloats(Count(maximum_batch_, kFc1FloatsPerSample), &offset);
    fc2_ = TakeFloats(Count(maximum_batch_, kFc2FloatsPerSample), &offset);
    logits_ = TakeFloats(Count(maximum_batch_, kLogitsFloatsPerSample), &offset);
    pool1_winners_ =
        TakeBytes(Count(maximum_batch_, kPool1WinnerBytesPerSample), &offset);
    pool2_winners_ =
        TakeBytes(Count(maximum_batch_, kPool2WinnerBytesPerSample), &offset);
    if (offset != required_bytes_) {
      throw std::logic_error("InferenceModel device memory plan byte mismatch");
    }
  }

  // Uploads every canonical parameter payload directly, in order.
  void CopyParametersToDevice(const ParameterSet& parameters) {
    const auto& specs = LenetParameterSpecs();
    for (std::size_t index = 0; index < specs.size(); ++index) {
      CUDA_CHECK(cudaMemcpyAsync(
          parameters_[index], parameters[index].values.data(),
          static_cast<std::size_t>(specs[index].element_count) * sizeof(float),
          cudaMemcpyHostToDevice, stream_));
    }
  }

  // Immutable configuration and the single owning arena.
  int maximum_batch_;
  cudaStream_t stream_;
  std::size_t required_bytes_;
  DeviceBuffer<std::uint8_t> arena_;

  // Canonical parameter payloads and forward-only activation storage.
  std::array<float*, 10> parameters_{};
  float* conv1_ = nullptr;
  float* pool1_ = nullptr;
  float* conv2_ = nullptr;
  float* pool2_ = nullptr;
  float* fc1_ = nullptr;
  float* fc2_ = nullptr;
  float* logits_ = nullptr;
  std::uint8_t* pool1_winners_ = nullptr;
  std::uint8_t* pool2_winners_ = nullptr;
};

// Public facade forwarding to the pimpl; validation and launches live in Impl.
InferenceModel::InferenceModel(int maximum_batch,
                               const ParameterSet& parameters,
                               cudaStream_t stream)
    : impl_(new Impl(maximum_batch, parameters, stream)) {}

InferenceModel::~InferenceModel() = default;

const float* InferenceModel::Forward(const float* normalized, int batch) {
  return impl_->Forward(normalized, batch);
}

std::size_t InferenceModel::RequiredDeviceBytes() const {
  return impl_->RequiredDeviceBytes();
}

}  // namespace course
