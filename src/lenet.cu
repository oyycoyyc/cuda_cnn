#include "lenet.h"

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

namespace {

constexpr std::size_t kParameterCount = 44426;
constexpr std::size_t kActivationFloatsPerSample = 10498;
constexpr std::size_t kGradientFloatsPerSample = 6588;
constexpr std::size_t kWinnerBytesPerSample = 1120;
constexpr std::size_t kMaximumScannedTensors = 58;

std::size_t CheckedAdd(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error("LeNet required device bytes overflow size_t");
  }
  return left + right;
}

std::size_t CheckedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("LeNet required device bytes overflow size_t");
  }
  return left * right;
}

std::size_t RequiredBytes(int maximum_batch_size) {
  if (maximum_batch_size <= 0) {
    throw std::invalid_argument("maximum_batch_size must be positive");
  }
  const std::size_t batch = static_cast<std::size_t>(maximum_batch_size);
  const std::size_t parameter_bytes =
      CheckedMultiply(CheckedMultiply(kParameterCount, 4), sizeof(float));
  const std::size_t sample_float_bytes = CheckedMultiply(
      CheckedMultiply(CheckedAdd(kActivationFloatsPerSample,
                                 kGradientFloatsPerSample),
                      batch),
      sizeof(float));
  const std::size_t winner_bytes =
      CheckedMultiply(kWinnerBytesPerSample, batch);
  return CheckedAdd(CheckedAdd(CheckedAdd(parameter_bytes, sample_float_bytes),
                               winner_bytes),
                    sizeof(int));
}

void ValidateAdamW(std::uint64_t global_step, float learning_rate,
                   const AdamWConfig& config) {
  if (global_step == 0) {
    throw std::invalid_argument("AdamW global_step must be at least one");
  }
  if (!std::isfinite(learning_rate) || learning_rate < 0.0F) {
    throw std::invalid_argument(
        "AdamW learning_rate must be finite and nonnegative");
  }
  if (!std::isfinite(config.beta1) || config.beta1 < 0.0F ||
      config.beta1 >= 1.0F || !std::isfinite(config.beta2) ||
      config.beta2 < 0.0F || config.beta2 >= 1.0F) {
    throw std::invalid_argument("AdamW beta values must be finite in [0,1)");
  }
  if (!std::isfinite(config.epsilon) || config.epsilon <= 0.0F) {
    throw std::invalid_argument("AdamW epsilon must be finite and positive");
  }
  if (!std::isfinite(config.weight_decay) || config.weight_decay < 0.0F) {
    throw std::invalid_argument(
        "AdamW weight_decay must be finite and nonnegative");
  }
}

struct ScanTensor {
  const char* name;
  const float* values;
  std::size_t count;
};

}  // namespace

class LeNet::Impl {
 public:
  Impl(int maximum_batch_size, std::uint64_t seed, cudaStream_t stream)
      : maximum_batch_size_(maximum_batch_size),
        stream_(stream),
        required_bytes_(RequiredBytes(maximum_batch_size)) {
    PrepareFiniteDiagnosticNames();
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(device_buffer_detail::QueryMemoryInfo(&free_bytes,
                                                      &total_bytes));
    if (free_bytes < required_bytes_) {
      std::ostringstream message;
      message << "insufficient CUDA memory: required_bytes=" << required_bytes_
              << " free_bytes=" << free_bytes
              << " total_bytes=" << total_bytes;
      throw std::runtime_error(message.str());
    }

    arena_ = DeviceBuffer<std::uint8_t>(required_bytes_);
    PlanArena();
    CUDA_CHECK(cudaMemsetAsync(arena_.get(), 0, required_bytes_, stream_));

    ParameterSet initial = CreateLenetParameters();
    InitializeLenetParameters(seed, &initial);
    CopyParametersToDevice(initial);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  const float* Forward(const float* normalized_images, int batch_size) {
    ValidateBatch(batch_size);
    if (normalized_images == nullptr) {
      throw std::invalid_argument("normalized_images must be non-null");
    }
    InvalidateWork();

    LaunchConvolutionForward(normalized_images, parameters_[0], parameters_[1],
                             conv1_pre_, batch_size, 1, 28, 28, 6, 5, 5,
                             stream_);
    LaunchReluForward(conv1_pre_, relu1_, Count(batch_size, 3456), stream_);
    LaunchMaxPoolForward(relu1_, pool1_, pool1_winners_, batch_size, 6, 24, 24,
                         stream_);
    LaunchConvolutionForward(pool1_, parameters_[2], parameters_[3],
                             conv2_pre_, batch_size, 6, 12, 12, 16, 5, 5,
                             stream_);
    LaunchReluForward(conv2_pre_, relu2_, Count(batch_size, 1024), stream_);
    LaunchMaxPoolForward(relu2_, pool2_, pool2_winners_, batch_size, 16, 8, 8,
                         stream_);
    // pool2_ is already contiguous N x 256; flattening is this pointer only.
    last_fc1_input_ = pool2_;
    LaunchLinearForward(last_fc1_input_, parameters_[4], parameters_[5],
                        fc1_pre_, batch_size, 256, 120, stream_);
    LaunchReluForward(fc1_pre_, relu3_, Count(batch_size, 120), stream_);
    LaunchLinearForward(relu3_, parameters_[6], parameters_[7], fc2_pre_,
                        batch_size, 120, 84, stream_);
    LaunchReluForward(fc2_pre_, relu4_, Count(batch_size, 84), stream_);
    LaunchLinearForward(relu4_, parameters_[8], parameters_[9], logits_,
                        batch_size, 84, 10, stream_);
    forward_input_ = normalized_images;
    state_batch_size_ = batch_size;
    state_ = ExecutionState::kForwardReady;
    return logits_;
  }

  void Backward(const float* logits_gradient, int batch_size) {
    ValidateBatch(batch_size);
    if (logits_gradient == nullptr) {
      throw std::invalid_argument("logits_gradient must be non-null");
    }
    if (state_ != ExecutionState::kForwardReady || forward_input_ == nullptr ||
        batch_size != state_batch_size_) {
      throw std::invalid_argument(
          "Backward requires a matching unconsumed Forward");
    }
    const float* const forward_input = forward_input_;
    InvalidateWork();

    LaunchLinearBackward(relu4_, parameters_[8], logits_gradient,
                         grad_relu4_, parameter_gradients_[8],
                         parameter_gradients_[9], batch_size, 84, 10, stream_);
    LaunchReluBackward(fc2_pre_, grad_relu4_, grad_relu4_,
                       Count(batch_size, 84), stream_);
    LaunchLinearBackward(relu3_, parameters_[6], grad_relu4_, grad_relu3_,
                         parameter_gradients_[6], parameter_gradients_[7],
                         batch_size, 120, 84, stream_);
    LaunchReluBackward(fc1_pre_, grad_relu3_, grad_relu3_,
                       Count(batch_size, 120), stream_);
    LaunchLinearBackward(pool2_, parameters_[4], grad_relu3_, grad_pool2_,
                         parameter_gradients_[4], parameter_gradients_[5],
                         batch_size, 256, 120, stream_);
    LaunchMaxPoolBackward(grad_pool2_, pool2_winners_, grad_relu2_, batch_size,
                          16, 8, 8, stream_);
    LaunchReluBackward(conv2_pre_, grad_relu2_, grad_relu2_,
                       Count(batch_size, 1024), stream_);
    LaunchConvolutionBackward(
        pool1_, parameters_[2], grad_relu2_, grad_pool1_,
        parameter_gradients_[2], parameter_gradients_[3], batch_size, 6, 12,
        12, 16, 5, 5, stream_);
    LaunchMaxPoolBackward(grad_pool1_, pool1_winners_, grad_relu1_, batch_size,
                          6, 24, 24, stream_);
    LaunchReluBackward(conv1_pre_, grad_relu1_, grad_relu1_,
                       Count(batch_size, 3456), stream_);
    LaunchConvolutionBackward(
        forward_input, parameters_[0], grad_relu1_, input_gradient_,
        parameter_gradients_[0], parameter_gradients_[1], batch_size, 1, 28,
        28, 6, 5, 5, stream_);
    state_batch_size_ = batch_size;
    state_ = ExecutionState::kGradientsReady;
  }

  void AdamWStep(std::uint64_t global_step, float learning_rate,
                 const AdamWConfig& config) {
    ValidateAdamW(global_step, learning_rate, config);
    if (state_ != ExecutionState::kGradientsReady) {
      throw std::invalid_argument(
          "AdamWStep requires current gradients from Backward");
    }
    const double step = static_cast<double>(global_step);
    const float correction1 = static_cast<float>(
        1.0 / (1.0 - std::pow(static_cast<double>(config.beta1), step)));
    const float correction2 = static_cast<float>(
        1.0 / (1.0 - std::pow(static_cast<double>(config.beta2), step)));
    InvalidateWork();
    const auto& specs = LenetParameterSpecs();
    for (std::size_t index = 0; index < specs.size(); ++index) {
      LaunchAdamW(parameters_[index], parameter_gradients_[index],
                  first_moments_[index], second_moments_[index],
                  static_cast<std::size_t>(specs[index].element_count),
                  learning_rate, config.beta1, config.beta2, config.epsilon,
                  correction1, correction2,
                  specs[index].is_bias ? 0.0F : config.weight_decay, stream_);
    }
  }

  ParameterSet ExportParameters() const {
    ParameterSet result = CreateLenetParameters();
    ExportParameters(&result);
    return result;
  }

  void ExportParameters(ParameterSet* destination) const {
    if (destination == nullptr) {
      throw std::invalid_argument("export destination must not be null");
    }
    ValidateLenetParameters(*destination);
    const auto& specs = LenetParameterSpecs();
    for (std::size_t index = 0; index < specs.size(); ++index) {
      CUDA_CHECK(cudaMemcpyAsync(
          (*destination)[index].values.data(), parameters_[index],
          static_cast<std::size_t>(specs[index].element_count) * sizeof(float),
          cudaMemcpyDeviceToHost, stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  void ImportParameters(const ParameterSet& parameters) {
    ValidateLenetParameters(parameters);
    InvalidateWork();
    CopyParametersToDevice(parameters);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  void RequireFinite(const std::string& phase) const {
    std::array<ScanTensor, kMaximumScannedTensors> tensors{};
    std::size_t count = 0;
    const auto append = [&tensors, &count](const char* name,
                                           const float* values,
                                           std::size_t value_count) {
      tensors[count++] = ScanTensor{name, values, value_count};
    };
    const auto& specs = LenetParameterSpecs();
    for (std::size_t index = 0; index < specs.size(); ++index) {
      append(specs[index].name, parameters_[index],
             static_cast<std::size_t>(specs[index].element_count));
    }
    if (state_ != ExecutionState::kIdle) {
      AppendActivations(append);
    }
    if (state_ == ExecutionState::kGradientsReady) {
      AppendGradients(append);
    }
    for (std::size_t index = 0; index < specs.size(); ++index) {
      append(moment_names_[index].c_str(), first_moments_[index],
             static_cast<std::size_t>(specs[index].element_count));
      append(moment_names_[index + specs.size()].c_str(),
             second_moments_[index],
             static_cast<std::size_t>(specs[index].element_count));
    }

    for (std::size_t index = 0; index < count; ++index) {
      LaunchFindFirstNonFinite(tensors[index].values, tensors[index].count,
                               scan_result_, stream_);
      CUDA_CHECK(cudaMemcpyAsync(&scan_results_[index], scan_result_,
                                 sizeof(int), cudaMemcpyDeviceToHost, stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    for (std::size_t index = 0; index < count; ++index) {
      if (scan_results_[index] != static_cast<int>(tensors[index].count)) {
        std::ostringstream message;
        message << phase << ": " << tensors[index].name << '['
                << scan_results_[index] << "] is not finite";
        throw std::runtime_error(message.str());
      }
    }
  }

  std::size_t RequiredDeviceBytes() const { return required_bytes_; }

 private:
  friend class LeNetTestAccess;

  enum class ExecutionState { kIdle, kForwardReady, kGradientsReady };

  void InvalidateWork() noexcept {
    state_ = ExecutionState::kIdle;
    state_batch_size_ = 0;
    forward_input_ = nullptr;
    last_fc1_input_ = nullptr;
  }

  void PrepareFiniteDiagnosticNames() {
    const auto& specs = LenetParameterSpecs();
    for (std::size_t index = 0; index < specs.size(); ++index) {
      gradient_names_[index] =
          std::string(specs[index].name) + ".gradient";
      moment_names_[index] = std::string(specs[index].name) + ".first_moment";
      moment_names_[index + specs.size()] =
          std::string(specs[index].name) + ".second_moment";
    }
  }

  bool FiniteDiagnosticNamesPrepared() const noexcept {
    for (const std::string& name : gradient_names_) {
      if (name.empty()) {
        return false;
      }
    }
    for (const std::string& name : moment_names_) {
      if (name.empty()) {
        return false;
      }
    }
    return true;
  }

  template <typename Append>
  void AppendActivations(const Append& append) const {
    const int batch = state_batch_size_;
    append("conv1.pre_activation", conv1_pre_, Count(batch, 3456));
    append("relu1", relu1_, Count(batch, 3456));
    append("pool1", pool1_, Count(batch, 864));
    append("conv2.pre_activation", conv2_pre_, Count(batch, 1024));
    append("relu2", relu2_, Count(batch, 1024));
    append("pool2", pool2_, Count(batch, 256));
    append("fc1.pre_activation", fc1_pre_, Count(batch, 120));
    append("relu3", relu3_, Count(batch, 120));
    append("fc2.pre_activation", fc2_pre_, Count(batch, 84));
    append("relu4", relu4_, Count(batch, 84));
    append("logits", logits_, Count(batch, 10));
  }

  template <typename Append>
  void AppendGradients(const Append& append) const {
    const auto& specs = LenetParameterSpecs();
    for (std::size_t index = 0; index < specs.size(); ++index) {
      append(gradient_names_[index].c_str(), parameter_gradients_[index],
             static_cast<std::size_t>(specs[index].element_count));
    }
    const int batch = state_batch_size_;
    append("relu4.gradient", grad_relu4_, Count(batch, 84));
    append("relu3.gradient", grad_relu3_, Count(batch, 120));
    append("pool2.gradient", grad_pool2_, Count(batch, 256));
    append("relu2.gradient", grad_relu2_, Count(batch, 1024));
    append("pool1.gradient", grad_pool1_, Count(batch, 864));
    append("relu1.gradient", grad_relu1_, Count(batch, 3456));
    append("input.gradient", input_gradient_, Count(batch, 784));
  }

  void ValidateBatch(int batch_size) const {
    if (batch_size <= 0) {
      throw std::invalid_argument("batch_size must be positive");
    }
    if (batch_size > maximum_batch_size_) {
      throw std::invalid_argument("batch_size exceeds maximum_batch_size");
    }
  }

  static std::size_t Count(int batch_size, std::size_t per_sample) {
    return static_cast<std::size_t>(batch_size) * per_sample;
  }

  void CopyParametersToDevice(const ParameterSet& parameters) {
    const auto& specs = LenetParameterSpecs();
    for (std::size_t index = 0; index < specs.size(); ++index) {
      CUDA_CHECK(cudaMemcpyAsync(
          parameters_[index], parameters[index].values.data(),
          static_cast<std::size_t>(specs[index].element_count) * sizeof(float),
          cudaMemcpyHostToDevice, stream_));
    }
  }

  template <typename T>
  T* Take(std::size_t count, std::size_t* offset) {
    T* result = reinterpret_cast<T*>(arena_.get() + *offset);
    *offset += count * sizeof(T);
    return result;
  }

  void PlanArena() {
    std::size_t offset = 0;
    const auto& specs = LenetParameterSpecs();
    for (std::size_t index = 0; index < specs.size(); ++index) {
      parameters_[index] = Take<float>(specs[index].element_count, &offset);
    }
    for (std::size_t index = 0; index < specs.size(); ++index) {
      parameter_gradients_[index] =
          Take<float>(specs[index].element_count, &offset);
    }
    for (std::size_t index = 0; index < specs.size(); ++index) {
      first_moments_[index] = Take<float>(specs[index].element_count, &offset);
    }
    for (std::size_t index = 0; index < specs.size(); ++index) {
      second_moments_[index] = Take<float>(specs[index].element_count, &offset);
    }

    const std::size_t batch = static_cast<std::size_t>(maximum_batch_size_);
    conv1_pre_ = Take<float>(batch * 3456, &offset);
    relu1_ = Take<float>(batch * 3456, &offset);
    pool1_ = Take<float>(batch * 864, &offset);
    conv2_pre_ = Take<float>(batch * 1024, &offset);
    relu2_ = Take<float>(batch * 1024, &offset);
    pool2_ = Take<float>(batch * 256, &offset);
    fc1_pre_ = Take<float>(batch * 120, &offset);
    relu3_ = Take<float>(batch * 120, &offset);
    fc2_pre_ = Take<float>(batch * 84, &offset);
    relu4_ = Take<float>(batch * 84, &offset);
    logits_ = Take<float>(batch * 10, &offset);

    grad_relu4_ = Take<float>(batch * 84, &offset);
    grad_relu3_ = Take<float>(batch * 120, &offset);
    grad_pool2_ = Take<float>(batch * 256, &offset);
    grad_relu2_ = Take<float>(batch * 1024, &offset);
    grad_pool1_ = Take<float>(batch * 864, &offset);
    grad_relu1_ = Take<float>(batch * 3456, &offset);
    input_gradient_ = Take<float>(batch * 784, &offset);

    pool1_winners_ = Take<std::uint8_t>(batch * 864, &offset);
    pool2_winners_ = Take<std::uint8_t>(batch * 256, &offset);
    scan_result_ = Take<int>(1, &offset);
    if (offset != required_bytes_) {
      throw std::logic_error("LeNet device memory plan byte mismatch");
    }
  }

  int maximum_batch_size_;
  cudaStream_t stream_;
  std::size_t required_bytes_;
  DeviceBuffer<std::uint8_t> arena_;
  std::array<float*, 10> parameters_{};
  std::array<float*, 10> parameter_gradients_{};
  std::array<float*, 10> first_moments_{};
  std::array<float*, 10> second_moments_{};

  float* conv1_pre_ = nullptr;
  float* relu1_ = nullptr;
  float* pool1_ = nullptr;
  float* conv2_pre_ = nullptr;
  float* relu2_ = nullptr;
  float* pool2_ = nullptr;
  float* fc1_pre_ = nullptr;
  float* relu3_ = nullptr;
  float* fc2_pre_ = nullptr;
  float* relu4_ = nullptr;
  float* logits_ = nullptr;

  float* grad_relu4_ = nullptr;
  float* grad_relu3_ = nullptr;
  float* grad_pool2_ = nullptr;
  float* grad_relu2_ = nullptr;
  float* grad_pool1_ = nullptr;
  float* grad_relu1_ = nullptr;
  float* input_gradient_ = nullptr;
  std::uint8_t* pool1_winners_ = nullptr;
  std::uint8_t* pool2_winners_ = nullptr;
  int* scan_result_ = nullptr;

  ExecutionState state_ = ExecutionState::kIdle;
  int state_batch_size_ = 0;
  const float* forward_input_ = nullptr;
  const float* last_fc1_input_ = nullptr;
  mutable std::array<int, kMaximumScannedTensors> scan_results_{};
  mutable std::array<std::string, 10> gradient_names_{};
  mutable std::array<std::string, 20> moment_names_{};
};

LeNet::LeNet(int maximum_batch_size, std::uint64_t seed, cudaStream_t stream)
    : impl_(new Impl(maximum_batch_size, seed, stream)) {}

LeNet::~LeNet() = default;

const float* LeNet::Forward(const float* normalized_images, int batch_size) {
  return impl_->Forward(normalized_images, batch_size);
}

void LeNet::Backward(const float* logits_gradient, int batch_size) {
  impl_->Backward(logits_gradient, batch_size);
}

void LeNet::AdamWStep(std::uint64_t global_step, float learning_rate,
                      const AdamWConfig& config) {
  impl_->AdamWStep(global_step, learning_rate, config);
}

ParameterSet LeNet::ExportParameters() const {
  return impl_->ExportParameters();
}

void LeNet::ExportParameters(ParameterSet* destination) const {
  impl_->ExportParameters(destination);
}

void LeNet::ImportParameters(const ParameterSet& parameters) {
  impl_->ImportParameters(parameters);
}

void LeNet::RequireFinite(const std::string& phase) const {
  impl_->RequireFinite(phase);
}

std::size_t LeNet::RequiredDeviceBytes() const {
  return impl_->RequiredDeviceBytes();
}

bool LeNetTestAccess::Fc1InputAliasesPool2(const LeNet& model) noexcept {
  return model.impl_->state_ != LeNet::Impl::ExecutionState::kIdle &&
         model.impl_->last_fc1_input_ == model.impl_->pool2_;
}

bool LeNetTestAccess::FiniteDiagnosticNamesPrepared(
    const LeNet& model) noexcept {
  return model.impl_->FiniteDiagnosticNamesPrepared();
}
