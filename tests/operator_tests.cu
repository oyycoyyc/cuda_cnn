#define CUDA_LENET_ENABLE_TEST_HOOKS

#include "cpu_reference.h"
#include "cuda_check.h"
#include "layers.h"
#include "random.h"
#include "tensor.h"
#include "test_harness.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

static_assert(
    std::is_same<decltype(device_buffer_detail::LiveAllocationCount()),
                 std::atomic<std::size_t>&>::value,
    "DeviceBuffer allocation accounting must be atomic");

// Performs no mathematical operation and accesses no tensors. A valid launch
// would map each thread to no output, execute no loops or boundary paths, and
// issue no writes, so races, synchronization, atomics, and numerical stability
// are inapplicable. This test deliberately launches it with a zero-sized grid
// to verify host-side launch-error attribution before any thread can exist.
__global__ void NoOpKernel() {}

template <typename T>
DeviceBuffer<T> CopyToDevice(const std::vector<T>& source) {
  DeviceBuffer<T> destination(source.size());
  CUDA_CHECK(cudaMemcpy(destination.get(), source.data(),
                        source.size() * sizeof(T), cudaMemcpyHostToDevice));
  return destination;
}

template <typename T>
std::vector<T> CopyFromDevice(const DeviceBuffer<T>& source) {
  std::vector<T> destination(source.size());
  CUDA_CHECK(cudaMemcpy(destination.data(), source.get(),
                        destination.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  return destination;
}

float NormalizedPixel(std::uint8_t pixel) {
  return (static_cast<float>(pixel) / 255.0F - 0.1307F) / 0.3081F;
}

void ExpectNearVectors(const std::vector<float>& expected,
                       const std::vector<float>& actual) {
  EXPECT_EQ(expected.size(), actual.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const float tolerance = 1.0e-4F + 1.0e-4F * std::fabs(expected[index]);
    EXPECT_NEAR(expected[index], actual[index], tolerance);
  }
}

void CheckInputTranslation(int batch_size) {
  constexpr int kImageSide = 28;
  constexpr int kImagePixels = kImageSide * kImageSide;
  constexpr std::uint64_t kSeed = UINT64_C(0x8877665544332211);
  constexpr std::uint32_t kEpoch = 7;
  const std::size_t element_count =
      static_cast<std::size_t>(batch_size) * kImagePixels;
  std::vector<std::uint8_t> images(element_count);
  std::vector<std::uint32_t> original_indices(batch_size);
  for (int sample = 0; sample < batch_size; ++sample) {
    original_indices[sample] =
        static_cast<std::uint32_t>(12345 + sample * 997);
    for (int pixel = 0; pixel < kImagePixels; ++pixel) {
      images[static_cast<std::size_t>(sample) * kImagePixels + pixel] =
          static_cast<std::uint8_t>((sample * 37 + pixel * 13 + 19) % 256);
    }
  }

  std::vector<float> expected(element_count);
  bool saw_padding = false;
  std::size_t first_padding_index = 0;
  for (int sample = 0; sample < batch_size; ++sample) {
    const TranslationOffset offset = TranslationOffsetForSample(
        kSeed, kEpoch, original_indices[sample]);
    for (int y = 0; y < kImageSide; ++y) {
      for (int x = 0; x < kImageSide; ++x) {
        // Positive dx/dy moves content right/down, so output samples x-dx/y-dy.
        const int source_x = x - offset.dx;
        const int source_y = y - offset.dy;
        std::uint8_t pixel = 0;
        if (source_x >= 0 && source_x < kImageSide && source_y >= 0 &&
            source_y < kImageSide) {
          pixel = images[static_cast<std::size_t>(sample) * kImagePixels +
                         source_y * kImageSide + source_x];
        } else {
          if (!saw_padding) {
            first_padding_index =
                static_cast<std::size_t>(sample) * kImagePixels +
                y * kImageSide + x;
          }
          saw_padding = true;
        }
        expected[static_cast<std::size_t>(sample) * kImagePixels +
                 y * kImageSide + x] = NormalizedPixel(pixel);
      }
    }
  }
  EXPECT_TRUE(saw_padding);

  DeviceBuffer<std::uint8_t> device_images = CopyToDevice(images);
  DeviceBuffer<std::uint32_t> device_indices = CopyToDevice(original_indices);
  DeviceBuffer<float> device_output(element_count);
  LaunchNormalizeTranslate(device_images.get(), device_indices.get(),
                           device_output.get(), batch_size, kSeed, kEpoch, true,
                           nullptr);
  const std::vector<float> actual = CopyFromDevice(device_output);
  ExpectNearVectors(expected, actual);
  // Translation pads with a pixel value of zero before normalization.
  EXPECT_NEAR(-0.42421293F, actual[first_padding_index], 1.5e-4F);
}

std::vector<float> Pattern(std::size_t count, int offset) {
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered =
        static_cast<int>((index * 37 + static_cast<std::size_t>(offset) * 17) %
                         97) -
        48;
    values[index] = static_cast<float>(centered) * 0.01F;
  }
  return values;
}

void CheckLinearAgainstCpu(int batch_size, int input_features,
                           int output_features) {
  const std::vector<float> input = Pattern(
      static_cast<std::size_t>(batch_size) * input_features, 1);
  const std::vector<float> weight = Pattern(
      static_cast<std::size_t>(output_features) * input_features, 2);
  const std::vector<float> bias = Pattern(output_features, 3);
  const std::vector<float> output_gradient = Pattern(
      static_cast<std::size_t>(batch_size) * output_features, 4);
  const std::vector<float> expected_output = cpu_reference::LinearForward(
      input, weight, bias, batch_size, input_features, output_features);
  const cpu_reference::LinearGradients expected_gradients =
      cpu_reference::LinearBackward(input, weight, output_gradient, batch_size,
                                    input_features, output_features);

  DeviceBuffer<float> device_input = CopyToDevice(input);
  DeviceBuffer<float> device_weight = CopyToDevice(weight);
  DeviceBuffer<float> device_bias = CopyToDevice(bias);
  DeviceBuffer<float> device_output(expected_output.size());
  DeviceBuffer<float> device_output_gradient = CopyToDevice(output_gradient);
  DeviceBuffer<float> device_input_gradient(input.size());
  DeviceBuffer<float> device_weight_gradient(weight.size());
  DeviceBuffer<float> device_bias_gradient(bias.size());

  LaunchLinearForward(device_input.get(), device_weight.get(),
                      device_bias.get(), device_output.get(), batch_size,
                      input_features, output_features, nullptr);
  LaunchLinearBackward(
      device_input.get(), device_weight.get(), device_output_gradient.get(),
      device_input_gradient.get(), device_weight_gradient.get(),
      device_bias_gradient.get(), batch_size, input_features, output_features,
      nullptr);

  ExpectNearVectors(expected_output, CopyFromDevice(device_output));
  ExpectNearVectors(expected_gradients.input,
                    CopyFromDevice(device_input_gradient));
  ExpectNearVectors(expected_gradients.weight,
                    CopyFromDevice(device_weight_gradient));
  ExpectNearVectors(expected_gradients.bias,
                    CopyFromDevice(device_bias_gradient));
}

void CheckConvolutionAgainstCpu(int batch_size, int input_channels,
                                int input_height, int input_width,
                                int output_channels, int kernel_height,
                                int kernel_width) {
  const int output_height = input_height - kernel_height + 1;
  const int output_width = input_width - kernel_width + 1;
  const std::vector<float> input = Pattern(
      static_cast<std::size_t>(batch_size) * input_channels * input_height *
          input_width,
      5);
  const std::vector<float> weight = Pattern(
      static_cast<std::size_t>(output_channels) * input_channels *
          kernel_height * kernel_width,
      6);
  const std::vector<float> bias = Pattern(output_channels, 7);
  const std::vector<float> output_gradient = Pattern(
      static_cast<std::size_t>(batch_size) * output_channels * output_height *
          output_width,
      8);
  const std::vector<float> expected_output =
      cpu_reference::ConvolutionForward(
          input, weight, bias, batch_size, input_channels, input_height,
          input_width, output_channels, kernel_height, kernel_width);
  const cpu_reference::ConvolutionGradients expected_gradients =
      cpu_reference::ConvolutionBackward(
          input, weight, output_gradient, batch_size, input_channels,
          input_height, input_width, output_channels, kernel_height,
          kernel_width);

  DeviceBuffer<float> device_input = CopyToDevice(input);
  DeviceBuffer<float> device_weight = CopyToDevice(weight);
  DeviceBuffer<float> device_bias = CopyToDevice(bias);
  DeviceBuffer<float> device_output(expected_output.size());
  DeviceBuffer<float> device_output_gradient = CopyToDevice(output_gradient);
  DeviceBuffer<float> device_input_gradient(input.size());
  DeviceBuffer<float> device_weight_gradient(weight.size());
  DeviceBuffer<float> device_bias_gradient(bias.size());

  LaunchConvolutionForward(
      device_input.get(), device_weight.get(), device_bias.get(),
      device_output.get(), batch_size, input_channels, input_height,
      input_width, output_channels, kernel_height, kernel_width, nullptr);
  LaunchConvolutionBackward(
      device_input.get(), device_weight.get(), device_output_gradient.get(),
      device_input_gradient.get(), device_weight_gradient.get(),
      device_bias_gradient.get(), batch_size, input_channels, input_height,
      input_width, output_channels, kernel_height, kernel_width, nullptr);

  ExpectNearVectors(expected_output, CopyFromDevice(device_output));
  ExpectNearVectors(expected_gradients.input,
                    CopyFromDevice(device_input_gradient));
  ExpectNearVectors(expected_gradients.weight,
                    CopyFromDevice(device_weight_gradient));
  ExpectNearVectors(expected_gradients.bias,
                    CopyFromDevice(device_bias_gradient));
}

double DotObjective(const std::vector<float>& output,
                    const std::vector<float>& output_gradient) {
  double objective = 0.0;
  for (std::size_t index = 0; index < output.size(); ++index) {
    objective += static_cast<double>(output[index]) * output_gradient[index];
  }
  return objective;
}

double LinearDeviceObjective(const std::vector<float>& input,
                             const std::vector<float>& weight,
                             const std::vector<float>& bias,
                             const std::vector<float>& output_gradient,
                             int batch_size, int input_features,
                             int output_features) {
  DeviceBuffer<float> device_input = CopyToDevice(input);
  DeviceBuffer<float> device_weight = CopyToDevice(weight);
  DeviceBuffer<float> device_bias = CopyToDevice(bias);
  DeviceBuffer<float> device_output(output_gradient.size());
  LaunchLinearForward(device_input.get(), device_weight.get(),
                      device_bias.get(), device_output.get(), batch_size,
                      input_features, output_features, nullptr);
  return DotObjective(CopyFromDevice(device_output), output_gradient);
}

void CheckProductionLinearFiniteDifferences(int input_features,
                                            int output_features, int offset) {
  constexpr int kBatchSize = 2;
  constexpr float kEpsilon = 1.0e-3F;
  const std::vector<float> input = Pattern(
      static_cast<std::size_t>(kBatchSize) * input_features, offset);
  std::vector<float> weight = Pattern(
      static_cast<std::size_t>(output_features) * input_features, offset + 1);
  const std::vector<float> bias = Pattern(output_features, offset + 2);
  const std::vector<float> output_gradient = Pattern(
      static_cast<std::size_t>(kBatchSize) * output_features, offset + 3);

  DeviceBuffer<float> device_input = CopyToDevice(input);
  DeviceBuffer<float> device_weight = CopyToDevice(weight);
  DeviceBuffer<float> device_output_gradient = CopyToDevice(output_gradient);
  DeviceBuffer<float> device_input_gradient(input.size());
  DeviceBuffer<float> device_weight_gradient(weight.size());
  DeviceBuffer<float> device_bias_gradient(bias.size());
  LaunchLinearBackward(
      device_input.get(), device_weight.get(), device_output_gradient.get(),
      device_input_gradient.get(), device_weight_gradient.get(),
      device_bias_gradient.get(), kBatchSize, input_features, output_features,
      nullptr);
  const std::vector<float> analytic =
      CopyFromDevice(device_weight_gradient);

  const std::vector<std::size_t> checked_indices{
      0, weight.size() / 2, weight.size() - 1};
  for (std::size_t index : checked_indices) {
    const float original = weight[index];
    weight[index] = original + kEpsilon;
    const double plus = LinearDeviceObjective(
        input, weight, bias, output_gradient, kBatchSize, input_features,
        output_features);
    weight[index] = original - kEpsilon;
    const double minus = LinearDeviceObjective(
        input, weight, bias, output_gradient, kBatchSize, input_features,
        output_features);
    weight[index] = original;
    const double numeric = (plus - minus) / (2.0 * kEpsilon);
    const double tolerance = 1.0e-2 + 1.0e-2 * std::fabs(numeric);
    EXPECT_NEAR(numeric, analytic[index], tolerance);
  }
}

double ConvolutionDeviceObjective(
    const std::vector<float>& input, const std::vector<float>& weight,
    const std::vector<float>& bias,
    const std::vector<float>& output_gradient, int batch_size,
    int input_channels, int input_height, int input_width,
    int output_channels, int kernel_height, int kernel_width) {
  DeviceBuffer<float> device_input = CopyToDevice(input);
  DeviceBuffer<float> device_weight = CopyToDevice(weight);
  DeviceBuffer<float> device_bias = CopyToDevice(bias);
  DeviceBuffer<float> device_output(output_gradient.size());
  LaunchConvolutionForward(
      device_input.get(), device_weight.get(), device_bias.get(),
      device_output.get(), batch_size, input_channels, input_height,
      input_width, output_channels, kernel_height, kernel_width, nullptr);
  return DotObjective(CopyFromDevice(device_output), output_gradient);
}

void CheckProductionConvolutionFiniteDifferences(
    int input_channels, int input_height, int input_width,
    int output_channels, int offset) {
  constexpr int kBatchSize = 1;
  constexpr int kKernelSize = 5;
  constexpr float kEpsilon = 1.0e-3F;
  const int output_height = input_height - kKernelSize + 1;
  const int output_width = input_width - kKernelSize + 1;
  const std::vector<float> input = Pattern(
      static_cast<std::size_t>(kBatchSize) * input_channels * input_height *
          input_width,
      offset);
  std::vector<float> weight = Pattern(
      static_cast<std::size_t>(output_channels) * input_channels * kKernelSize *
          kKernelSize,
      offset + 1);
  const std::vector<float> bias = Pattern(output_channels, offset + 2);
  const std::vector<float> output_gradient = Pattern(
      static_cast<std::size_t>(kBatchSize) * output_channels * output_height *
          output_width,
      offset + 3);

  DeviceBuffer<float> device_input = CopyToDevice(input);
  DeviceBuffer<float> device_weight = CopyToDevice(weight);
  DeviceBuffer<float> device_output_gradient = CopyToDevice(output_gradient);
  DeviceBuffer<float> device_input_gradient(input.size());
  DeviceBuffer<float> device_weight_gradient(weight.size());
  DeviceBuffer<float> device_bias_gradient(bias.size());
  LaunchConvolutionBackward(
      device_input.get(), device_weight.get(), device_output_gradient.get(),
      device_input_gradient.get(), device_weight_gradient.get(),
      device_bias_gradient.get(), kBatchSize, input_channels, input_height,
      input_width, output_channels, kKernelSize, kKernelSize, nullptr);
  const std::vector<float> analytic =
      CopyFromDevice(device_weight_gradient);

  const std::vector<std::size_t> checked_indices{
      0, weight.size() / 2, weight.size() - 1};
  for (std::size_t index : checked_indices) {
    const float original = weight[index];
    weight[index] = original + kEpsilon;
    const double plus = ConvolutionDeviceObjective(
        input, weight, bias, output_gradient, kBatchSize, input_channels,
        input_height, input_width, output_channels, kKernelSize, kKernelSize);
    weight[index] = original - kEpsilon;
    const double minus = ConvolutionDeviceObjective(
        input, weight, bias, output_gradient, kBatchSize, input_channels,
        input_height, input_width, output_channels, kKernelSize, kKernelSize);
    weight[index] = original;
    const double numeric = (plus - minus) / (2.0 * kEpsilon);
    const double tolerance = 1.0e-2 + 1.0e-2 * std::fabs(numeric);
    EXPECT_NEAR(numeric, analytic[index], tolerance);
  }
}

struct SoftmaxReferenceResult {
  std::vector<float> probabilities;
  std::vector<float> losses;
  std::vector<float> gradients;
};

SoftmaxReferenceResult SoftmaxReference(
    const std::vector<float>& logits, const std::vector<std::uint8_t>& labels,
    int batch_size, int class_count) {
  SoftmaxReferenceResult result{
      std::vector<float>(logits.size()), std::vector<float>(batch_size),
      std::vector<float>(logits.size())};
  for (int sample = 0; sample < batch_size; ++sample) {
    const std::size_t base = static_cast<std::size_t>(sample) * class_count;
    double maximum = static_cast<double>(logits[base]);
    for (int class_index = 1; class_index < class_count; ++class_index) {
      maximum = std::fmax(maximum,
                          static_cast<double>(logits[base + class_index]));
    }
    double sum = 0.0;
    for (int class_index = 0; class_index < class_count; ++class_index) {
      sum += std::exp(static_cast<double>(logits[base + class_index]) -
                      maximum);
    }
    result.losses[sample] = static_cast<float>(
        std::log(sum) + maximum - logits[base + labels[sample]]);
    for (int class_index = 0; class_index < class_count; ++class_index) {
      const std::size_t index = base + class_index;
      const float probability = static_cast<float>(
          std::exp(static_cast<double>(logits[index]) - maximum) / sum);
      result.probabilities[index] = probability;
      result.gradients[index] =
          (probability - (class_index == labels[sample] ? 1.0F : 0.0F)) /
          static_cast<float>(batch_size);
    }
  }
  return result;
}

void CheckSoftmaxBatch(int batch_size) {
  constexpr int kClassCount = 4;
  std::vector<float> logits(static_cast<std::size_t>(batch_size) * kClassCount);
  std::vector<std::uint8_t> labels(batch_size);
  for (int sample = 0; sample < batch_size; ++sample) {
    labels[sample] = static_cast<std::uint8_t>((sample * 3 + 1) % kClassCount);
    for (int class_index = 0; class_index < kClassCount; ++class_index) {
      logits[static_cast<std::size_t>(sample) * kClassCount + class_index] =
          static_cast<float>((sample % 7) * 0.25 + class_index - 2);
    }
  }
  logits[0] = 1000.0F;
  logits[1] = -1000.0F;
  logits[2] = 999.0F;
  logits[3] = -999.0F;
  labels[0] = 1;
  const SoftmaxReferenceResult expected =
      SoftmaxReference(logits, labels, batch_size, kClassCount);

  DeviceBuffer<float> device_logits = CopyToDevice(logits);
  DeviceBuffer<std::uint8_t> device_labels = CopyToDevice(labels);
  DeviceBuffer<float> device_probabilities(logits.size());
  DeviceBuffer<float> device_losses(batch_size);
  DeviceBuffer<float> device_mean(1);
  DeviceBuffer<float> device_gradients(logits.size());
  LaunchSoftmaxCrossEntropy(
      device_logits.get(), device_labels.get(), device_probabilities.get(),
      device_losses.get(), device_mean.get(), device_gradients.get(), batch_size,
      kClassCount, nullptr);

  const std::vector<float> probabilities =
      CopyFromDevice(device_probabilities);
  const std::vector<float> losses = CopyFromDevice(device_losses);
  const std::vector<float> gradients = CopyFromDevice(device_gradients);
  const float mean = CopyFromDevice(device_mean)[0];
  for (int sample = 0; sample < batch_size; ++sample) {
    float probability_sum = 0.0F;
    for (int class_index = 0; class_index < kClassCount; ++class_index) {
      const std::size_t index =
          static_cast<std::size_t>(sample) * kClassCount + class_index;
      EXPECT_TRUE(std::isfinite(probabilities[index]));
      EXPECT_NEAR(expected.probabilities[index], probabilities[index], 2.0e-6F);
      EXPECT_NEAR(expected.gradients[index], gradients[index], 2.0e-6F);
      probability_sum += probabilities[index];
    }
    EXPECT_NEAR(1.0F, probability_sum, 2.0e-6F);
    EXPECT_TRUE(std::isfinite(losses[sample]));
    EXPECT_NEAR(expected.losses[sample], losses[sample], 2.0e-4F);
  }
  float fixed_order_sum = 0.0F;
  for (float loss : losses) {
    fixed_order_sum += loss;
  }
  EXPECT_EQ(fixed_order_sum / static_cast<float>(batch_size), mean);
}

struct AdamReferenceState {
  std::vector<double> parameters;
  std::vector<double> first_moments;
  std::vector<double> second_moments;
};

void AdamWReferenceStep(AdamReferenceState* state,
                        const std::vector<float>& gradients,
                        double learning_rate, double beta1, double beta2,
                        double epsilon, double weight_decay, int step) {
  const double inverse_bias_correction1 =
      1.0 / (1.0 - std::pow(beta1, step));
  const double inverse_bias_correction2 =
      1.0 / (1.0 - std::pow(beta2, step));
  for (std::size_t index = 0; index < gradients.size(); ++index) {
    const double parameter = state->parameters[index];
    const double gradient = gradients[index];
    state->first_moments[index] =
        beta1 * state->first_moments[index] + (1.0 - beta1) * gradient;
    state->second_moments[index] =
        beta2 * state->second_moments[index] +
        (1.0 - beta2) * gradient * gradient;
    const double corrected_first =
        state->first_moments[index] * inverse_bias_correction1;
    const double corrected_second =
        state->second_moments[index] * inverse_bias_correction2;
    state->parameters[index] =
        parameter - learning_rate *
                        (corrected_first /
                             (std::sqrt(corrected_second) + epsilon) +
                         weight_decay * parameter);
  }
}

void CheckAdamWSteps(int step_count, float weight_decay) {
  constexpr float kLearningRate = 0.003F;
  constexpr float kBeta1 = 0.9F;
  constexpr float kBeta2 = 0.999F;
  constexpr float kEpsilon = 1.0e-8F;
  const std::vector<float> initial_parameters{1.25F, -0.75F, 0.125F, -2.0F};
  const std::vector<float> gradients{0.2F, -0.4F, 0.05F, 1.5F};
  const std::vector<float> initial_first{0.0F, 0.1F, -0.2F, 0.3F};
  const std::vector<float> initial_second{0.0F, 0.04F, 0.09F, 0.16F};
  AdamReferenceState expected{
      std::vector<double>(initial_parameters.begin(), initial_parameters.end()),
      std::vector<double>(initial_first.begin(), initial_first.end()),
      std::vector<double>(initial_second.begin(), initial_second.end())};
  DeviceBuffer<float> device_parameters = CopyToDevice(initial_parameters);
  DeviceBuffer<float> device_gradients = CopyToDevice(gradients);
  DeviceBuffer<float> device_first = CopyToDevice(initial_first);
  DeviceBuffer<float> device_second = CopyToDevice(initial_second);

  for (int step = 1; step <= step_count; ++step) {
    const float inverse_bias_correction1 = static_cast<float>(
        1.0 / (1.0 - std::pow(static_cast<double>(kBeta1), step)));
    const float inverse_bias_correction2 = static_cast<float>(
        1.0 / (1.0 - std::pow(static_cast<double>(kBeta2), step)));
    LaunchAdamW(device_parameters.get(), device_gradients.get(),
                device_first.get(), device_second.get(), gradients.size(),
                kLearningRate, kBeta1, kBeta2, kEpsilon,
                inverse_bias_correction1, inverse_bias_correction2,
                weight_decay, nullptr);
    AdamWReferenceStep(&expected, gradients, kLearningRate, kBeta1, kBeta2,
                       kEpsilon, weight_decay, step);
  }

  const std::vector<float> parameters = CopyFromDevice(device_parameters);
  const std::vector<float> first = CopyFromDevice(device_first);
  const std::vector<float> second = CopyFromDevice(device_second);
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    EXPECT_NEAR(expected.parameters[index], parameters[index], 2.0e-6);
    EXPECT_NEAR(expected.first_moments[index], first[index], 2.0e-6);
    EXPECT_NEAR(expected.second_moments[index], second[index], 2.0e-6);
  }
}

void CheckFirstBadIndex(const std::vector<float>& values, int expected_index) {
  DeviceBuffer<float> device_values = CopyToDevice(values);
  DeviceBuffer<int> device_result(1);
  LaunchFindFirstNonFinite(device_values.get(), values.size(),
                           device_result.get(), nullptr);
  EXPECT_EQ(expected_index, CopyFromDevice(device_result)[0]);
}

std::string SourceWithoutCommentsOrLiterals(const std::string& source) {
  enum class State { kCode, kLineComment, kBlockComment, kString, kCharacter };
  State state = State::kCode;
  std::string code(source.size(), ' ');
  for (std::size_t index = 0; index < source.size(); ++index) {
    const char current = source[index];
    const char next = index + 1 < source.size() ? source[index + 1] : '\0';
    if (state == State::kCode && current == '/' && next == '/') {
      state = State::kLineComment;
      ++index;
    } else if (state == State::kCode && current == '/' && next == '*') {
      state = State::kBlockComment;
      ++index;
    } else if (state == State::kLineComment && current == '\n') {
      state = State::kCode;
      code[index] = current;
    } else if (state == State::kBlockComment && current == '*' && next == '/') {
      state = State::kCode;
      ++index;
    } else if (state == State::kCode && current == '"') {
      state = State::kString;
    } else if (state == State::kCode && current == '\'') {
      state = State::kCharacter;
    } else if ((state == State::kString || state == State::kCharacter) &&
               current == '\\') {
      ++index;
    } else if (state == State::kString && current == '"') {
      state = State::kCode;
    } else if (state == State::kCharacter && current == '\'') {
      state = State::kCode;
    } else if (state == State::kCode) {
      code[index] = current;
    }
  }
  return code;
}

bool ContainsIdentifier(const std::string& source,
                        const std::string& identifier) {
  for (std::size_t index = 0; index < source.size();) {
    const bool starts_identifier =
        (source[index] >= 'A' && source[index] <= 'Z') ||
        (source[index] >= 'a' && source[index] <= 'z') || source[index] == '_';
    if (!starts_identifier) {
      ++index;
      continue;
    }
    const std::size_t start = index++;
    while (index < source.size() &&
           ((source[index] >= 'A' && source[index] <= 'Z') ||
            (source[index] >= 'a' && source[index] <= 'z') ||
            (source[index] >= '0' && source[index] <= '9') ||
            source[index] == '_')) {
      ++index;
    }
    if (source.compare(start, index - start, identifier) == 0) {
      return true;
    }
  }
  return false;
}

}  // namespace

TEST_CASE(cuda_check_reports_expression_file_line_code_and_text) {
  const int expected_line = __LINE__ + 2;
  try {
    CUDA_CHECK(cudaErrorInvalidValue);
    test_harness::Fail(__FILE__, __LINE__, "CUDA_CHECK did not throw");
  } catch (const std::runtime_error& error) {
    const std::string message(error.what());
    EXPECT_TRUE(message.find("cudaErrorInvalidValue") != std::string::npos);
    EXPECT_TRUE(message.find("operator_tests.cu") != std::string::npos);
    EXPECT_TRUE(message.find("line=" + std::to_string(expected_line)) !=
                std::string::npos);
    EXPECT_TRUE(message.find("code=1") != std::string::npos);
    EXPECT_TRUE(message.find("invalid argument") != std::string::npos);
  }
}

TEST_CASE(cuda_check_detects_invalid_kernel_launch_without_synchronizing) {
  NoOpKernel<<<0, 1>>>();
  EXPECT_THROW_CONTAINS(CUDA_KERNEL_CHECK(), "cudaGetLastError()");
}

TEST_CASE(device_buffer_is_move_only) {
  EXPECT_TRUE(!std::is_copy_constructible<DeviceBuffer<float>>::value);
  EXPECT_TRUE(!std::is_copy_assignable<DeviceBuffer<float>>::value);
  EXPECT_TRUE(std::is_nothrow_move_constructible<DeviceBuffer<float>>::value);
  EXPECT_TRUE(std::is_nothrow_move_assignable<DeviceBuffer<float>>::value);
}

TEST_CASE(device_buffer_zero_count_does_not_allocate) {
  const std::size_t before = DeviceBufferAllocationCountForTests();
  const DeviceBuffer<float> empty(0);
  EXPECT_EQ(nullptr, empty.get());
  EXPECT_EQ(std::size_t{0}, empty.size());
  EXPECT_EQ(before, DeviceBufferAllocationCountForTests());
}

TEST_CASE(device_buffer_move_transfers_one_allocation) {
  const std::size_t before = DeviceBufferAllocationCountForTests();
  {
    DeviceBuffer<float> source(4);
    float* const pointer = source.get();
    DeviceBuffer<float> destination(std::move(source));
    EXPECT_EQ(nullptr, source.get());
    EXPECT_EQ(std::size_t{0}, source.size());
    EXPECT_EQ(pointer, destination.get());
    EXPECT_EQ(std::size_t{4}, destination.size());
    EXPECT_EQ(before + 1, DeviceBufferAllocationCountForTests());
  }
  EXPECT_EQ(before, DeviceBufferAllocationCountForTests());
}

TEST_CASE(device_buffer_move_assignment_releases_destination_then_transfers) {
  const std::size_t before = DeviceBufferAllocationCountForTests();
  {
    DeviceBuffer<float> source(4);
    DeviceBuffer<float> destination(2);
    float* const source_pointer = source.get();
    EXPECT_EQ(before + 2, DeviceBufferAllocationCountForTests());

    destination = std::move(source);

    EXPECT_EQ(nullptr, source.get());
    EXPECT_EQ(std::size_t{0}, source.size());
    EXPECT_EQ(source_pointer, destination.get());
    EXPECT_EQ(std::size_t{4}, destination.size());
    EXPECT_EQ(before + 1, DeviceBufferAllocationCountForTests());
  }
  EXPECT_EQ(before, DeviceBufferAllocationCountForTests());
}

TEST_CASE(device_buffer_round_trips_257_elements) {
  std::vector<float> source(257);
  for (std::size_t index = 0; index < source.size(); ++index) {
    source[index] = static_cast<float>(index) * 0.25F - 7.0F;
  }
  std::vector<float> result(source.size(), 0.0F);
  DeviceBuffer<float> device(source.size());

  CUDA_CHECK(cudaMemcpy(device.get(), source.data(),
                        source.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(result.data(), device.get(),
                        result.size() * sizeof(float), cudaMemcpyDeviceToHost));

  EXPECT_EQ(source, result);
}

TEST_CASE(input_evaluation_normalizes_without_translation) {
  constexpr int kImagePixels = 28 * 28;
  std::vector<std::uint8_t> images(kImagePixels, 17);
  images[0] = 0;
  images[1] = 255;
  const std::vector<std::uint32_t> original_indices{4000000000U};
  DeviceBuffer<std::uint8_t> device_images = CopyToDevice(images);
  DeviceBuffer<std::uint32_t> device_indices = CopyToDevice(original_indices);
  DeviceBuffer<float> device_output(images.size());

  LaunchNormalizeTranslate(device_images.get(), device_indices.get(),
                           device_output.get(), 1, UINT64_C(99), 0, false,
                           nullptr);
  const std::vector<float> actual = CopyFromDevice(device_output);

  EXPECT_NEAR(-0.42421293F, actual[0], 1.5e-4F);
  EXPECT_NEAR(2.8214865F, actual[1], 4.0e-4F);
  EXPECT_NEAR(NormalizedPixel(17), actual[2], 1.5e-4F);
}

TEST_CASE(input_translation_matches_cpu_for_batch_17) {
  CheckInputTranslation(17);
}

TEST_CASE(input_translation_matches_cpu_for_batch_257) {
  CheckInputTranslation(257);
}

TEST_CASE(input_rejects_impossible_launch_dimensions) {
  const auto* const images = reinterpret_cast<const std::uint8_t*>(1);
  const auto* const indices = reinterpret_cast<const std::uint32_t*>(1);
  auto* const output = reinterpret_cast<float*>(1);
  EXPECT_THROW_CONTAINS(LaunchNormalizeTranslate(
                            images, indices, output, 0, 1, 1, true, nullptr),
                        "batch_size");
  EXPECT_THROW_CONTAINS(LaunchNormalizeTranslate(
                            images, indices, output, 1, 1, 0, true, nullptr),
                        "one_based_epoch");
}

TEST_CASE(relu_forward_backward_matches_cpu_and_backward_aliases_gradient) {
  constexpr std::size_t kCount = 257;
  std::vector<float> input(kCount);
  std::vector<float> output_gradient(kCount);
  for (std::size_t index = 0; index < kCount; ++index) {
    input[index] = static_cast<float>(static_cast<int>(index % 11) - 5);
    output_gradient[index] = static_cast<float>(index) * 0.125F - 9.0F;
  }
  input[0] = -2.0F;
  input[1] = -0.0F;
  input[2] = 0.0F;
  input[3] = 3.0F;
  const std::vector<float> expected_forward =
      cpu_reference::ReluForward(input);
  const std::vector<float> expected_backward =
      cpu_reference::ReluBackward(input, output_gradient);
  DeviceBuffer<float> device_input = CopyToDevice(input);
  DeviceBuffer<float> device_forward(kCount);
  DeviceBuffer<float> device_gradient = CopyToDevice(output_gradient);

  LaunchReluForward(device_input.get(), device_forward.get(), kCount, nullptr);
  LaunchReluBackward(device_input.get(), device_gradient.get(),
                     device_gradient.get(), kCount, nullptr);

  ExpectNearVectors(expected_forward, CopyFromDevice(device_forward));
  const std::vector<float> actual_backward = CopyFromDevice(device_gradient);
  ExpectNearVectors(expected_backward, actual_backward);
  EXPECT_EQ(0.0F, actual_backward[1]);
  EXPECT_EQ(0.0F, actual_backward[2]);
  EXPECT_EQ(output_gradient[3], actual_backward[3]);
}

TEST_CASE(relu_zero_covers_non_divisible_element_count) {
  constexpr std::size_t kCount = 257;
  const std::vector<float> nonzero(kCount, 23.0F);
  DeviceBuffer<float> device_values = CopyToDevice(nonzero);

  LaunchZero(device_values.get(), kCount, nullptr);

  EXPECT_EQ(std::vector<float>(kCount, 0.0F),
            CopyFromDevice(device_values));
}

TEST_CASE(relu_launchers_reject_impossible_element_counts) {
  const std::size_t impossible = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW_CONTAINS(LaunchZero(nullptr, impossible, nullptr), "count");
  EXPECT_THROW_CONTAINS(
      LaunchReluForward(nullptr, nullptr, impossible, nullptr), "count");
  EXPECT_THROW_CONTAINS(
      LaunchReluBackward(nullptr, nullptr, nullptr, impossible, nullptr),
      "count");
}

TEST_CASE(maxpool_forward_uses_first_tie_and_row_major_offsets) {
  const std::vector<float> input{
      5.0F, 5.0F, 1.0F, 7.0F,
      1.0F, 0.0F, 2.0F, 3.0F,
      1.0F, 2.0F, 1.0F, 2.0F,
      9.0F, 4.0F, 3.0F, 8.0F};
  const cpu_reference::MaxPoolResult expected =
      cpu_reference::MaxPoolForward(input, 1, 1, 4, 4);
  DeviceBuffer<float> device_input = CopyToDevice(input);
  DeviceBuffer<float> device_output(expected.output.size());
  DeviceBuffer<std::uint8_t> device_offsets(expected.winner_offsets.size());

  LaunchMaxPoolForward(device_input.get(), device_output.get(),
                       device_offsets.get(), 1, 1, 4, 4, nullptr);

  ExpectNearVectors(expected.output, CopyFromDevice(device_output));
  const std::vector<std::uint8_t> actual_offsets =
      CopyFromDevice(device_offsets);
  EXPECT_EQ(std::vector<std::uint8_t>({0, 1, 2, 3}), actual_offsets);
  EXPECT_EQ(expected.winner_offsets, actual_offsets);
}

TEST_CASE(maxpool_backward_zeroes_nonwinners_and_writes_only_winners) {
  const std::vector<float> output_gradient{10.0F, 20.0F, 30.0F, 40.0F};
  const std::vector<std::uint8_t> winner_offsets{0, 1, 2, 3};
  const std::vector<float> expected = cpu_reference::MaxPoolBackward(
      output_gradient, winner_offsets, 1, 1, 4, 4);
  DeviceBuffer<float> device_output_gradient = CopyToDevice(output_gradient);
  DeviceBuffer<std::uint8_t> device_offsets = CopyToDevice(winner_offsets);
  DeviceBuffer<float> device_input_gradient =
      CopyToDevice(std::vector<float>(16, -77.0F));

  LaunchMaxPoolBackward(device_output_gradient.get(), device_offsets.get(),
                        device_input_gradient.get(), 1, 1, 4, 4, nullptr);

  const std::vector<float> actual = CopyFromDevice(device_input_gradient);
  ExpectNearVectors(expected, actual);
  EXPECT_EQ(std::vector<float>({10.0F, 0.0F, 0.0F, 20.0F,
                                0.0F, 0.0F, 0.0F, 0.0F,
                                0.0F, 0.0F, 0.0F, 0.0F,
                                30.0F, 0.0F, 0.0F, 40.0F}),
            actual);
}

TEST_CASE(maxpool_rejects_impossible_dimensions_before_launch) {
  const auto* const input = reinterpret_cast<const float*>(1);
  auto* const output = reinterpret_cast<float*>(1);
  auto* const offsets = reinterpret_cast<std::uint8_t*>(1);
  EXPECT_THROW_CONTAINS(LaunchMaxPoolForward(
                            input, output, offsets, 1, 1, 3, 4, nullptr),
                        "dimensions");
  EXPECT_THROW_CONTAINS(LaunchMaxPoolForward(
                            input, output, offsets,
                            std::numeric_limits<int>::max(),
                            std::numeric_limits<int>::max(), 2, 2, nullptr),
                         "overflow");
}

TEST_CASE(linear_forward_backward_matches_cpu_for_small_shape) {
  CheckLinearAgainstCpu(3, 5, 4);
}

TEST_CASE(linear_forward_backward_matches_cpu_for_non_divisible_shape) {
  CheckLinearAgainstCpu(17, 120, 84);
}

TEST_CASE(linear_one_element_forward_backward_boundary) {
  CheckLinearAgainstCpu(1, 1, 1);
}

TEST_CASE(linear_production_weights_match_centered_finite_differences) {
  // The scalar objective is L=sum_i forward[i]*output_gradient[i], exactly the
  // upstream gradient supplied to backward.
  CheckProductionLinearFiniteDifferences(256, 120, 9);
  CheckProductionLinearFiniteDifferences(120, 84, 13);
  CheckProductionLinearFiniteDifferences(84, 10, 17);
}

TEST_CASE(linear_launchers_reject_invalid_dimensions_and_pointers) {
  const auto* const read = reinterpret_cast<const float*>(1);
  auto* const write = reinterpret_cast<float*>(1);
  EXPECT_THROW_CONTAINS(
      LaunchLinearForward(read, read, read, write, 0, 1, 1, nullptr),
      "dimensions");
  EXPECT_THROW_CONTAINS(
      LaunchLinearForward(nullptr, read, read, write, 1, 1, 1, nullptr),
      "pointer");
  EXPECT_THROW_CONTAINS(
      LaunchLinearBackward(read, read, read, write, write, write, 1, 1, 1,
                           nullptr),
      "overlap");
  EXPECT_THROW_CONTAINS(
      LaunchLinearForward(
          reinterpret_cast<const float*>(
              std::numeric_limits<std::uintptr_t>::max() - 1),
          read, read, write, 1, 1, 1, nullptr),
      "pointer range");
}

TEST_CASE(convolution_forward_backward_matches_cpu_for_non_divisible_shape) {
  CheckConvolutionAgainstCpu(2, 2, 7, 8, 3, 5, 5);
}

TEST_CASE(convolution_forward_backward_matches_cpu_for_conv1_shape) {
  CheckConvolutionAgainstCpu(1, 1, 28, 28, 6, 5, 5);
}

TEST_CASE(convolution_forward_backward_matches_cpu_for_conv2_shape) {
  CheckConvolutionAgainstCpu(1, 6, 12, 12, 16, 5, 5);
}

TEST_CASE(convolution_one_element_forward_backward_boundary) {
  CheckConvolutionAgainstCpu(1, 1, 1, 1, 1, 1, 1);
}

TEST_CASE(convolution_production_weights_match_centered_finite_differences) {
  // The scalar objective is L=sum_i forward[i]*output_gradient[i], exactly the
  // upstream gradient supplied to backward.
  CheckProductionConvolutionFiniteDifferences(1, 28, 28, 6, 21);
  CheckProductionConvolutionFiniteDifferences(6, 12, 12, 16, 25);
}

TEST_CASE(convolution_launchers_reject_invalid_dimensions_and_pointers) {
  const auto* const read = reinterpret_cast<const float*>(1);
  auto* const write = reinterpret_cast<float*>(1);
  EXPECT_THROW_CONTAINS(LaunchConvolutionForward(
                            read, read, read, write, 1, 1, 4, 5, 1, 5, 5,
                            nullptr),
                        "kernel");
  EXPECT_THROW_CONTAINS(LaunchConvolutionForward(
                            nullptr, read, read, write, 1, 1, 5, 5, 1, 5, 5,
                            nullptr),
                        "pointer");
  EXPECT_THROW_CONTAINS(LaunchConvolutionBackward(
                            read, read, read, write, write, write, 1, 1, 5, 5,
                            1, 5, 5, nullptr),
                        "overlap");
  EXPECT_THROW_CONTAINS(LaunchConvolutionForward(
                            read, read, read, write,
                            std::numeric_limits<int>::max(),
                            std::numeric_limits<int>::max(), 5, 5, 1, 5, 5,
                            nullptr),
                        "overflow");
}

TEST_CASE(convolution_gradient_source_policy_forbids_atomic_operations) {
  std::ifstream input("src/kernels/convolution.cu", std::ios::binary);
  EXPECT_TRUE(input.is_open());
  std::ostringstream contents;
  contents << input.rdbuf();
  const std::string code =
      SourceWithoutCommentsOrLiterals(contents.str());
  EXPECT_TRUE(!ContainsIdentifier(code, "atomicAdd"));
}

TEST_CASE(softmax_cross_entropy_is_stable_and_scaled_for_boundary_batches) {
  CheckSoftmaxBatch(1);
  CheckSoftmaxBatch(17);
  CheckSoftmaxBatch(128);
  CheckSoftmaxBatch(129);
}

TEST_CASE(softmax_inference_is_stable_and_label_independent) {
  constexpr int kBatchSize = 2;
  constexpr int kClassCount = 3;
  const std::vector<float> logits{1000.0F, 999.0F, -1000.0F,
                                  -1000.0F, -999.0F, 1000.0F};
  const std::vector<std::uint8_t> dummy_labels{0, 0};
  const std::vector<float> expected =
      SoftmaxReference(logits, dummy_labels, kBatchSize, kClassCount)
          .probabilities;
  DeviceBuffer<float> device_logits = CopyToDevice(logits);
  DeviceBuffer<float> device_probabilities(logits.size());

  LaunchSoftmax(device_logits.get(), device_probabilities.get(), kBatchSize,
                kClassCount, nullptr);

  const std::vector<float> actual = CopyFromDevice(device_probabilities);
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_TRUE(std::isfinite(actual[index]));
    EXPECT_NEAR(expected[index], actual[index], 2.0e-6F);
  }
}

TEST_CASE(softmax_launchers_reject_invalid_shapes_pointers_and_overlap) {
  const auto* const labels = reinterpret_cast<const std::uint8_t*>(1);
  const auto* const read = reinterpret_cast<const float*>(16);
  auto* const write = reinterpret_cast<float*>(64);
  auto* const losses = reinterpret_cast<float*>(128);
  auto* const mean = reinterpret_cast<float*>(192);
  auto* const gradients = reinterpret_cast<float*>(256);
  EXPECT_THROW_CONTAINS(
      LaunchSoftmax(read, write, 0, 10, nullptr), "dimensions");
  EXPECT_THROW_CONTAINS(
      LaunchSoftmax(read, write, 1, 257, nullptr), "class_count");
  EXPECT_THROW_CONTAINS(
      LaunchSoftmax(nullptr, write, 1, 10, nullptr), "pointer");
  EXPECT_THROW_CONTAINS(
      LaunchSoftmax(read, const_cast<float*>(read), 1, 1, nullptr), "overlap");
  EXPECT_THROW_CONTAINS(
      LaunchSoftmaxCrossEntropy(
          reinterpret_cast<const float*>(
              std::numeric_limits<std::uintptr_t>::max() - 1),
          labels, write, losses, mean, gradients, 1, 1, nullptr),
      "pointer range");
}

TEST_CASE(metrics_argmax_uses_smallest_tie_and_reduces_exact_count) {
  constexpr int kBatchSize = 129;
  constexpr int kClassCount = 4;
  std::vector<float> logits(static_cast<std::size_t>(kBatchSize) * kClassCount);
  std::vector<std::uint8_t> labels(kBatchSize);
  std::vector<std::uint8_t> expected_predictions(kBatchSize);
  std::vector<int> expected_flags(kBatchSize);
  int expected_count = 0;
  for (int sample = 0; sample < kBatchSize; ++sample) {
    const int winner = sample % kClassCount;
    const std::size_t base = static_cast<std::size_t>(sample) * kClassCount;
    for (int class_index = 0; class_index < kClassCount; ++class_index) {
      logits[base + class_index] = static_cast<float>(class_index - 7);
    }
    logits[base + winner] = 20.0F;
    if (winner + 1 < kClassCount) {
      logits[base + winner + 1] = 20.0F;
    }
    expected_predictions[sample] = static_cast<std::uint8_t>(winner);
    labels[sample] = static_cast<std::uint8_t>((sample % 3 == 0)
                                                   ? winner
                                                   : (winner + 1) % kClassCount);
    expected_flags[sample] = labels[sample] == winner ? 1 : 0;
    expected_count += expected_flags[sample];
  }
  DeviceBuffer<float> device_logits = CopyToDevice(logits);
  DeviceBuffer<std::uint8_t> device_labels = CopyToDevice(labels);
  DeviceBuffer<std::uint8_t> device_predictions(kBatchSize);
  DeviceBuffer<int> device_flags(kBatchSize);
  DeviceBuffer<int> device_count(1);

  LaunchArgmaxAndCountCorrect(
      device_logits.get(), device_labels.get(), device_predictions.get(),
      device_flags.get(), device_count.get(), kBatchSize, kClassCount, nullptr);

  EXPECT_EQ(expected_predictions, CopyFromDevice(device_predictions));
  EXPECT_EQ(expected_flags, CopyFromDevice(device_flags));
  EXPECT_EQ(expected_count, CopyFromDevice(device_count)[0]);
}

TEST_CASE(metrics_launcher_rejects_invalid_shapes_pointers_and_overlap) {
  const auto* const logits = reinterpret_cast<const float*>(16);
  const auto* const labels = reinterpret_cast<const std::uint8_t*>(64);
  auto* const predictions = reinterpret_cast<std::uint8_t*>(96);
  auto* const flags = reinterpret_cast<int*>(128);
  auto* const count = reinterpret_cast<int*>(256);
  EXPECT_THROW_CONTAINS(LaunchArgmaxAndCountCorrect(
                            logits, labels, predictions, flags, count, 1, 0,
                            nullptr),
                        "dimensions");
  EXPECT_THROW_CONTAINS(LaunchArgmaxAndCountCorrect(
                            nullptr, labels, predictions, flags, count, 1, 10,
                            nullptr),
                        "pointer");
  EXPECT_THROW_CONTAINS(LaunchArgmaxAndCountCorrect(
                            logits, labels, reinterpret_cast<std::uint8_t*>(flags),
                            flags, count, 1, 10, nullptr),
                        "overlap");
}

TEST_CASE(adamw_matches_fp64_reference_for_first_and_tenth_steps) {
  CheckAdamWSteps(1, 0.01F);
  CheckAdamWSteps(10, 0.01F);
}

TEST_CASE(adamw_zero_decay_matches_fp64_bias_update) {
  CheckAdamWSteps(10, 0.0F);
}

TEST_CASE(adamw_handles_zero_and_rejects_invalid_arguments) {
  LaunchAdamW(nullptr, nullptr, nullptr, nullptr, 0, 0.001F, 0.9F, 0.999F,
              1.0e-8F, 10.0F, 1000.0F, 0.0F, nullptr);
  auto* const write = reinterpret_cast<float*>(16);
  const auto* const read = reinterpret_cast<const float*>(64);
  auto* const other_write = reinterpret_cast<float*>(48);
  auto* const final_write = reinterpret_cast<float*>(80);
  EXPECT_THROW_CONTAINS(LaunchAdamW(write, read, write, other_write, 1,
                                    0.001F, 0.9F, 0.999F, 1.0e-8F, 10.0F,
                                    1000.0F, 0.01F, nullptr),
                        "overlap");
  EXPECT_THROW_CONTAINS(LaunchAdamW(write, read, other_write, final_write, 1,
                                    -0.001F, 0.9F, 0.999F, 1.0e-8F, 10.0F,
                                    1000.0F, 0.01F, nullptr),
                        "learning_rate");
}

TEST_CASE(finite_scan_reports_all_finite_sentinel_for_zero_and_257_values) {
  DeviceBuffer<int> device_result(1);
  LaunchFindFirstNonFinite(nullptr, 0, device_result.get(), nullptr);
  EXPECT_EQ(0, CopyFromDevice(device_result)[0]);

  const std::vector<float> finite_values(257, -3.25F);
  DeviceBuffer<float> device_values = CopyToDevice(finite_values);
  LaunchFindFirstNonFinite(device_values.get(), finite_values.size(),
                           device_result.get(), nullptr);
  EXPECT_EQ(257, CopyFromDevice(device_result)[0]);
}

TEST_CASE(finite_scan_reports_smallest_nan_or_infinity_index) {
  std::vector<float> values(257, 1.0F);
  values[193] = -std::numeric_limits<float>::infinity();
  values[17] = std::numeric_limits<float>::infinity();
  values[128] = std::numeric_limits<float>::quiet_NaN();
  CheckFirstBadIndex(values, 17);
}

TEST_CASE(finite_scan_detects_each_non_finite_classification) {
  std::vector<float> values(257, 1.0F);
  values[31] = std::numeric_limits<float>::quiet_NaN();
  CheckFirstBadIndex(values, 31);
  values[31] = 1.0F;
  values[129] = std::numeric_limits<float>::infinity();
  CheckFirstBadIndex(values, 129);
  values[129] = 1.0F;
  values[256] = -std::numeric_limits<float>::infinity();
  CheckFirstBadIndex(values, 256);
}

TEST_CASE(finite_scan_rejects_invalid_count_pointers_and_overlap) {
  auto* const result = reinterpret_cast<int*>(64);
  const auto* const values = reinterpret_cast<const float*>(64);
  EXPECT_THROW_CONTAINS(
      LaunchFindFirstNonFinite(nullptr, 1, result, nullptr), "values");
  EXPECT_THROW_CONTAINS(
      LaunchFindFirstNonFinite(values, 1, nullptr, nullptr), "result");
  EXPECT_THROW_CONTAINS(
      LaunchFindFirstNonFinite(values, 1, result, nullptr), "overlap");
  EXPECT_THROW_CONTAINS(
      LaunchFindFirstNonFinite(
          values,
          static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1U,
          result, nullptr),
      "INT_MAX");
}
