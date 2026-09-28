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
#include <limits>
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
