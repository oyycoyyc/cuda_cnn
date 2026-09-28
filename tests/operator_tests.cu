#define CUDA_LENET_ENABLE_TEST_HOOKS

#include "cuda_check.h"
#include "tensor.h"
#include "test_harness.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
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
