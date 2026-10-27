// Tests-first contract for course/gpu_ops.h and course/inference.h.
//
// Every assertion below is written against the approved public declarations
// before course/gpu_ops.cu and course/inference.cu exist, so the suite is
// expected to fail at link time until those implementations are provided.
// Oracles are the original reference LeNet, the original
// TranslationOffsetForSample helper, and explicit host scalar fixtures derived
// from the documented mode formulas. Nothing here is a production CPU fallback.

#define CUDA_LENET_ENABLE_TEST_HOOKS

#include "course/gpu_ops.h"
#include "course/inference.h"
#include "course/tests/test_support.h"

#include "cuda_check.h"
#include "lenet.h"
#include "parameters.h"
#include "random.h"
#include "tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace {

using course::LaunchArgmax;
using course::LaunchPrepareInput;
using course_test_support::SameBits;

// Supported course geometry.
constexpr int kImageSide = 28;
constexpr int kImagePixels = kImageSide * kImageSide;
constexpr int kClassCount = 10;

// Fixed normalization constants from the model contract.
constexpr float kNormalizationMean = 0.1307F;
constexpr float kNormalizationStddev = 0.3081F;

// Copies a host vector to a newly allocated device buffer.
template <typename T>
DeviceBuffer<T> CopyToDevice(const std::vector<T>& source) {
  DeviceBuffer<T> destination(source.size());
  if (!source.empty()) {
    CUDA_CHECK(cudaMemcpy(destination.get(), source.data(),
                          source.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  return destination;
}

// Synchronizes the stream, then transfers count device elements to the host.
template <typename T>
std::vector<T> CopyFromDevice(const T* source, std::size_t count,
                              cudaStream_t stream) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<T> destination(count);
  if (count != 0) {
    CUDA_CHECK(cudaMemcpy(destination.data(), source, count * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }
  return destination;
}

// Overwrites a device logits buffer from a host fixture.
void UploadLogits(const std::vector<float>& source, DeviceBuffer<float>* target) {
  CUDA_CHECK(cudaMemcpy(target->get(), source.data(),
                        source.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

// Zeroes the sticky bad-logit flag exactly once for a request, as the caller
// contract requires; LaunchArgmax itself must never clear it.
void ZeroBadFlag(DeviceBuffer<int>* flag, cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(flag->get(), 0, sizeof(int), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// RAII non-blocking stream so the borrowed-stream contract is exercised
// without relying on legacy default-stream ordering.
class NonblockingTestStream {
 public:
  NonblockingTestStream() : stream_(nullptr) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  ~NonblockingTestStream() {
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  NonblockingTestStream(const NonblockingTestStream&) = delete;
  NonblockingTestStream& operator=(const NonblockingTestStream&) = delete;

  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_;
};

// Documented kUnit transform: (value - 0.1307F) / 0.3081F.
float NormalizedUnit(float value) {
  return (value - kNormalizationMean) / kNormalizationStddev;
}

// Documented kByte transform: (value / 255 - 0.1307F) / 0.3081F.
float NormalizedByte(float value) {
  return (value / 255.0F - kNormalizationMean) / kNormalizationStddev;
}

// A black pixel is a value of zero; its normalized form is used as translation
// padding in every mode.
float NormalizedBlackPixel() { return NormalizedUnit(0.0F); }

// Host fixture for one already-prepared output value. kNormalized passes
// source values through unchanged.
float PreparedPixel(float source, course::InputMode mode) {
  switch (mode) {
    case course::InputMode::kUnit:
      return NormalizedUnit(source);
    case course::InputMode::kByte:
      return NormalizedByte(source);
    case course::InputMode::kNormalized:
      return source;
  }
  return 0.0F;
}

// Builds one valid packed FP32 fixture for a mode. The leading values include
// fractional and boundary cases; the remainder is a deterministic in-range
// pattern. No value is ever routed through uint8.
std::vector<float> PackedFixture(std::size_t count, course::InputMode mode) {
  const float unit_head[] = {0.0F,
                             -0.0F,
                             1.0F,
                             1.0F / 3.0F,
                             0.1F,
                             0.99999994F,
                             std::numeric_limits<float>::denorm_min(),
                             0.5F};
  const float byte_head[] = {0.0F, 255.0F, 127.5F, 0.25F, 254.75F, 128.5F,
                             1.0F, 200.5F};
  const float normalized_head[] = {-2.5F,
                                   3.5F,
                                   -1.0e-8F,
                                   std::numeric_limits<float>::denorm_min(),
                                   -0.0F,
                                   1.0F,
                                   -1.0F,
                                   0.25F};
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    switch (mode) {
      case course::InputMode::kUnit:
        values[index] =
            static_cast<float>((index * 29 + 7) % 1000) / 1000.0F;
        break;
      case course::InputMode::kByte:
        values[index] =
            static_cast<float>((index * 37 + 11) % 255001) / 1000.0F;
        break;
      case course::InputMode::kNormalized:
        values[index] =
            static_cast<float>(static_cast<int>((index * 53 + 13) % 401) -
                               200) *
            0.01F;
        break;
    }
  }
  const float* head = nullptr;
  std::size_t head_count = 0;
  switch (mode) {
    case course::InputMode::kUnit:
      head = unit_head;
      head_count = sizeof(unit_head) / sizeof(unit_head[0]);
      break;
    case course::InputMode::kByte:
      head = byte_head;
      head_count = sizeof(byte_head) / sizeof(byte_head[0]);
      break;
    case course::InputMode::kNormalized:
      head = normalized_head;
      head_count = sizeof(normalized_head) / sizeof(normalized_head[0]);
      break;
  }
  for (std::size_t index = 0; index < head_count && index < values.size();
       ++index) {
    values[index] = head[index];
  }
  return values;
}

// Requires one kByte output element to match its full-precision FP32 fixture
// and to differ from both plausible uint8 fixtures (truncation and rounding).
void ExpectNoUint8Quantization(const std::vector<float>& packed,
                               const std::vector<float>& actual,
                               std::size_t index) {
  const float value = packed[index];
  const std::uint8_t truncated = static_cast<std::uint8_t>(value);
  const std::uint8_t rounded = static_cast<std::uint8_t>(value + 0.5F);
  EXPECT_TRUE(SameBits(NormalizedByte(value), actual[index]));
  EXPECT_FALSE(
      SameBits(NormalizedByte(static_cast<float>(truncated)), actual[index]));
  EXPECT_FALSE(
      SameBits(NormalizedByte(static_cast<float>(rounded)), actual[index]));
}

// Smallest-index argmax oracle used only to pin expected class integers from
// reference-model logits; equal values keep the earlier index.
std::int32_t HostArgmax(const float* row, int classes) {
  std::int32_t best = 0;
  for (int class_index = 1; class_index < classes; ++class_index) {
    if (row[class_index] > row[best]) {
      best = class_index;
    }
  }
  return best;
}

// Deterministic, schema-valid parameter fixture with small finite values.
ParameterSet PatternParameters() {
  ParameterSet parameters = CreateLenetParameters();
  std::size_t ordinal = 0;
  for (ParameterTensor& tensor : parameters) {
    for (float& value : tensor.values) {
      const int centered =
          static_cast<int>((ordinal * 37) % 97) - 48;
      value = static_cast<float>(centered) * 0.01F;
      ++ordinal;
    }
  }
  return parameters;
}

// Deterministic pre-normalized FP32 input fixture in the normalized domain.
std::vector<float> NormalizedInput(int batch) {
  const std::size_t count =
      static_cast<std::size_t>(batch) * kImagePixels;
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    values[index] =
        static_cast<float>(static_cast<int>((index * 53 + 11) % 211) - 105) *
        0.01F;
  }
  return values;
}

// Copies one row of logits for comparison and argmax classification.
std::vector<float> RowCopy(const std::vector<float>& logits, int sample) {
  const std::size_t begin =
      static_cast<std::size_t>(sample) * kClassCount;
  return std::vector<float>(logits.begin() + begin,
                            logits.begin() + begin + kClassCount);
}

// ---------------------------------------------------------------------------
// Input preparation: modes, exact FP32 values, and translation
// ---------------------------------------------------------------------------

TEST_CASE(float_input_modes_and_translation) {
  constexpr int kBatch = 6;
  const std::size_t kPixelCount =
      static_cast<std::size_t>(kBatch) * kImagePixels;
  const course::InputMode modes[] = {course::InputMode::kUnit,
                                     course::InputMode::kByte,
                                     course::InputMode::kNormalized};

  // No augmentation: every packed value must produce the exact documented
  // scalar fixture, bit for bit, including fractional and boundary values.
  for (course::InputMode mode : modes) {
    const std::vector<float> packed = PackedFixture(kPixelCount, mode);
    DeviceBuffer<float> device_packed = CopyToDevice(packed);
    DeviceBuffer<float> device_output(kPixelCount);
    LaunchPrepareInput(device_packed.get(), nullptr, device_output.get(),
                       kBatch, mode, UINT64_C(0), 0, false, nullptr);
    const std::vector<float> actual =
        CopyFromDevice(device_output.get(), kPixelCount, nullptr);
    EXPECT_EQ(packed.size(), actual.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
      EXPECT_TRUE(SameBits(PreparedPixel(packed[index], mode), actual[index]));
    }
  }

  // kUnit fractional values are consumed directly; they must not be quantized.
  {
    const std::vector<float> packed =
        PackedFixture(kPixelCount, course::InputMode::kUnit);
    DeviceBuffer<float> device_packed = CopyToDevice(packed);
    DeviceBuffer<float> device_output(kPixelCount);
    LaunchPrepareInput(device_packed.get(), nullptr, device_output.get(),
                       kBatch, course::InputMode::kUnit, UINT64_C(0), 0, false,
                       nullptr);
    const std::vector<float> actual =
        CopyFromDevice(device_output.get(), kPixelCount, nullptr);
    EXPECT_TRUE(SameBits(NormalizedUnit(1.0F / 3.0F), actual[3]));
    EXPECT_TRUE(SameBits(NormalizedUnit(0.1F), actual[4]));
  }

  // kByte fractional values stay FP32: neither truncation nor rounding to a
  // uint8 pixel may reproduce the result.
  {
    const std::vector<float> packed =
        PackedFixture(kPixelCount, course::InputMode::kByte);
    DeviceBuffer<float> device_packed = CopyToDevice(packed);
    DeviceBuffer<float> device_output(kPixelCount);
    LaunchPrepareInput(device_packed.get(), nullptr, device_output.get(),
                       kBatch, course::InputMode::kByte, UINT64_C(0), 0, false,
                       nullptr);
    const std::vector<float> actual =
        CopyFromDevice(device_output.get(), kPixelCount, nullptr);
    EXPECT_EQ(127.5F, packed[2]);
    EXPECT_EQ(254.75F, packed[4]);
    EXPECT_EQ(128.5F, packed[5]);
    ExpectNoUint8Quantization(packed, actual, 2);
    ExpectNoUint8Quantization(packed, actual, 4);
    ExpectNoUint8Quantization(packed, actual, 5);
  }

  // kNormalized values, including negatives, pass through bit for bit.
  {
    const std::vector<float> packed =
        PackedFixture(kPixelCount, course::InputMode::kNormalized);
    DeviceBuffer<float> device_packed = CopyToDevice(packed);
    DeviceBuffer<float> device_output(kPixelCount);
    LaunchPrepareInput(device_packed.get(), nullptr, device_output.get(),
                       kBatch, course::InputMode::kNormalized, UINT64_C(0), 0,
                       false, nullptr);
    const std::vector<float> actual =
        CopyFromDevice(device_output.get(), kPixelCount, nullptr);
    EXPECT_TRUE(SameBits(-2.5F, actual[0]));
    EXPECT_TRUE(SameBits(3.5F, actual[1]));
    EXPECT_TRUE(SameBits(-0.0F, actual[4]));
  }

  // Augmented translation: the original helper, not the shuffled position,
  // selects each offset. These known fixtures have nonzero offsets, and rows
  // 0, 1, and 4 share original index 4.
  constexpr std::uint64_t kSeed = UINT64_C(0x8877665544332211);
  constexpr std::uint32_t kEpoch = 7;
  const std::vector<std::uint32_t> indices{4, 4, 5, 4000000000U, 4, 9};
  const TranslationOffset moving =
      TranslationOffsetForSample(kSeed, kEpoch, indices[0]);
  const TranslationOffset other =
      TranslationOffsetForSample(kSeed, kEpoch, indices[2]);
  EXPECT_EQ(2, moving.dx);
  EXPECT_EQ(-2, moving.dy);
  EXPECT_EQ(-2, other.dx);
  EXPECT_EQ(0, other.dy);
  EXPECT_TRUE(moving.dx != 0 || moving.dy != 0);
  EXPECT_TRUE(moving.dx != other.dx || moving.dy != other.dy);

  const DeviceBuffer<std::uint32_t> device_indices = CopyToDevice(indices);
  for (course::InputMode mode : modes) {
    // The index-consistency fixture repeats one identical nonconstant source
    // image on every row: equal original indices must select the same
    // translation without replacing the underlying source pixels.
    const std::vector<float> image = PackedFixture(kImagePixels, mode);
    std::vector<float> packed(kPixelCount);
    for (int sample = 0; sample < kBatch; ++sample) {
      std::memcpy(&packed[static_cast<std::size_t>(sample) * kImagePixels],
                  image.data(), kImagePixels * sizeof(float));
    }
    DeviceBuffer<float> device_packed = CopyToDevice(packed);
    DeviceBuffer<float> device_output(kPixelCount);
    LaunchPrepareInput(device_packed.get(), device_indices.get(),
                       device_output.get(), kBatch, mode, kSeed, kEpoch, true,
                       nullptr);
    const std::vector<float> actual =
        CopyFromDevice(device_output.get(), kPixelCount, nullptr);

    std::vector<float> expected(kPixelCount);
    bool saw_padding = false;
    std::size_t first_padding_index = 0;
    for (int sample = 0; sample < kBatch; ++sample) {
      const TranslationOffset offset = TranslationOffsetForSample(
          kSeed, kEpoch, indices[static_cast<std::size_t>(sample)]);
      const std::size_t sample_base =
          static_cast<std::size_t>(sample) * kImagePixels;
      for (int y = 0; y < kImageSide; ++y) {
        for (int x = 0; x < kImageSide; ++x) {
          // Positive dx/dy moves content right/down: output samples x-dx/y-dy.
          const int source_x = x - offset.dx;
          const int source_y = y - offset.dy;
          const std::size_t pixel_index =
              sample_base + static_cast<std::size_t>(y) * kImageSide + x;
          if (source_x >= 0 && source_x < kImageSide && source_y >= 0 &&
              source_y < kImageSide) {
            expected[pixel_index] = PreparedPixel(
                packed[sample_base + static_cast<std::size_t>(source_y) *
                                         kImageSide +
                       static_cast<std::size_t>(source_x)],
                mode);
          } else {
            // Padding is the normalized form of a black (zero) pixel in every
            // mode, including kNormalized; it is not raw FP32 zero.
            expected[pixel_index] = NormalizedBlackPixel();
            if (!saw_padding) {
              saw_padding = true;
              first_padding_index = pixel_index;
            }
          }
        }
      }
    }
    EXPECT_TRUE(saw_padding);
    for (std::size_t index = 0; index < actual.size(); ++index) {
      EXPECT_TRUE(SameBits(expected[index], actual[index]));
    }
    if (saw_padding) {
      EXPECT_TRUE(
          SameBits(NormalizedBlackPixel(), actual[first_padding_index]));
      EXPECT_FALSE(SameBits(0.0F, actual[first_padding_index]));
    }

    // Index consistency: equal original indices give identical rows, while a
    // different index changes the translated row.
    EXPECT_EQ(0, std::memcmp(&actual[0 * kImagePixels],
                             &actual[1 * kImagePixels],
                             kImagePixels * sizeof(float)));
    EXPECT_EQ(0, std::memcmp(&actual[0 * kImagePixels],
                             &actual[4 * kImagePixels],
                             kImagePixels * sizeof(float)));
    EXPECT_TRUE(std::memcmp(&actual[0 * kImagePixels],
                            &actual[2 * kImagePixels],
                            kImagePixels * sizeof(float)) != 0);
  }

  // The same non-augmented preparation on a non-blocking stream must agree
  // bit for bit with the default-stream result.
  {
    const std::vector<float> packed =
        PackedFixture(kPixelCount, course::InputMode::kUnit);
    DeviceBuffer<float> device_packed = CopyToDevice(packed);
    DeviceBuffer<float> device_output(kPixelCount);
    NonblockingTestStream stream;
    LaunchPrepareInput(device_packed.get(), nullptr, device_output.get(),
                       kBatch, course::InputMode::kUnit, UINT64_C(0), 0, false,
                       stream.get());
    const std::vector<float> actual =
        CopyFromDevice(device_output.get(), kPixelCount, stream.get());
    for (std::size_t index = 0; index < actual.size(); ++index) {
      EXPECT_TRUE(SameBits(NormalizedUnit(packed[index]), actual[index]));
    }
  }
}

TEST_CASE(input_rejects_invalid_batch_epoch_pointers_and_overlap) {
  constexpr int kBatch = 2;
  const std::size_t kPixelCount =
      static_cast<std::size_t>(kBatch) * kImagePixels;
  const std::vector<float> packed(kPixelCount, 0.25F);
  const std::vector<std::uint32_t> indices(kBatch, 1);
  const DeviceBuffer<float> device_packed = CopyToDevice(packed);
  const DeviceBuffer<std::uint32_t> device_indices = CopyToDevice(indices);
  DeviceBuffer<float> device_output(kPixelCount);
  DeviceBuffer<float> device_shared(kPixelCount + 1);

  EXPECT_THROWS_AS(
      LaunchPrepareInput(device_packed.get(), nullptr, device_output.get(), 0,
                         course::InputMode::kUnit, UINT64_C(0), 0, false,
                         nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      LaunchPrepareInput(device_packed.get(), nullptr, device_output.get(), -1,
                         course::InputMode::kUnit, UINT64_C(0), 0, false,
                         nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      LaunchPrepareInput(device_packed.get(), device_indices.get(),
                         device_output.get(), kBatch,
                         course::InputMode::kUnit, UINT64_C(0), 0, true,
                         nullptr),
      std::invalid_argument);
  // Augmentation requires the original-index buffer; a missing index pointer
  // is rejected even when the epoch is valid.
  EXPECT_THROWS_AS(
      LaunchPrepareInput(device_packed.get(), nullptr, device_output.get(),
                         kBatch, course::InputMode::kUnit, UINT64_C(0), 1, true,
                         nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      LaunchPrepareInput(nullptr, nullptr, device_output.get(), kBatch,
                         course::InputMode::kUnit, UINT64_C(0), 0, false,
                         nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      LaunchPrepareInput(device_packed.get(), nullptr, nullptr, kBatch,
                         course::InputMode::kUnit, UINT64_C(0), 0, false,
                         nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      LaunchPrepareInput(device_packed.get(), nullptr, device_output.get(),
                         kBatch, static_cast<course::InputMode>(42),
                         UINT64_C(0), 0, false, nullptr),
      std::invalid_argument);

  // Identical extents and a one-element partial overlap are both refused.
  EXPECT_THROWS_AS(
      LaunchPrepareInput(device_shared.get(), nullptr, device_shared.get(),
                         kBatch, course::InputMode::kUnit, UINT64_C(0), 0,
                         false, nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      LaunchPrepareInput(device_shared.get(), nullptr,
                         device_shared.get() + 1, kBatch,
                         course::InputMode::kUnit, UINT64_C(0), 0, false,
                         nullptr),
      std::invalid_argument);

  // A non-augmented request ignores indices and epoch, including epoch zero.
  LaunchPrepareInput(device_packed.get(), nullptr, device_output.get(), kBatch,
                     course::InputMode::kUnit, UINT64_C(0), 0, false, nullptr);
  const std::vector<float> actual =
      CopyFromDevice(device_output.get(), kPixelCount, nullptr);
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_TRUE(SameBits(NormalizedUnit(0.25F), actual[index]));
  }
}

// ---------------------------------------------------------------------------
// Label-free argmax with smallest-index ties and a sticky bad-logit flag
// ---------------------------------------------------------------------------

TEST_CASE(label_free_argmax_ties_and_sticky_nonfinite) {
  constexpr int kBatch = 4;
  constexpr int kClasses = 10;
  std::vector<float> logits(static_cast<std::size_t>(kBatch) * kClasses);
  const auto row_base = [kClasses](int sample) {
    return static_cast<std::size_t>(sample) * kClasses;
  };

  // Row 0: tie between classes 3 and 7; row 1: every class tied; row 2: the
  // finite extreme FLT_MAX wins; row 3: a unique interior maximum.
  for (int class_index = 0; class_index < kClasses; ++class_index) {
    logits[row_base(0) + class_index] =
        (class_index == 3 || class_index == 7) ? 5.0F
                                               : -static_cast<float>(class_index);
    logits[row_base(1) + class_index] = -2.0F;
    logits[row_base(2) + class_index] = static_cast<float>(class_index);
    logits[row_base(3) + class_index] = 10.0F - static_cast<float>(class_index);
  }
  logits[row_base(2) + 2] = 9.0F;
  logits[row_base(2) + 9] = std::numeric_limits<float>::max();
  logits[row_base(3) + 5] = 12.0F;
  const std::vector<std::int32_t> expected{3, 0, 9, 5};

  DeviceBuffer<float> device_logits = CopyToDevice(logits);
  DeviceBuffer<std::int32_t> device_predictions(kBatch);
  DeviceBuffer<int> device_bad(1);
  ZeroBadFlag(&device_bad, nullptr);

  // Two launches inside one request: the flag stays clear because the caller
  // zeroed it once and every logit is finite.
  for (int repetition = 0; repetition < 2; ++repetition) {
    LaunchArgmax(device_logits.get(), device_predictions.get(),
                 device_bad.get(), kBatch, kClasses, nullptr);
    const std::vector<std::int32_t> predictions =
        CopyFromDevice(device_predictions.get(), kBatch, nullptr);
    const std::vector<int> flags = CopyFromDevice(device_bad.get(), 1, nullptr);
    EXPECT_EQ(expected, predictions);
    EXPECT_EQ(0, flags[0]);
  }

  // A NaN in the very last element of the last row must still be detected:
  // every logit of every row is examined.
  logits[row_base(kBatch - 1) + (kClasses - 1)] =
      std::numeric_limits<float>::quiet_NaN();
  UploadLogits(logits, &device_logits);
  LaunchArgmax(device_logits.get(), device_predictions.get(),
               device_bad.get(), kBatch, kClasses, nullptr);
  {
    const std::vector<int> flags = CopyFromDevice(device_bad.get(), 1, nullptr);
    const std::vector<std::int32_t> predictions =
        CopyFromDevice(device_predictions.get(), kBatch, nullptr);
    EXPECT_EQ(1, flags[0]);
    // Finite rows before the poisoned row are still classified exactly.
    EXPECT_EQ(expected[0], predictions[0]);
    EXPECT_EQ(expected[1], predictions[1]);
    EXPECT_EQ(expected[2], predictions[2]);
  }

  // The flag is sticky: a later all-finite request in the same sequence must
  // not clear the observed failure.
  logits[row_base(kBatch - 1) + (kClasses - 1)] = -1.0F;
  UploadLogits(logits, &device_logits);
  LaunchArgmax(device_logits.get(), device_predictions.get(), device_bad.get(),
               kBatch, kClasses, nullptr);
  {
    const std::vector<int> flags = CopyFromDevice(device_bad.get(), 1, nullptr);
    EXPECT_EQ(1, flags[0]);
  }

  // Each non-finite class is detected by a fresh request whose flag was zeroed
  // exactly once by its caller.
  const float nonfinite[] = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()};
  for (float value : nonfinite) {
    logits[row_base(kBatch - 1) + (kClasses - 1)] = value;
    UploadLogits(logits, &device_logits);
    ZeroBadFlag(&device_bad, nullptr);
    LaunchArgmax(device_logits.get(), device_predictions.get(),
                 device_bad.get(), kBatch, kClasses, nullptr);
    const std::vector<int> flags = CopyFromDevice(device_bad.get(), 1, nullptr);
    EXPECT_EQ(1, flags[0]);
  }

  // A fresh all-finite request starts clear.
  logits[row_base(kBatch - 1) + (kClasses - 1)] = -1.0F;
  UploadLogits(logits, &device_logits);
  ZeroBadFlag(&device_bad, nullptr);
  LaunchArgmax(device_logits.get(), device_predictions.get(), device_bad.get(),
               kBatch, kClasses, nullptr);
  {
    const std::vector<int> flags = CopyFromDevice(device_bad.get(), 1, nullptr);
    const std::vector<std::int32_t> predictions =
        CopyFromDevice(device_predictions.get(), kBatch, nullptr);
    EXPECT_EQ(0, flags[0]);
    EXPECT_EQ(expected, predictions);
  }
}

TEST_CASE(label_free_argmax_multiblock_sticky_nonfinite) {
  // Four 256-thread blocks: 128 poisoned rows are spread across every block so
  // many row threads concurrently update the shared sticky flag. This is
  // behavioural coverage of multi-block accumulation and sticky retention; it
  // deliberately does not (and cannot) prove race freedom, because current
  // hardware also observes the expected value from the pre-atomic plain stores.
  constexpr int kBatch = 1024;
  constexpr int kClasses = 10;
  std::vector<float> poisoned(static_cast<std::size_t>(kBatch) * kClasses);
  const auto row_base = [](int sample) {
    return static_cast<std::size_t>(sample) * kClasses;
  };
  for (int sample = 0; sample < kBatch; ++sample) {
    for (int class_index = 0; class_index < kClasses; ++class_index) {
      poisoned[row_base(sample) + class_index] = -1.0F;
    }
    poisoned[row_base(sample) + (sample % kClasses)] = 10.0F;
  }
  const float nonfinite[] = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()};
  int poisoned_rows = 0;
  for (int sample = 3; sample < kBatch; sample += 8) {
    poisoned[row_base(sample) + (sample % kClasses)] =
        nonfinite[(sample / 8) % 3];
    ++poisoned_rows;
  }
  EXPECT_EQ(128, poisoned_rows);

  DeviceBuffer<float> device_logits = CopyToDevice(poisoned);
  DeviceBuffer<std::int32_t> device_predictions(kBatch);
  DeviceBuffer<int> device_bad(1);
  ZeroBadFlag(&device_bad, nullptr);
  LaunchArgmax(device_logits.get(), device_predictions.get(), device_bad.get(),
               kBatch, kClasses, nullptr);
  {
    const std::vector<int> flags = CopyFromDevice(device_bad.get(), 1, nullptr);
    EXPECT_EQ(1, flags[0]);
  }

  // A later all-finite launch of the same request shape must not clear the
  // sticky flag, and every row must still be classified exactly.
  std::vector<float> finite(static_cast<std::size_t>(kBatch) * kClasses);
  for (int sample = 0; sample < kBatch; ++sample) {
    for (int class_index = 0; class_index < kClasses; ++class_index) {
      finite[row_base(sample) + class_index] = -1.0F;
    }
    finite[row_base(sample) + (sample % kClasses)] = 10.0F;
  }
  UploadLogits(finite, &device_logits);
  LaunchArgmax(device_logits.get(), device_predictions.get(), device_bad.get(),
               kBatch, kClasses, nullptr);
  {
    const std::vector<int> flags = CopyFromDevice(device_bad.get(), 1, nullptr);
    const std::vector<std::int32_t> predictions =
        CopyFromDevice(device_predictions.get(), kBatch, nullptr);
    EXPECT_EQ(1, flags[0]);
    for (int sample = 0; sample < kBatch; ++sample) {
      EXPECT_EQ(static_cast<std::int32_t>(sample % kClasses),
                predictions[sample]);
    }
  }
}

TEST_CASE(argmax_rejects_invalid_shapes_pointers_and_overlap) {
  constexpr int kBatch = 2;
  constexpr int kClasses = 10;
  const auto* const logits = reinterpret_cast<const float*>(32);
  auto* const predictions = reinterpret_cast<std::int32_t*>(64);
  auto* const bad = reinterpret_cast<int*>(96);

  EXPECT_THROWS_AS(
      LaunchArgmax(logits, predictions, bad, 0, kClasses, nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      LaunchArgmax(logits, predictions, bad, -1, kClasses, nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(LaunchArgmax(logits, predictions, bad, kBatch, 0, nullptr),
                   std::invalid_argument);
  EXPECT_THROWS_AS(LaunchArgmax(logits, predictions, bad, kBatch, 257, nullptr),
                   std::invalid_argument);
  EXPECT_THROWS_AS(
      LaunchArgmax(nullptr, predictions, bad, kBatch, kClasses, nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(LaunchArgmax(logits, nullptr, bad, kBatch, kClasses, nullptr),
                   std::invalid_argument);
  EXPECT_THROWS_AS(
      LaunchArgmax(logits, predictions, nullptr, kBatch, kClasses, nullptr),
      std::invalid_argument);

  DeviceBuffer<float> shared(static_cast<std::size_t>(kBatch) * kClasses);
  DeviceBuffer<std::int32_t> prediction_storage(kBatch);
  DeviceBuffer<int> bad_storage(1);

  // Predictions may not fall inside the logits extent, nor partially overlap
  // it, and the flag may not alias either other buffer.
  EXPECT_THROWS_AS(
      LaunchArgmax(shared.get(),
                   reinterpret_cast<std::int32_t*>(shared.get()),
                   bad_storage.get(), kBatch, kClasses, nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      LaunchArgmax(shared.get(),
                   reinterpret_cast<std::int32_t*>(shared.get() + 1),
                   bad_storage.get(), kBatch, kClasses, nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      LaunchArgmax(shared.get(), prediction_storage.get(),
                   reinterpret_cast<int*>(shared.get()), kBatch, kClasses,
                   nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      LaunchArgmax(shared.get(), prediction_storage.get(),
                   reinterpret_cast<int*>(prediction_storage.get()), kBatch,
                   kClasses, nullptr),
      std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Forward-only inference model against the original reference LeNet
// ---------------------------------------------------------------------------

TEST_CASE(inference_logits_bitwise_match_reference) {
  const ParameterSet parameters = PatternParameters();
  NonblockingTestStream stream;

  // Forward-only storage excludes gradients, moments, and finite scans, so it
  // must be strictly smaller than the original trainable arena at every
  // matching capacity.
  const std::array<int, 3> capacities{{1, 17, 1024}};
  for (std::size_t index = 0; index < capacities.size(); ++index) {
    course::InferenceModel model(capacities[index], parameters, stream.get());
    LeNet reference(capacities[index], UINT64_C(4242) + index, stream.get());
    reference.ImportParameters(parameters);
    EXPECT_TRUE(model.RequiredDeviceBytes() > 0);
    EXPECT_TRUE(model.RequiredDeviceBytes() < reference.RequiredDeviceBytes());
  }

  constexpr int kCapacity = 1024;
  const std::array<int, 3> batches{{1, 17, 1024}};
  const std::vector<float> input = NormalizedInput(kCapacity);
  DeviceBuffer<float> device_input = CopyToDevice(input);
  DeviceBuffer<std::int32_t> device_predictions(kCapacity);
  DeviceBuffer<int> device_bad(1);
  ZeroBadFlag(&device_bad, stream.get());

  course::InferenceModel model(kCapacity, parameters, stream.get());
  LeNet reference(kCapacity, UINT64_C(9001), stream.get());
  reference.ImportParameters(parameters);

  const std::size_t forward_events =
      DeviceBufferSuccessfulAllocationEventCountForTests();
  for (int batch : batches) {
    const float* const model_logits = model.Forward(device_input.get(), batch);
    const float* const reference_logits =
        reference.Forward(device_input.get(), batch);
    const std::size_t logit_count =
        static_cast<std::size_t>(batch) * kClassCount;
    const std::vector<float> actual =
        CopyFromDevice(model_logits, logit_count, stream.get());
    const std::vector<float> expected =
        CopyFromDevice(reference_logits, logit_count, stream.get());

    // Identical already-normalized FP32 input and identical parameters must
    // produce bit-identical logits; no tolerance is permitted.
    EXPECT_TRUE(std::memcmp(expected.data(), actual.data(),
                            logit_count * sizeof(float)) == 0);

    LaunchArgmax(model_logits, device_predictions.get(), device_bad.get(),
                 batch, kClassCount, stream.get());
    const std::vector<std::int32_t> predictions =
        CopyFromDevice(device_predictions.get(), batch, stream.get());
    for (int sample = 0; sample < batch; ++sample) {
      const std::vector<float> row = RowCopy(expected, sample);
      EXPECT_EQ(HostArgmax(row.data(), kClassCount), predictions[sample]);
    }
  }
  EXPECT_EQ(forward_events,
            DeviceBufferSuccessfulAllocationEventCountForTests());
  const std::vector<int> flags = CopyFromDevice(device_bad.get(), 1, stream.get());
  EXPECT_EQ(0, flags[0]);
}

TEST_CASE(inference_tail_repeated_no_allocation) {
  constexpr int kCapacity = 17;
  const ParameterSet parameters = PatternParameters();
  NonblockingTestStream stream;

  LeNet reference(kCapacity, UINT64_C(777), stream.get());
  reference.ImportParameters(parameters);
  const std::vector<float> input = NormalizedInput(kCapacity);
  DeviceBuffer<float> device_input = CopyToDevice(input);

  const std::size_t live_baseline = DeviceBufferAllocationCountForTests();
  const std::size_t events_baseline =
      DeviceBufferSuccessfulAllocationEventCountForTests();
  {
    course::InferenceModel model(kCapacity, parameters, stream.get());
    // One checked fixed arena only.
    EXPECT_EQ(live_baseline + 1, DeviceBufferAllocationCountForTests());
    EXPECT_EQ(events_baseline + 1,
              DeviceBufferSuccessfulAllocationEventCountForTests());

    DeviceBuffer<std::int32_t> device_predictions(kCapacity);
    DeviceBuffer<int> device_bad(1);
    ZeroBadFlag(&device_bad, stream.get());
    const std::size_t forward_events =
        DeviceBufferSuccessfulAllocationEventCountForTests();
    const std::size_t forward_live = DeviceBufferAllocationCountForTests();

    // Repeated capacity, tail, and single-sample batches. Each result must
    // match the original reference model run with the same batch, and stale
    // tail rows from a previous larger batch must never leak into output.
    const std::array<int, 9> batches{{17, 1, 17, 3, 17, 1, 16, 17, 2}};
    const float* stable_logits = nullptr;
    for (int batch : batches) {
      const float* const model_logits = model.Forward(device_input.get(), batch);
      const float* const reference_logits =
          reference.Forward(device_input.get(), batch);
      if (stable_logits == nullptr) {
        stable_logits = model_logits;
      }
      // The arena is fixed: every Forward exposes the same logits address.
      EXPECT_EQ(stable_logits, model_logits);
      const std::size_t logit_count =
          static_cast<std::size_t>(batch) * kClassCount;
      const std::vector<float> actual =
          CopyFromDevice(model_logits, logit_count, stream.get());
      const std::vector<float> expected =
          CopyFromDevice(reference_logits, logit_count, stream.get());
      EXPECT_TRUE(std::memcmp(expected.data(), actual.data(),
                              logit_count * sizeof(float)) == 0);

      LaunchArgmax(model_logits, device_predictions.get(), device_bad.get(),
                   batch, kClassCount, stream.get());
      const std::vector<std::int32_t> predictions =
          CopyFromDevice(device_predictions.get(), batch, stream.get());
      for (int sample = 0; sample < batch; ++sample) {
        const std::vector<float> row = RowCopy(expected, sample);
        EXPECT_EQ(HostArgmax(row.data(), kClassCount), predictions[sample]);
      }

      // Forward and argmax allocate no device storage.
      EXPECT_EQ(forward_events,
                DeviceBufferSuccessfulAllocationEventCountForTests());
      EXPECT_EQ(forward_live, DeviceBufferAllocationCountForTests());
    }

    // Two identical consecutive calls return the same stable pointer and
    // bit-identical logits.
    const float* const first = model.Forward(device_input.get(), 3);
    const std::vector<float> first_logits =
        CopyFromDevice(first, 3 * kClassCount, stream.get());
    const float* const second = model.Forward(device_input.get(), 3);
    const std::vector<float> second_logits =
        CopyFromDevice(second, 3 * kClassCount, stream.get());
    EXPECT_EQ(first, second);
    EXPECT_EQ(first_logits, second_logits);

    const std::vector<int> flags =
        CopyFromDevice(device_bad.get(), 1, stream.get());
    EXPECT_EQ(0, flags[0]);
  }
  // Destruction releases exactly the single arena again.
  EXPECT_EQ(live_baseline, DeviceBufferAllocationCountForTests());
}

TEST_CASE(inference_rejects_invalid_parameters_capacity_and_pointers) {
  const ParameterSet parameters = PatternParameters();
  NonblockingTestStream stream;

  EXPECT_THROWS_AS(course::InferenceModel(0, parameters, stream.get()),
                   std::invalid_argument);
  EXPECT_THROWS_AS(course::InferenceModel(-3, parameters, stream.get()),
                   std::invalid_argument);

  ParameterSet missing_tensor = parameters;
  missing_tensor.pop_back();
  EXPECT_THROWS_AS(
      course::InferenceModel(4, missing_tensor, stream.get()),
      std::invalid_argument);

  ParameterSet extra_tensor = parameters;
  extra_tensor.push_back(extra_tensor.front());
  EXPECT_THROWS_AS(course::InferenceModel(4, extra_tensor, stream.get()),
                   std::invalid_argument);

  ParameterSet short_values = parameters;
  short_values[2].values.pop_back();
  EXPECT_THROWS_AS(course::InferenceModel(4, short_values, stream.get()),
                   std::invalid_argument);

  ParameterSet wrong_shape = parameters;
  wrong_shape[0].dimensions[0] = 7;
  EXPECT_THROWS_AS(course::InferenceModel(4, wrong_shape, stream.get()),
                   std::invalid_argument);

  const float nonfinite[] = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()};
  struct PoisonedValue {
    std::size_t tensor;
    std::size_t index;
  };
  const PoisonedValue positions[] = {{0, 0}, {1, 5}, {8, 42}};
  for (std::size_t poison = 0; poison < 3; ++poison) {
    ParameterSet poisoned = parameters;
    poisoned[positions[poison].tensor].values[positions[poison].index] =
        nonfinite[poison];
    EXPECT_THROWS_AS(course::InferenceModel(4, poisoned, stream.get()),
                     std::invalid_argument);
  }

  course::InferenceModel model(4, parameters, stream.get());
  const std::vector<float> input = NormalizedInput(4);
  DeviceBuffer<float> device_input = CopyToDevice(input);

  EXPECT_THROWS_AS(model.Forward(nullptr, 1), std::invalid_argument);
  EXPECT_THROWS_AS(model.Forward(device_input.get(), 0), std::invalid_argument);
  EXPECT_THROWS_AS(model.Forward(device_input.get(), 5), std::invalid_argument);

  // The full-capacity boundary remains valid after the rejected calls.
  const float* const logits = model.Forward(device_input.get(), 4);
  EXPECT_TRUE(logits != nullptr);
  const std::vector<float> values =
      CopyFromDevice(logits, 4 * kClassCount, stream.get());
  EXPECT_EQ(static_cast<std::size_t>(4 * kClassCount), values.size());
}

}  // namespace
