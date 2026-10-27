// Tests-first contract for course/training.h and course/training.cu.
//
// Every case below is written against the approved declarations before the
// training implementation exists, so the suite is expected to fail at link
// time until course/training.cu supplies RunCourseTrain and the small pure
// schedule/config/selection policies. Fixtures are tiny raw FP32/int32 files in
// this run's exclusively owned scratch tree; no mounted dataset, test split,
// pretrained weights, or CPU neural-network fallback is used. Real one-epoch
// training runs prove that parameters actually move away from the original
// seeded initialization and that the persisted bundle is usable by both the
// reference LeNet and the course forward-only model.

#include "course/gpu_ops.h"
#include "course/inference.h"
#include "course/model_io.h"
#include "course/tests/test_support.h"
#include "course/training.h"

#include "cuda_check.h"
#include "lenet.h"
#include "parameters.h"
#include "random.h"
#include "tensor.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using course_test_support::Crc32;
using course_test_support::Hex8;
using course_test_support::MakeDirectory;
using course_test_support::MakeScratchDirectory;
using course_test_support::PathExists;
using course_test_support::ReadBytes;
using course_test_support::ReadText;
using course_test_support::SameBits;
using course_test_support::WriteFloats;
using course_test_support::WriteInt32s;
using course_test_support::WriteText;

// Supported course geometry.
constexpr int kImageSide = 28;
constexpr int kImagePixels = kImageSide * kImageSide;
constexpr int kClassCount = 10;
constexpr std::uint32_t kChannelCount = 1;

// Fixed normalization constants from the model contract.
constexpr float kNormalizationMean = 0.1307F;
constexpr float kNormalizationStddev = 0.3081F;

// Default course seed; every fixture pins it explicitly.
const std::uint64_t kSeed = UINT64_C(1337);

// Small training batch keeps bounded fixtures multi-batch without implying a
// different default; the CLI still supplies the pinned 128.
constexpr std::uint32_t kFixtureBatch = 4;

// RAII non-blocking stream so every borrowed-stream boundary is exercised
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

// Canonical course shape for a supplied sample count.
course::Shape FixtureShape(std::uint32_t sample_count) {
  course::Shape shape;
  shape.n = sample_count;
  shape.c = kChannelCount;
  shape.h = static_cast<std::uint32_t>(kImageSide);
  shape.w = static_cast<std::uint32_t>(kImageSide);
  shape.k = static_cast<std::uint32_t>(kClassCount);
  return shape;
}

// Owns the two raw fixture paths written for one bounded training run.
struct FixturePaths {
  std::string images;
  std::string labels;
};

// Writes host-order FP32 images and int32 labels below an owned directory.
FixturePaths WriteFixture(const std::string& directory,
                          const std::vector<float>& images,
                          const std::vector<std::int32_t>& labels) {
  MakeDirectory(directory);
  FixturePaths paths;
  paths.images = directory + "/images.f32";
  paths.labels = directory + "/labels.i32";
  WriteFloats(paths.images, images);
  WriteInt32s(paths.labels, labels);
  return paths;
}

// Builds a fully explicit direct TrainOptions value. Tests never parse the CLI
// and never depend on hidden environment variables.
course::TrainOptions MakeOptions(const FixturePaths& paths,
                                 const std::string& work_dir,
                                 course::InputMode input_mode,
                                 std::uint32_t sample_count,
                                 std::uint32_t epochs = 1,
                                 std::uint32_t batch_size = kFixtureBatch,
                                 std::uint64_t seed = kSeed) {
  course::TrainOptions options;
  options.images_path = paths.images;
  options.labels_path = paths.labels;
  options.work_dir = work_dir;
  options.shape = FixtureShape(sample_count);
  options.input_mode = input_mode;
  options.epochs = epochs;
  options.batch_size = batch_size;
  options.seed = seed;
  return options;
}

// Deterministic in-range kUnit fixture with varied fractional pixels.
std::vector<float> PatternUnitImages(std::uint32_t sample_count) {
  const std::size_t pixel_count =
      static_cast<std::size_t>(sample_count) * kImagePixels;
  std::vector<float> values(pixel_count);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] =
        static_cast<float>((index * 29 + 7) % 1000) / 1000.0F;
  }
  return values;
}

// A constant kUnit image makes every prediction within one evaluation uniform,
// which keeps the split oracle unambiguous without depending on model skill.
std::vector<float> ConstantUnitImages(std::uint32_t sample_count,
                                      float value) {
  return std::vector<float>(
      static_cast<std::size_t>(sample_count) * kImagePixels, value);
}

// A constant kByte image for the fractional-pixel regression.
std::vector<float> ConstantByteImages(std::uint32_t sample_count,
                                      float value) {
  return std::vector<float>(
      static_cast<std::size_t>(sample_count) * kImagePixels, value);
}

// Cyclic valid labels keep every class reachable in the small fixture.
std::vector<std::int32_t> CyclicLabels(std::uint32_t sample_count) {
  std::vector<std::int32_t> labels(sample_count);
  for (std::uint32_t index = 0; index < sample_count; ++index) {
    labels[index] = static_cast<std::int32_t>(index % kClassCount);
  }
  return labels;
}

// Recreates the original deterministic seeded parameter initialization
// without allocating device state.
ParameterSet SeededInitialParameters(std::uint64_t seed) {
  ParameterSet parameters = CreateLenetParameters();
  InitializeLenetParameters(seed, &parameters);
  return parameters;
}

// Counts parameter values whose exact FP32 representation differs.
std::size_t CountParameterBitDifferences(const ParameterSet& left,
                                         const ParameterSet& right) {
  if (left.size() != right.size()) {
    return std::numeric_limits<std::size_t>::max();
  }
  std::size_t differences = 0;
  for (std::size_t tensor = 0; tensor < left.size(); ++tensor) {
    if (left[tensor].values.size() != right[tensor].values.size()) {
      return std::numeric_limits<std::size_t>::max();
    }
    for (std::size_t index = 0; index < left[tensor].values.size(); ++index) {
      if (!SameBits(left[tensor].values[index],
                    right[tensor].values[index])) {
        ++differences;
      }
    }
  }
  return differences;
}

// Returns ascending indices [0, sample_count).
std::vector<std::uint32_t> AllIndices(std::uint32_t sample_count) {
  std::vector<std::uint32_t> indices(sample_count);
  for (std::uint32_t index = 0; index < sample_count; ++index) {
    indices[index] = index;
  }
  return indices;
}

// Copies selected raw FP32 images into a packed device batch, prepares them on
// the GPU in the persisted mode without augmentation, and synchronizes before
// the temporary packed upload is released.
DeviceBuffer<float> PrepareNormalized(
    const std::vector<float>& raw_images,
    const std::vector<std::uint32_t>& indices, course::InputMode input_mode,
    std::uint64_t seed, cudaStream_t stream) {
  const std::size_t sample_count = indices.size();
  std::vector<float> packed(sample_count *
                            static_cast<std::size_t>(kImagePixels));
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const std::size_t source =
        static_cast<std::size_t>(indices[sample]) * kImagePixels;
    std::memcpy(packed.data() + sample * kImagePixels,
                raw_images.data() + source,
                static_cast<std::size_t>(kImagePixels) * sizeof(float));
  }

  DeviceBuffer<float> device_packed(packed.size());
  CUDA_CHECK(cudaMemcpyAsync(device_packed.get(), packed.data(),
                             packed.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  DeviceBuffer<float> normalized(packed.size());
  course::LaunchPrepareInput(device_packed.get(), nullptr, normalized.get(),
                             static_cast<int>(sample_count), input_mode, seed,
                             0, false, stream);
  // Tests may synchronize; the production workflow keeps its own boundaries.
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return normalized;
}

// Runs the reference LeNet forward over already-normalized device input.
std::vector<float> ReferenceLogits(const ParameterSet& parameters,
                                   const DeviceBuffer<float>& normalized,
                                   int batch) {
  NonblockingTestStream stream;
  LeNet reference(batch, kSeed, stream.get());
  reference.ImportParameters(parameters);
  const float* const logits = reference.Forward(normalized.get(), batch);
  return CopyFromDevice(logits,
                        static_cast<std::size_t>(batch) * kClassCount,
                        stream.get());
}

// Runs the course forward-only model over the same normalized input.
std::vector<float> CourseLogits(const ParameterSet& parameters,
                                const DeviceBuffer<float>& normalized,
                                int batch) {
  NonblockingTestStream stream;
  course::InferenceModel model(batch, parameters, stream.get());
  const float* const logits = model.Forward(normalized.get(), batch);
  return CopyFromDevice(logits,
                        static_cast<std::size_t>(batch) * kClassCount,
                        stream.get());
}

// Smallest-index argmax used only to turn reference logits into class integers.
std::int32_t HostArgmax(const float* row) {
  std::int32_t best = 0;
  for (int class_index = 1; class_index < kClassCount; ++class_index) {
    if (row[class_index] > row[best]) {
      best = class_index;
    }
  }
  return best;
}

// Evaluates one exact index subset and returns correct / subset size. The
// model is forward-only; no labels beyond the supplied training fixture are
// available anywhere in this suite.
float AccuracyOnIndices(const ParameterSet& parameters,
                        const std::vector<float>& raw_images,
                        const std::vector<std::uint8_t>& labels,
                        const std::vector<std::uint32_t>& indices,
                        course::InputMode input_mode) {
  NonblockingTestStream stream;
  const DeviceBuffer<float> normalized =
      PrepareNormalized(raw_images, indices, input_mode, kSeed, stream.get());
  const int batch = static_cast<int>(indices.size());
  course::InferenceModel model(batch, parameters, stream.get());
  const float* const logits = model.Forward(normalized.get(), batch);
  const std::vector<float> host_logits =
      CopyFromDevice(logits, static_cast<std::size_t>(batch) * kClassCount,
                     stream.get());
  std::uint32_t correct = 0;
  for (int sample = 0; sample < batch; ++sample) {
    const std::int32_t predicted =
        HostArgmax(host_logits.data() +
                   static_cast<std::size_t>(sample) * kClassCount);
    if (labels[indices[static_cast<std::size_t>(sample)]] == predicted) {
      ++correct;
    }
  }
  return static_cast<float>(correct) / static_cast<float>(indices.size());
}

// Confirms that every logit is finite.
void ExpectFiniteLogits(const std::vector<float>& logits) {
  for (const float value : logits) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

// Returns the persisted metadata spelling for one input mode.
const char* InputModeName(course::InputMode input_mode) {
  switch (input_mode) {
    case course::InputMode::kUnit:
      return "unit";
    case course::InputMode::kByte:
      return "byte";
    case course::InputMode::kNormalized:
      return "normalized";
  }
  return "";
}

// Extracts the whitespace-delimited token following key in text.
bool ExtractToken(const std::string& text, const std::string& key,
                  std::string* value) {
  const std::string::size_type position = text.find(key);
  if (position == std::string::npos) {
    return false;
  }
  const std::string::size_type begin = position + key.size();
  std::string::size_type end = text.find_first_of(" \r\n", begin);
  if (end == std::string::npos) {
    end = text.size();
  }
  *value = text.substr(begin, end - begin);
  return !value->empty();
}

// Requires a completion-record key/value pair.
void ExpectToken(const std::string& text, const std::string& key,
                 const std::string& expected) {
  std::string actual;
  EXPECT_TRUE(ExtractToken(text, key, &actual));
  EXPECT_EQ(expected, actual);
}

// Parses the validation_accuracy token as an FP32 classic-locale number.
bool ParseFloatToken(const std::string& text, const std::string& key,
                     float* value) {
  std::string token;
  if (!ExtractToken(text, key, &token)) {
    return false;
  }
  char* end = nullptr;
  const float parsed = std::strtof(token.c_str(), &end);
  if (end == token.c_str() || end == nullptr || *end != '\0') {
    return false;
  }
  *value = parsed;
  return true;
}

// Parses one unsigned decimal token, rejecting signs, padding, and overflow.
bool ParseU32Token(const std::string& text, const std::string& key,
                   std::uint32_t* value) {
  std::string token;
  if (!ExtractToken(text, key, &token) || token.empty()) {
    return false;
  }
  std::uint64_t parsed = 0;
  for (const char digit : token) {
    if (digit < '0' || digit > '9') {
      return false;
    }
    parsed = parsed * 10u + static_cast<std::uint64_t>(digit - '0');
    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
  }
  *value = static_cast<std::uint32_t>(parsed);
  return true;
}

// Splits captured diagnostics into individual lines. Epoch fields are always
// parsed from their own line, never from a global first-token search.
std::vector<std::string> DiagnosticLines(const std::string& text) {
  std::vector<std::string> lines;
  std::string::size_type start = 0;
  while (start < text.size()) {
    const std::string::size_type end = text.find('\n', start);
    if (end == std::string::npos) {
      lines.push_back(text.substr(start));
      break;
    }
    lines.push_back(text.substr(start, end - start));
    start = end + 1;
  }
  return lines;
}

// True only when line begins with the exact event prefix.
bool LineStartsWith(const std::string& line, const std::string& prefix) {
  return line.size() >= prefix.size() &&
         line.compare(0, prefix.size(), prefix) == 0;
}

// Verifies the required completion record against the saved bundle.
void ExpectCompletionRecord(const std::string& diagnostics,
                            std::uint32_t epochs,
                            std::uint32_t best_epoch,
                            float validation_accuracy,
                            course::InputMode input_mode,
                            const std::string& checkpoint_crc32) {
  EXPECT_TRUE(diagnostics.find("event=course_train status=pass") !=
              std::string::npos);
  ExpectToken(diagnostics, "epochs=", std::to_string(epochs));
  ExpectToken(diagnostics, "best_epoch=", std::to_string(best_epoch));
  ExpectToken(diagnostics, "input_mode=", InputModeName(input_mode));
  ExpectToken(diagnostics, "checkpoint_crc32=", checkpoint_crc32);

  float parsed = 0.0F;
  EXPECT_TRUE(ParseFloatToken(diagnostics, "validation_accuracy=", &parsed));
  EXPECT_TRUE(SameBits(validation_accuracy, parsed));
}

// Loads the saved bundle and checks invariants shared by the happy-path cases.
course::CourseModel LoadValidatedBundle(
    const std::string& work_dir, course::InputMode expected_mode) {
  const course::CourseModel bundle = course::LoadCourseModel(work_dir);
  EXPECT_EQ(expected_mode, bundle.input_mode);
  EXPECT_EQ(static_cast<std::size_t>(10), bundle.checkpoint.parameters.size());
  EXPECT_TRUE(std::isfinite(
      bundle.checkpoint.metadata.validation_accuracy));
  EXPECT_TRUE(bundle.checkpoint.metadata.validation_accuracy >= 0.0F);
  EXPECT_TRUE(bundle.checkpoint.metadata.validation_accuracy <= 1.0F);
  EXPECT_TRUE(SameBits(kNormalizationMean,
                       bundle.checkpoint.metadata.normalization_mean));
  EXPECT_TRUE(SameBits(kNormalizationStddev,
                       bundle.checkpoint.metadata.normalization_stddev));
  return bundle;
}

// ---------------------------------------------------------------------------
// Real bounded training workflow
// ---------------------------------------------------------------------------

TEST_CASE(train_saves_readable_model_without_test_split_files) {
  const std::string root = MakeScratchDirectory("training_readable_bundle");
  const std::uint32_t sample_count = 10;
  const std::vector<float> images = PatternUnitImages(sample_count);
  const std::vector<std::int32_t> labels = CyclicLabels(sample_count);
  const FixturePaths fixture = WriteFixture(root + "/data", images, labels);

  // The selected work directory does not exist yet; an unrelated sibling must
  // survive the workflow's directory creation.
  const std::string work_dir = root + "/work";
  const std::string unrelated = root + "/unrelated.txt";
  WriteText(unrelated, "preserve me");
  EXPECT_FALSE(PathExists(work_dir));

  course::TrainOptions options =
      MakeOptions(fixture, work_dir, course::InputMode::kUnit, sample_count);
  std::ostringstream diagnostics;
  EXPECT_EQ(0, course::RunCourseTrain(options, diagnostics));

  EXPECT_TRUE(PathExists(work_dir));
  EXPECT_TRUE(PathExists(work_dir + "/lenet.bin"));
  EXPECT_TRUE(PathExists(work_dir + "/course.meta"));
  EXPECT_TRUE(PathExists(unrelated));
  EXPECT_EQ(std::string("preserve me"), ReadText(unrelated));

  const course::CourseModel bundle =
      LoadValidatedBundle(work_dir, course::InputMode::kUnit);
  EXPECT_EQ(1u, bundle.checkpoint.metadata.best_epoch);

  // The completion record must carry the validated saved CRC, not a
  // placeholder, and training must never claim a test-set accuracy.
  const std::string checkpoint_crc32 =
      Hex8(Crc32(ReadBytes(work_dir + "/lenet.bin")));
  const std::string diagnostic_text = diagnostics.str();
  ExpectCompletionRecord(diagnostic_text, options.epochs,
                         bundle.checkpoint.metadata.best_epoch,
                         bundle.checkpoint.metadata.validation_accuracy,
                         course::InputMode::kUnit, checkpoint_crc32);
  EXPECT_TRUE(diagnostic_text.find("test_accuracy") == std::string::npos);
  EXPECT_TRUE(diagnostic_text.find("test accuracy") == std::string::npos);

  // One real optimizer epoch must move weights away from the original seeded
  // initialization.
  const ParameterSet initial = SeededInitialParameters(options.seed);
  EXPECT_TRUE(CountParameterBitDifferences(initial, bundle.checkpoint.parameters) >
              0);

  // Read the raw files back exactly as the workflow did. No value passed
  // through uint8.
  const course::Shape shape = FixtureShape(sample_count);
  const std::vector<float> raw_images =
      course::ReadImages(fixture.images, shape, course::InputMode::kUnit);
  const std::vector<std::uint8_t> raw_labels =
      course::ReadLabels(fixture.labels, shape);
  EXPECT_TRUE(std::memcmp(images.data(), raw_images.data(),
                          images.size() * sizeof(float)) == 0);

  const std::vector<std::uint32_t> all_indices = AllIndices(sample_count);
  const DeviceBuffer<float> normalized =
      PrepareNormalized(raw_images, all_indices, course::InputMode::kUnit,
                        options.seed, nullptr);
  const int batch = static_cast<int>(sample_count);
  const std::vector<float> course_logits =
      CourseLogits(bundle.checkpoint.parameters, normalized, batch);
  const std::vector<float> reference_logits =
      ReferenceLogits(bundle.checkpoint.parameters, normalized, batch);
  EXPECT_EQ(static_cast<std::size_t>(batch) * kClassCount,
            course_logits.size());
  EXPECT_TRUE(std::memcmp(reference_logits.data(), course_logits.data(),
                          course_logits.size() * sizeof(float)) == 0);
  ExpectFiniteLogits(course_logits);

  // Every supplied label remains in the supported class range after the
  // persisted bundle round-trip.
  for (const std::uint8_t label : raw_labels) {
    EXPECT_TRUE(label < kClassCount);
  }
}

TEST_CASE(two_epoch_training_reselects_and_publishes_consistent_bundle) {
  const std::string root = MakeScratchDirectory("training_two_epochs");
  const std::uint32_t sample_count = 10;
  const FixturePaths fixture =
      WriteFixture(root + "/data", PatternUnitImages(sample_count),
                   CyclicLabels(sample_count));
  const std::string work_dir = root + "/work";
  const course::TrainOptions options =
      MakeOptions(fixture, work_dir, course::InputMode::kUnit, sample_count, 2,
                  kFixtureBatch);

  std::ostringstream diagnostics;
  EXPECT_EQ(0, course::RunCourseTrain(options, diagnostics));

  const std::string text = diagnostics.str();
  const std::vector<std::string> lines = DiagnosticLines(text);
  std::vector<std::uint32_t> epoch_numbers;
  std::vector<float> epoch_accuracies;
  std::size_t completion_count = 0;
  std::string completion_line;
  std::size_t unrelated_completion_count = 0;

  for (const std::string& line : lines) {
    if (LineStartsWith(line, "event=course_train_epoch ")) {
      std::uint32_t epoch = 0;
      float accuracy = 0.0F;
      EXPECT_TRUE(ParseU32Token(line, "epoch=", &epoch));
      EXPECT_TRUE(ParseFloatToken(line, "val_accuracy=", &accuracy));
      EXPECT_TRUE(std::isfinite(accuracy));
      EXPECT_TRUE(accuracy >= 0.0F);
      EXPECT_TRUE(accuracy <= 1.0F);
      epoch_numbers.push_back(epoch);
      epoch_accuracies.push_back(accuracy);
    } else if (LineStartsWith(line, "event=course_train status=pass ")) {
      ++completion_count;
      completion_line = line;
    } else if (line.find("event=course_train") != std::string::npos) {
      ++unrelated_completion_count;
    }
  }

  // Exactly two ordered epoch diagnostics, then exactly one completion record.
  EXPECT_EQ(static_cast<std::size_t>(2), epoch_numbers.size());
  EXPECT_EQ(static_cast<std::size_t>(2), epoch_accuracies.size());
  EXPECT_EQ(static_cast<std::size_t>(1), completion_count);
  EXPECT_EQ(static_cast<std::size_t>(0), unrelated_completion_count);
  if (epoch_numbers.size() == 2) {
    EXPECT_EQ(1u, epoch_numbers[0]);
    EXPECT_EQ(2u, epoch_numbers[1]);
  }

  // Independently derive strict-improvement selection from the two epoch
  // diagnostics. Equal or lower second-epoch accuracy naturally keeps epoch 1.
  std::uint32_t expected_best_epoch = 0;
  float expected_best_accuracy = -1.0F;
  for (std::size_t index = 0;
       index < epoch_accuracies.size() && index < epoch_numbers.size();
       ++index) {
    if (epoch_accuracies[index] > expected_best_accuracy) {
      expected_best_accuracy = epoch_accuracies[index];
      expected_best_epoch = epoch_numbers[index];
    }
  }
  EXPECT_TRUE(expected_best_epoch == 1u || expected_best_epoch == 2u);

  const course::CourseModel bundle =
      LoadValidatedBundle(work_dir, course::InputMode::kUnit);
  EXPECT_EQ(expected_best_epoch, bundle.checkpoint.metadata.best_epoch);
  EXPECT_TRUE(SameBits(expected_best_accuracy,
                       bundle.checkpoint.metadata.validation_accuracy));

  // The completion record is parsed from its own line, and its CRC must match
  // the validated saved checkpoint bytes.
  ExpectToken(completion_line, "epochs=", "2");
  ExpectToken(completion_line, "best_epoch=",
              std::to_string(expected_best_epoch));
  ExpectToken(completion_line, "input_mode=", "unit");
  const std::string saved_crc =
      Hex8(Crc32(ReadBytes(work_dir + "/lenet.bin")));
  ExpectToken(completion_line, "checkpoint_crc32=", saved_crc);
  float completion_accuracy = 0.0F;
  EXPECT_TRUE(
      ParseFloatToken(completion_line, "validation_accuracy=",
                      &completion_accuracy));
  EXPECT_TRUE(SameBits(expected_best_accuracy, completion_accuracy));
  EXPECT_TRUE(text.find("test_accuracy") == std::string::npos);
  EXPECT_TRUE(text.find("test accuracy") == std::string::npos);
}

TEST_CASE(training_replaces_valid_bundle_and_preserves_unrelated_entries) {
  const std::string root = MakeScratchDirectory("training_replace_bundle");
  const std::uint32_t sample_count = 8;
  const FixturePaths fixture =
      WriteFixture(root + "/data", PatternUnitImages(sample_count),
                   CyclicLabels(sample_count));

  // The work directory already exists and contains unrelated entries that the
  // workflow must never clear.
  const std::string work_dir = root + "/work";
  MakeDirectory(work_dir);
  MakeDirectory(work_dir + "/history");
  WriteText(work_dir + "/notes.txt", "unrelated work entry");
  WriteText(work_dir + "/history/old.log", "old history");

  const course::TrainOptions options =
      MakeOptions(fixture, work_dir, course::InputMode::kUnit, sample_count);
  std::ostringstream first_diagnostics;
  EXPECT_EQ(0, course::RunCourseTrain(options, first_diagnostics));
  EXPECT_EQ(std::string("unrelated work entry"),
            ReadText(work_dir + "/notes.txt"));
  EXPECT_EQ(std::string("old history"), ReadText(work_dir + "/history/old.log"));
  static_cast<void>(LoadValidatedBundle(work_dir, course::InputMode::kUnit));

  // A complete, valid reserved pair may be replaced by a later workflow run
  // without touching the surrounding directory.
  std::ostringstream second_diagnostics;
  EXPECT_EQ(0, course::RunCourseTrain(options, second_diagnostics));
  EXPECT_EQ(std::string("unrelated work entry"),
            ReadText(work_dir + "/notes.txt"));
  EXPECT_EQ(std::string("old history"), ReadText(work_dir + "/history/old.log"));
  static_cast<void>(LoadValidatedBundle(work_dir, course::InputMode::kUnit));
}

TEST_CASE(training_refuses_malformed_reserved_pair) {
  const std::string root = MakeScratchDirectory("training_malformed_pair");
  const std::uint32_t sample_count = 6;
  const FixturePaths fixture =
      WriteFixture(root + "/data", PatternUnitImages(sample_count),
                   CyclicLabels(sample_count));
  const course::TrainOptions options =
      MakeOptions(fixture, root + "/work", course::InputMode::kUnit,
                  sample_count);

  // A lone, junk reserved artifact must be refused rather than overwritten or
  // "recovered" by clearing the directory.
  MakeDirectory(options.work_dir);
  WriteText(options.work_dir + "/lenet.bin", "not a checkpoint");
  std::ostringstream diagnostics;
  EXPECT_THROWS_AS(course::RunCourseTrain(options, diagnostics),
                   std::runtime_error);
  EXPECT_EQ(std::string("not a checkpoint"),
            ReadText(options.work_dir + "/lenet.bin"));
  EXPECT_FALSE(PathExists(options.work_dir + "/course.meta"));
}

TEST_CASE(float_training_keeps_fractional_pixels) {
  const std::string root = MakeScratchDirectory("training_fractional_pixels");
  const std::uint32_t sample_count = 8;

  // 25.1F and 25.4F collapse to the same uint8 under both truncation and
  // rounding, yet differ as FP32 source pixels.
  const float fractional_left = 25.1F;
  const float fractional_right = 25.4F;
  EXPECT_EQ(static_cast<int>(fractional_left),
            static_cast<int>(fractional_right));
  EXPECT_EQ(static_cast<int>(fractional_left + 0.5F),
            static_cast<int>(fractional_right + 0.5F));

  const FixturePaths left_fixture =
      WriteFixture(root + "/left_data",
                   ConstantByteImages(sample_count, fractional_left),
                   CyclicLabels(sample_count));
  const FixturePaths right_fixture =
      WriteFixture(root + "/right_data",
                   ConstantByteImages(sample_count, fractional_right),
                   CyclicLabels(sample_count));

  const std::string left_work = root + "/left_work";
  const std::string right_work = root + "/right_work";
  MakeDirectory(left_work);
  MakeDirectory(right_work);
  WriteText(left_work + "/notes.txt", "left");
  WriteText(right_work + "/notes.txt", "right");

  const course::TrainOptions left_options =
      MakeOptions(left_fixture, left_work, course::InputMode::kByte,
                  sample_count);
  const course::TrainOptions right_options =
      MakeOptions(right_fixture, right_work, course::InputMode::kByte,
                  sample_count);
  std::ostringstream left_diagnostics;
  std::ostringstream right_diagnostics;
  EXPECT_EQ(0, course::RunCourseTrain(left_options, left_diagnostics));
  EXPECT_EQ(0, course::RunCourseTrain(right_options, right_diagnostics));

  // Unrelated entries survive training in both supplied directories.
  EXPECT_EQ(std::string("left"), ReadText(left_work + "/notes.txt"));
  EXPECT_EQ(std::string("right"), ReadText(right_work + "/notes.txt"));

  const course::CourseModel left_bundle =
      LoadValidatedBundle(left_work, course::InputMode::kByte);
  const course::CourseModel right_bundle =
      LoadValidatedBundle(right_work, course::InputMode::kByte);

  // Both runs really optimized from the same seeded initialization.
  const ParameterSet initial = SeededInitialParameters(kSeed);
  EXPECT_TRUE(
      CountParameterBitDifferences(initial, left_bundle.checkpoint.parameters) >
      0);
  EXPECT_TRUE(
      CountParameterBitDifferences(initial, right_bundle.checkpoint.parameters) >
      0);

  // The fractional difference survives GPU preparation and changes the trained
  // weights, so this is not a source-file preservation check.
  EXPECT_TRUE(CountParameterBitDifferences(left_bundle.checkpoint.parameters,
                                           right_bundle.checkpoint.parameters) >
              0);

  const course::Shape shape = FixtureShape(sample_count);
  const std::vector<float> left_raw =
      course::ReadImages(left_fixture.images, shape, course::InputMode::kByte);
  const std::vector<float> right_raw =
      course::ReadImages(right_fixture.images, shape, course::InputMode::kByte);
  EXPECT_FALSE(SameBits(left_raw[0], right_raw[0]));
  EXPECT_TRUE(SameBits(fractional_left, left_raw[0]));
  EXPECT_TRUE(SameBits(fractional_right, right_raw[0]));

  // The persisted kByte mode is usable directly by both forward providers.
  const std::vector<std::uint32_t> all_indices = AllIndices(sample_count);
  const DeviceBuffer<float> normalized =
      PrepareNormalized(left_raw, all_indices, course::InputMode::kByte, kSeed,
                        nullptr);
  const int batch = static_cast<int>(sample_count);
  const std::vector<float> course_logits =
      CourseLogits(left_bundle.checkpoint.parameters, normalized, batch);
  const std::vector<float> reference_logits =
      ReferenceLogits(left_bundle.checkpoint.parameters, normalized, batch);
  EXPECT_TRUE(std::memcmp(reference_logits.data(), course_logits.data(),
                          course_logits.size() * sizeof(float)) == 0);
  ExpectFiniteLogits(course_logits);
}

TEST_CASE(validation_uses_training_only_indices) {
  const std::string root = MakeScratchDirectory("training_validation_split");
  const std::uint32_t sample_count = 15;
  const std::uint32_t validation_count =
      course::CourseValidationCount(sample_count);
  EXPECT_EQ(3u, validation_count);

  // Reproduce the original deterministic split as the oracle. Every pixel is
  // identical, so every prediction within one evaluation is uniform. Labels
  // cover every class across the partitions such that, for any one predicted
  // class, the validation, training-subset, and all-sample accuracies are all
  // different. This keeps the subset distinction strong without assuming that
  // correct training predicts class 0 or 1.
  const DatasetSplit split =
      MakeTrainValidationSplit(sample_count, validation_count, kSeed);
  EXPECT_EQ(static_cast<std::size_t>(sample_count - validation_count),
            split.training_indices.size());
  EXPECT_EQ(static_cast<std::size_t>(validation_count),
            split.validation_indices.size());
  std::vector<int> visits(sample_count, 0);
  for (const std::uint32_t index : split.training_indices) {
    EXPECT_TRUE(index < sample_count);
    ++visits[index];
  }
  for (const std::uint32_t index : split.validation_indices) {
    EXPECT_TRUE(index < sample_count);
    ++visits[index];
  }
  for (const int count : visits) {
    EXPECT_EQ(1, count);
  }

  // The original split helper is deterministic.
  const DatasetSplit repeated =
      MakeTrainValidationSplit(sample_count, validation_count, kSeed);
  EXPECT_TRUE(repeated.training_indices == split.training_indices);
  EXPECT_TRUE(repeated.validation_indices == split.validation_indices);

  // Validation keeps three distinct labels; training covers all ten classes.
  // The resulting per-class count fractions differ for every possible uniform
  // predicted class.
  std::vector<std::int32_t> labels(sample_count, 0);
  for (std::size_t rank = 0; rank < split.validation_indices.size(); ++rank) {
    labels[split.validation_indices[rank]] =
        static_cast<std::int32_t>(rank % 3);
  }
  for (std::size_t rank = 0; rank < split.training_indices.size(); ++rank) {
    labels[split.training_indices[rank]] =
        static_cast<std::int32_t>(rank % kClassCount);
  }
  const FixturePaths fixture =
      WriteFixture(root + "/data", ConstantUnitImages(sample_count, 0.25F),
                   labels);

  const course::TrainOptions options =
      MakeOptions(fixture, root + "/work", course::InputMode::kUnit,
                  sample_count);
  std::ostringstream diagnostics;
  EXPECT_EQ(0, course::RunCourseTrain(options, diagnostics));

  const course::CourseModel bundle =
      LoadValidatedBundle(root + "/work", course::InputMode::kUnit);
  EXPECT_EQ(1u, bundle.checkpoint.metadata.best_epoch);

  const course::Shape shape = FixtureShape(sample_count);
  const std::vector<float> raw_images =
      course::ReadImages(fixture.images, shape, course::InputMode::kUnit);
  const std::vector<std::uint8_t> raw_labels =
      course::ReadLabels(fixture.labels, shape);

  const float validation_accuracy =
      AccuracyOnIndices(bundle.checkpoint.parameters, raw_images, raw_labels,
                        split.validation_indices, course::InputMode::kUnit);
  const float training_accuracy =
      AccuracyOnIndices(bundle.checkpoint.parameters, raw_images, raw_labels,
                        split.training_indices, course::InputMode::kUnit);
  const float all_accuracy =
      AccuracyOnIndices(bundle.checkpoint.parameters, raw_images, raw_labels,
                        AllIndices(sample_count), course::InputMode::kUnit);

  // Uniform predictions and the per-partition label distribution keep the
  // held-out partition unambiguous against both the training partition and the
  // whole fixture, whatever class the trained model prefers.
  EXPECT_FALSE(SameBits(validation_accuracy, training_accuracy));
  EXPECT_FALSE(SameBits(validation_accuracy, all_accuracy));
  EXPECT_FALSE(SameBits(training_accuracy, all_accuracy));
  EXPECT_TRUE(SameBits(bundle.checkpoint.metadata.validation_accuracy,
                       validation_accuracy));
}

TEST_CASE(training_rejects_invalid_options_before_work) {
  const std::string root = MakeScratchDirectory("training_invalid_options");
  const FixturePaths fixture =
      WriteFixture(root + "/data", PatternUnitImages(4), CyclicLabels(4));
  const course::TrainOptions valid =
      MakeOptions(fixture, root + "/work", course::InputMode::kUnit, 4);

  course::TrainOptions options = valid;
  options.epochs = 0;
  std::ostringstream epochs_diagnostics;
  EXPECT_THROWS_AS(course::RunCourseTrain(options, epochs_diagnostics),
                   std::invalid_argument);

  options = valid;
  options.shape.n = 1;
  std::ostringstream count_diagnostics;
  EXPECT_THROWS_AS(course::RunCourseTrain(options, count_diagnostics),
                   std::invalid_argument);

  options = valid;
  options.shape.h = 27;
  std::ostringstream shape_diagnostics;
  EXPECT_THROWS_AS(course::RunCourseTrain(options, shape_diagnostics),
                   std::invalid_argument);

  options = valid;
  options.batch_size = 0;
  std::ostringstream batch_diagnostics;
  EXPECT_THROWS_AS(course::RunCourseTrain(options, batch_diagnostics),
                   std::invalid_argument);

  options = valid;
  options.batch_size =
      static_cast<std::uint32_t>(std::numeric_limits<int>::max()) + 1u;
  std::ostringstream large_batch_diagnostics;
  EXPECT_THROWS_AS(course::RunCourseTrain(options, large_batch_diagnostics),
                   std::invalid_argument);

  options = valid;
  options.input_mode = static_cast<course::InputMode>(99);
  std::ostringstream mode_diagnostics;
  EXPECT_THROWS_AS(course::RunCourseTrain(options, mode_diagnostics),
                   std::invalid_argument);

  options = valid;
  options.work_dir.clear();
  std::ostringstream path_diagnostics;
  EXPECT_THROWS_AS(course::RunCourseTrain(options, path_diagnostics),
                   std::invalid_argument);

  // None of the rejected calls may create the selected directory or a bundle.
  EXPECT_FALSE(PathExists(root + "/work"));
}

// ---------------------------------------------------------------------------
// Production-used pure policies
// ---------------------------------------------------------------------------

TEST_CASE(training_hyperparameters_and_schedule) {
  // Split size rule: 60000 -> 5000, otherwise max(1, N / 5).
  EXPECT_EQ(5000u, course::CourseValidationCount(60000));
  EXPECT_EQ(1u, course::CourseValidationCount(2));
  EXPECT_EQ(1u, course::CourseValidationCount(5));
  EXPECT_EQ(1u, course::CourseValidationCount(9));
  EXPECT_EQ(2u, course::CourseValidationCount(10));
  EXPECT_EQ(3u, course::CourseValidationCount(19));
  EXPECT_EQ(4u, course::CourseValidationCount(20));
  EXPECT_THROWS_AS(course::CourseValidationCount(0), std::invalid_argument);
  EXPECT_THROWS_AS(course::CourseValidationCount(1), std::invalid_argument);

  // Pinned piecewise schedule: 1..12, 13..17, 18+.
  EXPECT_TRUE(SameBits(1.0e-3F, course::CourseLearningRate(1)));
  EXPECT_TRUE(SameBits(1.0e-3F, course::CourseLearningRate(12)));
  EXPECT_TRUE(SameBits(1.0e-4F, course::CourseLearningRate(13)));
  EXPECT_TRUE(SameBits(1.0e-4F, course::CourseLearningRate(17)));
  EXPECT_TRUE(SameBits(1.0e-5F, course::CourseLearningRate(18)));
  EXPECT_TRUE(SameBits(1.0e-5F, course::CourseLearningRate(1000)));
  EXPECT_THROWS_AS(course::CourseLearningRate(0), std::invalid_argument);

  // Pinned decoupled AdamW configuration, with bias decay applied by LeNet.
  const AdamWConfig config = course::CourseAdamWConfig();
  EXPECT_TRUE(SameBits(0.9F, config.beta1));
  EXPECT_TRUE(SameBits(0.999F, config.beta2));
  EXPECT_TRUE(SameBits(1.0e-8F, config.epsilon));
  EXPECT_TRUE(SameBits(1.0e-4F, config.weight_decay));
}

TEST_CASE(best_selection_requires_strict_validation_improvement) {
  course::BestValidationSelection selection;
  EXPECT_EQ(0u, selection.best_epoch());
  EXPECT_TRUE(SameBits(-1.0F, selection.best_accuracy()));

  // The first valid observation is selected even when accuracy is low.
  EXPECT_TRUE(selection.Consider(1, 0.25F));
  EXPECT_EQ(1u, selection.best_epoch());
  EXPECT_TRUE(SameBits(0.25F, selection.best_accuracy()));

  // Equal and lower values never replace the current best.
  EXPECT_FALSE(selection.Consider(2, 0.25F));
  EXPECT_FALSE(selection.Consider(3, 0.10F));
  EXPECT_EQ(1u, selection.best_epoch());
  EXPECT_TRUE(SameBits(0.25F, selection.best_accuracy()));

  EXPECT_TRUE(selection.Consider(4, 0.75F));
  EXPECT_EQ(4u, selection.best_epoch());
  EXPECT_TRUE(SameBits(0.75F, selection.best_accuracy()));
  EXPECT_FALSE(selection.Consider(5, 0.50F));

  EXPECT_TRUE(selection.Consider(7, 1.0F));
  EXPECT_EQ(7u, selection.best_epoch());
  EXPECT_FALSE(selection.Consider(8, 1.0F));
  EXPECT_EQ(7u, selection.best_epoch());

  // Invalid observations are rejected without changing the selected state.
  EXPECT_THROWS_AS(selection.Consider(0, 0.5F), std::invalid_argument);
  EXPECT_THROWS_AS(selection.Consider(6, -0.01F), std::invalid_argument);
  EXPECT_THROWS_AS(selection.Consider(6, 1.01F), std::invalid_argument);
  EXPECT_THROWS_AS(
      selection.Consider(6, std::numeric_limits<float>::quiet_NaN()),
      std::invalid_argument);
  EXPECT_EQ(7u, selection.best_epoch());
  EXPECT_TRUE(SameBits(1.0F, selection.best_accuracy()));
}

}  // namespace
