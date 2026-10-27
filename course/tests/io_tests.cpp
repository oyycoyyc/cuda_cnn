// Tests-first contract for course/interface.h and course/model_io.h.
//
// Every assertion below is written against the approved public API before an
// implementation exists, so the suite is expected to fail at build/link time
// until course/io.cpp and course/model_io.cpp are implemented. Fixtures use
// the original checkpoint and ParameterSet helpers, real raw array files, and
// real model bundles in a build-owned scratch tree; nothing mocks the parser
// or the file format.

#include "course/interface.h"
#include "course/model_io.h"
#include "course/tests/test_support.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

using course_test_support::Crc32;
using course_test_support::CreateHardLink;
using course_test_support::CreateSymlink;
using course_test_support::Hex8;
using course_test_support::IsSymlink;
using course_test_support::MakeDirectory;
using course_test_support::MakeScratchDirectory;
using course_test_support::PathExists;
using course_test_support::ReadBytes;
using course_test_support::ReadText;
using course_test_support::ReplaceAll;
using course_test_support::SameBits;
using course_test_support::ScratchRoot;
using course_test_support::ToUpper;
using course_test_support::WriteBytes;
using course_test_support::WriteFloats;
using course_test_support::WriteInt32s;
using course_test_support::WriteText;

// A single supported image is 1x28x28 FP32 values.
const std::size_t kImageValueCount = 28u * 28u;

// Returns the only supported geometry with the requested batch.
course::Shape UnitShape(std::uint32_t batch) {
  return course::Shape{batch, 1, 28, 28, 10};
}

// Builds a well-formed train argument vector with optional N/C/H/W/K.
std::vector<const char*> TrainArgv(const char* n = "2", const char* c = "1",
                                   const char* h = "28", const char* w = "28",
                                   const char* k = "10") {
  return std::vector<const char*>{"train", "images.f32", "labels.i32", n, c, h,
                                  w,       k,            "work"};
}

// Builds a well-formed infer argument vector with optional N.
std::vector<const char*> InferArgv(const char* n = "1") {
  return std::vector<const char*>{"infer", "images.f32", n, "1", "28", "28",
                                  "10"};
}

// Parses with the vector's own extent, which is also the tested argc.
void ExpectTrainRejected(const std::vector<const char*>& argv,
                         const char* input_mode, const char* epochs) {
  EXPECT_THROWS(course::ParseTrainArgs(static_cast<int>(argv.size()),
                                       argv.data(), input_mode, epochs));
}

// Parses infer with the vector's own extent, which is also the tested argc.
void ExpectInferRejected(const std::vector<const char*>& argv,
                         const char* model_dir, const char* batch,
                         const char* profile) {
  EXPECT_THROWS(course::ParseInferArgs(static_cast<int>(argv.size()),
                                       argv.data(), model_dir, batch, profile));
}

// Fills a full image array with the requested leading values.
std::vector<float> ImageValues(const std::vector<float>& head, float fill) {
  std::vector<float> values(kImageValueCount, fill);
  for (std::size_t index = 0; index < head.size(); ++index) {
    values[index] = head[index];
  }
  return values;
}

// Writes a raw image array and returns its path.
std::string SaveImages(const std::string& directory, const char* name,
                       const std::vector<float>& values) {
  const std::string path = directory + "/" + name;
  WriteFloats(path, values);
  return path;
}

// Writes a raw label array and returns its path.
std::string SaveLabels(const std::string& directory, const char* name,
                       const std::vector<std::int32_t>& values) {
  const std::string path = directory + "/" + name;
  WriteInt32s(path, values);
  return path;
}

// Canonical valid tiny model fixture built from the original ParameterSet
// helper, including signed zero, a denormal, and a non-representable fraction.
Checkpoint SampleCheckpoint() {
  Checkpoint checkpoint{{12, 0.75F, 0.1307F, 0.3081F},
                        CreateLenetParameters()};
  std::uint32_t ordinal = 0;
  for (ParameterTensor& tensor : checkpoint.parameters) {
    for (float& value : tensor.values) {
      value = static_cast<float>(static_cast<int>(ordinal % 201) - 100) /
              64.0F;
      ++ordinal;
    }
  }
  checkpoint.parameters[0].values[0] = -0.0F;
  checkpoint.parameters[0].values[1] =
      std::numeric_limits<float>::denorm_min();
  checkpoint.parameters[3].values[2] = 1.0F / 3.0F;
  return checkpoint;
}

// A different valid checkpoint whose bytes must not match SampleCheckpoint.
Checkpoint MutatedCheckpoint() {
  Checkpoint checkpoint = SampleCheckpoint();
  checkpoint.parameters[5].values[0] += 1.0F;
  return checkpoint;
}

// Requires two checkpoints to agree field by field and bit by bit.
void ExpectEquivalentCheckpoints(const Checkpoint& expected,
                                 const Checkpoint& actual) {
  EXPECT_EQ(expected.metadata.best_epoch, actual.metadata.best_epoch);
  EXPECT_TRUE(
      SameBits(expected.metadata.validation_accuracy,
               actual.metadata.validation_accuracy));
  EXPECT_TRUE(SameBits(expected.metadata.normalization_mean,
                       actual.metadata.normalization_mean));
  EXPECT_TRUE(SameBits(expected.metadata.normalization_stddev,
                       actual.metadata.normalization_stddev));
  EXPECT_EQ(expected.parameters.size(), actual.parameters.size());
  if (expected.parameters.size() != actual.parameters.size()) {
    return;
  }
  for (std::size_t index = 0; index < expected.parameters.size(); ++index) {
    const ParameterTensor& expected_tensor = expected.parameters[index];
    const ParameterTensor& actual_tensor = actual.parameters[index];
    EXPECT_EQ(expected_tensor.name, actual_tensor.name);
    EXPECT_EQ(expected_tensor.rank, actual_tensor.rank);
    EXPECT_EQ(expected_tensor.dimensions, actual_tensor.dimensions);
    EXPECT_EQ(expected_tensor.values.size(), actual_tensor.values.size());
    if (expected_tensor.values.size() != actual_tensor.values.size()) {
      continue;
    }
    for (std::size_t value = 0; value < expected_tensor.values.size();
         ++value) {
      EXPECT_TRUE(SameBits(expected_tensor.values[value],
                           actual_tensor.values[value]));
    }
  }
}

// Builds the exact four-line course.meta text for a mode and CRC spelling.
std::string MetadataText(const std::string& input_mode,
                         const std::string& crc) {
  return "course_lenet_v1\ninput_mode=" + input_mode +
         "\nshape=1,28,28,10\ncheckpoint_crc32=" + crc + "\n";
}

// Returns the metadata CRC spelling for the checkpoint currently on disk.
std::string BundleCrc(const std::string& directory) {
  return Hex8(Crc32(ReadBytes(directory + "/lenet.bin")));
}

// Publishes a valid unit bundle, replaces its metadata with the supplied text
// (substituting the real CRC for %CRC%), and requires the loader to reject it.
void ExpectMetadataRejected(const char* label, std::string metadata) {
  const std::string directory =
      MakeScratchDirectory(std::string("metadata_reject_") + label);
  course::SaveCourseModel(directory, SampleCheckpoint(),
                          course::InputMode::kUnit);
  ReplaceAll(&metadata, "%CRC%", BundleCrc(directory));
  WriteText(directory + "/course.meta", metadata);
  EXPECT_THROWS(course::LoadCourseModel(directory));
}

// ---------------------------------------------------------------------------
// InputMode parsing
// ---------------------------------------------------------------------------

TEST_CASE(ParseInputMode_maps_exact_names_and_null) {
  EXPECT_EQ(course::InputMode::kUnit, course::ParseInputMode(nullptr));
  EXPECT_EQ(course::InputMode::kUnit, course::ParseInputMode("unit"));
  EXPECT_EQ(course::InputMode::kByte, course::ParseInputMode("byte"));
  EXPECT_EQ(course::InputMode::kNormalized,
            course::ParseInputMode("normalized"));
}

TEST_CASE(ParseInputMode_rejects_empty_padded_case_and_unknown) {
  const char* const invalid[] = {"",   "Unit",     "UNIT",  " unit",
                                 "unit ", "byte\n", "kUnit", "normalise",
                                 "float"};
  for (const char* value : invalid) {
    EXPECT_THROWS(course::ParseInputMode(value));
  }
}

// ---------------------------------------------------------------------------
// Train argument parsing
// ---------------------------------------------------------------------------

TEST_CASE(ParseTrainArgs_accepts_exact_argc_and_defaults) {
  const std::vector<const char*> argv = TrainArgv();
  const course::TrainOptions options =
      course::ParseTrainArgs(9, argv.data(), nullptr, nullptr);
  EXPECT_EQ(std::string("images.f32"), options.images_path);
  EXPECT_EQ(std::string("labels.i32"), options.labels_path);
  EXPECT_EQ(std::string("work"), options.work_dir);
  EXPECT_EQ(2u, options.shape.n);
  EXPECT_EQ(1u, options.shape.c);
  EXPECT_EQ(28u, options.shape.h);
  EXPECT_EQ(28u, options.shape.w);
  EXPECT_EQ(10u, options.shape.k);
  EXPECT_EQ(course::InputMode::kUnit, options.input_mode);
  EXPECT_EQ(20u, options.epochs);
  EXPECT_EQ(128u, options.batch_size);
  EXPECT_EQ(static_cast<std::uint64_t>(1337), options.seed);
}

TEST_CASE(ParseTrainArgs_accepts_explicit_mode_and_epochs) {
  const std::vector<const char*> argv = TrainArgv("3");
  const course::TrainOptions options =
      course::ParseTrainArgs(9, argv.data(), "byte", "1000");
  EXPECT_EQ(3u, options.shape.n);
  EXPECT_EQ(course::InputMode::kByte, options.input_mode);
  EXPECT_EQ(1000u, options.epochs);
  EXPECT_EQ(128u, options.batch_size);
  EXPECT_EQ(static_cast<std::uint64_t>(1337), options.seed);

  const course::TrainOptions normalized =
      course::ParseTrainArgs(9, argv.data(), "normalized", "1");
  EXPECT_EQ(course::InputMode::kNormalized, normalized.input_mode);
  EXPECT_EQ(1u, normalized.epochs);
}

TEST_CASE(ParseTrainArgs_rejects_wrong_argc) {
  std::vector<const char*> short_argv = TrainArgv();
  short_argv.pop_back();
  EXPECT_THROWS(course::ParseTrainArgs(8, short_argv.data(), nullptr, nullptr));

  std::vector<const char*> long_argv = TrainArgv();
  long_argv.push_back("extra");
  EXPECT_THROWS(course::ParseTrainArgs(10, long_argv.data(), nullptr, nullptr));

  const std::vector<const char*> argv = TrainArgv();
  EXPECT_THROWS(course::ParseTrainArgs(8, argv.data(), nullptr, nullptr));
}

TEST_CASE(ParseTrainArgs_rejects_signed_zero_below_two_and_overflow_n) {
  const char* const invalid_n[] = {
      "-1", "0", "1", "4294967296", "99999999999999999999", "", "2.0", "2x"};
  for (const char* value : invalid_n) {
    ExpectTrainRejected(TrainArgv(value), nullptr, nullptr);
  }
}

TEST_CASE(ParseTrainArgs_rejects_unsupported_dimensions) {
  ExpectTrainRejected(TrainArgv("2", "2", "28", "28", "10"), nullptr, nullptr);
  ExpectTrainRejected(TrainArgv("2", "0", "28", "28", "10"), nullptr, nullptr);
  ExpectTrainRejected(TrainArgv("2", "1", "27", "28", "10"), nullptr, nullptr);
  ExpectTrainRejected(TrainArgv("2", "1", "28", "29", "10"), nullptr, nullptr);
  ExpectTrainRejected(TrainArgv("2", "1", "28", "28", "9"), nullptr, nullptr);
}

TEST_CASE(ParseTrainArgs_rejects_invalid_input_mode) {
  const char* const invalid_mode[] = {"", "Unit", "unit ", "float", "byte\n"};
  for (const char* value : invalid_mode) {
    ExpectTrainRejected(TrainArgv(), value, nullptr);
  }
}

TEST_CASE(ParseTrainArgs_rejects_invalid_epochs) {
  const char* const invalid_epochs[] = {"",   "0",    "1001", "20x",
                                        "-1", "1e2",  " 20",  "twenty"};
  for (const char* value : invalid_epochs) {
    ExpectTrainRejected(TrainArgv(), nullptr, value);
  }
}

TEST_CASE(ParseTrainArgs_rejects_empty_required_paths) {
  const std::vector<const char*> empty_images = {
      "train", "", "labels.i32", "2", "1", "28", "28", "10", "work"};
  EXPECT_THROWS(course::ParseTrainArgs(9, empty_images.data(), nullptr,
                                       nullptr));

  const std::vector<const char*> empty_labels = {
      "train", "images.f32", "", "2", "1", "28", "28", "10", "work"};
  EXPECT_THROWS(course::ParseTrainArgs(9, empty_labels.data(), nullptr,
                                       nullptr));

  const std::vector<const char*> empty_work = {
      "train", "images.f32", "labels.i32", "2", "1", "28", "28", "10", ""};
  EXPECT_THROWS(
      course::ParseTrainArgs(9, empty_work.data(), nullptr, nullptr));
}

// ---------------------------------------------------------------------------
// Infer argument parsing
// ---------------------------------------------------------------------------

TEST_CASE(ParseInferArgs_accepts_exact_argc_and_defaults) {
  const std::vector<const char*> argv = InferArgv();
  const course::InferOptions options =
      course::ParseInferArgs(7, argv.data(), "model/run", nullptr, nullptr);
  EXPECT_EQ(std::string("images.f32"), options.images_path);
  EXPECT_EQ(std::string("model/run"), options.model_dir);
  EXPECT_EQ(1u, options.shape.n);
  EXPECT_EQ(1u, options.shape.c);
  EXPECT_EQ(28u, options.shape.h);
  EXPECT_EQ(28u, options.shape.w);
  EXPECT_EQ(10u, options.shape.k);
  EXPECT_EQ(1024u, options.batch_size);
  EXPECT_FALSE(options.profile);
}

TEST_CASE(ParseInferArgs_accepts_batch_bounds_and_exact_profile) {
  const std::vector<const char*> argv = InferArgv("16384");
  const course::InferOptions enabled =
      course::ParseInferArgs(7, argv.data(), "model/run", "16384", "1");
  EXPECT_EQ(16384u, enabled.batch_size);
  EXPECT_TRUE(enabled.profile);

  const course::InferOptions single =
      course::ParseInferArgs(7, argv.data(), "model/run", "1", "1");
  EXPECT_EQ(1u, single.batch_size);
  EXPECT_TRUE(single.profile);
}

TEST_CASE(ParseInferArgs_profile_only_enables_on_exact_one) {
  const std::vector<const char*> argv = InferArgv();
  const char* const disabled[] = {"0", "true", "", "01", " 1", "2"};
  for (const char* value : disabled) {
    const course::InferOptions options =
        course::ParseInferArgs(7, argv.data(), "model/run", nullptr, value);
    EXPECT_FALSE(options.profile);
  }
}

TEST_CASE(ParseInferArgs_rejects_wrong_argc) {
  std::vector<const char*> short_argv = InferArgv();
  short_argv.pop_back();
  EXPECT_THROWS(
      course::ParseInferArgs(6, short_argv.data(), "model/run", nullptr,
                             nullptr));

  std::vector<const char*> long_argv = InferArgv();
  long_argv.push_back("extra");
  EXPECT_THROWS(
      course::ParseInferArgs(8, long_argv.data(), "model/run", nullptr,
                             nullptr));
}

TEST_CASE(ParseInferArgs_requires_nonempty_model_dir) {
  const std::vector<const char*> argv = InferArgv();
  EXPECT_THROWS(
      course::ParseInferArgs(7, argv.data(), nullptr, nullptr, nullptr));
  EXPECT_THROWS(course::ParseInferArgs(7, argv.data(), "", nullptr, nullptr));
}

TEST_CASE(ParseInferArgs_rejects_invalid_batch) {
  const char* const invalid_batch[] = {"0",  "16385", "",     "abc",
                                       "-1", "1e3",   "1024x"};
  for (const char* value : invalid_batch) {
    ExpectInferRejected(InferArgv(), "model/run", value, nullptr);
  }
}

TEST_CASE(ParseInferArgs_rejects_invalid_n_and_dimensions) {
  ExpectInferRejected(InferArgv("-1"), "model/run", nullptr, nullptr);
  ExpectInferRejected(InferArgv("0"), "model/run", nullptr, nullptr);
  ExpectInferRejected(InferArgv("4294967296"), "model/run", nullptr, nullptr);

  const std::vector<const char*> wrong_channels = {
      "infer", "images.f32", "1", "2", "28", "28", "10"};
  EXPECT_THROWS(course::ParseInferArgs(7, wrong_channels.data(), "model/run",
                                       nullptr, nullptr));

  const std::vector<const char*> wrong_height = {
      "infer", "images.f32", "1", "1", "27", "28", "10"};
  EXPECT_THROWS(course::ParseInferArgs(7, wrong_height.data(), "model/run",
                                       nullptr, nullptr));

  const std::vector<const char*> wrong_width = {
      "infer", "images.f32", "1", "1", "28", "29", "10"};
  EXPECT_THROWS(course::ParseInferArgs(7, wrong_width.data(), "model/run",
                                       nullptr, nullptr));

  const std::vector<const char*> wrong_classes = {
      "infer", "images.f32", "1", "1", "28", "28", "9"};
  EXPECT_THROWS(course::ParseInferArgs(7, wrong_classes.data(), "model/run",
                                       nullptr, nullptr));

  const std::vector<const char*> empty_images = {
      "infer", "", "1", "1", "28", "28", "10"};
  EXPECT_THROWS(course::ParseInferArgs(7, empty_images.data(), "model/run",
                                       nullptr, nullptr));
}

TEST_CASE(ErrorTaxonomy_usage_and_shape_errors_throw_invalid_argument) {
  EXPECT_THROWS_AS(course::ParseInputMode("Unit"), std::invalid_argument);

  std::vector<const char*> short_train = TrainArgv();
  short_train.pop_back();
  EXPECT_THROWS_AS(
      course::ParseTrainArgs(8, short_train.data(), nullptr, nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      course::ParseTrainArgs(9, TrainArgv("0").data(), nullptr, nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      course::ParseTrainArgs(9, TrainArgv("2", "28", "28", "10", "9").data(),
                             nullptr, nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      course::ParseTrainArgs(9, TrainArgv().data(), "unknown", nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      course::ParseTrainArgs(9, TrainArgv().data(), nullptr, "1001"),
      std::invalid_argument);

  EXPECT_THROWS_AS(
      course::ParseInferArgs(7, InferArgv().data(), nullptr, nullptr, nullptr),
      std::invalid_argument);
  EXPECT_THROWS_AS(
      course::ParseInferArgs(7, InferArgv().data(), "model/run", "16385",
                             nullptr),
      std::invalid_argument);

  const std::string directory =
      MakeScratchDirectory("taxonomy_usage_errors");
  const std::string unused_path = directory + "/unused.f32";
  EXPECT_THROWS_AS(course::ReadImages("", UnitShape(1),
                                      course::InputMode::kUnit),
                   std::invalid_argument);
  EXPECT_THROWS_AS(course::ReadImages(unused_path,
                                      course::Shape{1, 2, 28, 28, 10},
                                      course::InputMode::kUnit),
                   std::invalid_argument);
  EXPECT_THROWS_AS(course::ReadLabels("", UnitShape(2)),
                   std::invalid_argument);
  EXPECT_THROWS_AS(course::ReadLabels(unused_path,
                                      course::Shape{1, 1, 28, 28, 9}),
                   std::invalid_argument);
}

TEST_CASE(ErrorTaxonomy_corrupt_file_data_throws_runtime_error) {
  const std::string directory =
      MakeScratchDirectory("taxonomy_corrupt_data");
  const std::vector<float> short_images(kImageValueCount - 1, 0.25F);
  const std::string short_images_path =
      SaveImages(directory, "short.f32", short_images);
  EXPECT_THROWS_AS(
      course::ReadImages(short_images_path, UnitShape(1),
                         course::InputMode::kUnit),
      std::runtime_error);

  const std::string out_of_range_path =
      SaveImages(directory, "range.f32", ImageValues({1.5F}, 0.25F));
  EXPECT_THROWS_AS(
      course::ReadImages(out_of_range_path, UnitShape(1),
                         course::InputMode::kUnit),
      std::runtime_error);

  const std::string nan_path = SaveImages(
      directory, "nan.f32", ImageValues({std::numeric_limits<float>::quiet_NaN()},
                                        0.25F));
  EXPECT_THROWS_AS(
      course::ReadImages(nan_path, UnitShape(1), course::InputMode::kUnit),
      std::runtime_error);

  const std::string bad_label_path =
      SaveLabels(directory, "bad_label.i32", std::vector<std::int32_t>{-1, 3});
  EXPECT_THROWS_AS(course::ReadLabels(bad_label_path, UnitShape(2)),
                   std::runtime_error);

  const std::string short_label_path =
      SaveLabels(directory, "short_label.i32", std::vector<std::int32_t>{3});
  EXPECT_THROWS_AS(course::ReadLabels(short_label_path, UnitShape(2)),
                   std::runtime_error);

  const std::string empty_model = MakeScratchDirectory("taxonomy_missing_model");
  EXPECT_THROWS_AS(course::LoadCourseModel(empty_model), std::runtime_error);

  const std::string malformed_model =
      MakeScratchDirectory("taxonomy_malformed_metadata");
  course::SaveCourseModel(malformed_model, SampleCheckpoint(),
                          course::InputMode::kUnit);
  WriteText(malformed_model + "/course.meta", "not metadata\n");
  EXPECT_THROWS_AS(course::LoadCourseModel(malformed_model),
                   std::runtime_error);

  Checkpoint broken = SampleCheckpoint();
  broken.parameters[0].values[0] =
      std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROWS_AS(
      course::SaveCourseModel(MakeScratchDirectory("taxonomy_broken_save"),
                              broken, course::InputMode::kUnit),
      std::runtime_error);
}

// ---------------------------------------------------------------------------
// Raw image reading
// ---------------------------------------------------------------------------

TEST_CASE(ReadImages_preserves_fractional_unit_values_exactly) {
  const std::string directory =
      MakeScratchDirectory("read_images_fractional");
  const std::vector<float> head = {
      0.0F,
      -0.0F,
      1.0F,
      0.5F,
      1.0F / 3.0F,
      0.1F,
      0.99999994F,
      std::numeric_limits<float>::denorm_min(),
  };
  const std::vector<float> values = ImageValues(head, 0.25F);
  const std::string path = SaveImages(directory, "images.f32", values);
  const std::vector<unsigned char> before = ReadBytes(path);

  const std::vector<float> loaded =
      course::ReadImages(path, UnitShape(1), course::InputMode::kUnit);
  EXPECT_EQ(values.size(), loaded.size());
  if (values.size() == loaded.size()) {
    for (std::size_t index = 0; index < values.size(); ++index) {
      EXPECT_TRUE(SameBits(values[index], loaded[index]));
    }
  }
  EXPECT_TRUE(before == ReadBytes(path));
}

TEST_CASE(ReadImages_accepts_unit_and_byte_bounds) {
  const std::string directory =
      MakeScratchDirectory("read_images_mode_bounds");

  const std::vector<float> unit =
      ImageValues({0.0F, 1.0F, -0.0F, 0.99999994F}, 0.5F);
  const std::vector<float> unit_loaded = course::ReadImages(
      SaveImages(directory, "unit.f32", unit), UnitShape(1),
      course::InputMode::kUnit);
  EXPECT_EQ(unit.size(), unit_loaded.size());
  if (unit.size() == unit_loaded.size()) {
    for (std::size_t index = 0; index < unit.size(); ++index) {
      EXPECT_TRUE(SameBits(unit[index], unit_loaded[index]));
    }
  }

  const std::vector<float> byte =
      ImageValues({0.0F, 255.0F, 1.0F, 254.0F}, 128.0F);
  const std::vector<float> byte_loaded = course::ReadImages(
      SaveImages(directory, "byte.f32", byte), UnitShape(1),
      course::InputMode::kByte);
  EXPECT_EQ(byte.size(), byte_loaded.size());
  if (byte.size() == byte_loaded.size()) {
    for (std::size_t index = 0; index < byte.size(); ++index) {
      EXPECT_TRUE(SameBits(byte[index], byte_loaded[index]));
    }
  }
}

TEST_CASE(ReadImages_accepts_finite_negative_normalized_values) {
  const std::string directory =
      MakeScratchDirectory("read_images_normalized_negative");
  const std::vector<float> head = {-2.5F, -1.0F, -1.0e-8F, 3.5F, -0.0F};
  const std::vector<float> values = ImageValues(head, 0.125F);
  const std::vector<float> loaded = course::ReadImages(
      SaveImages(directory, "normalized.f32", values), UnitShape(1),
      course::InputMode::kNormalized);
  EXPECT_EQ(values.size(), loaded.size());
  if (values.size() == loaded.size()) {
    for (std::size_t index = 0; index < values.size(); ++index) {
      EXPECT_TRUE(SameBits(values[index], loaded[index]));
    }
  }
}

TEST_CASE(ReadImages_rejects_unit_and_byte_out_of_range_values) {
  const std::string directory =
      MakeScratchDirectory("read_images_out_of_range");

  EXPECT_THROWS(course::ReadImages(
      SaveImages(directory, "unit_low.f32", ImageValues({-0.0001F}, 0.5F)),
      UnitShape(1), course::InputMode::kUnit));
  EXPECT_THROWS(course::ReadImages(
      SaveImages(directory, "unit_high.f32", ImageValues({1.0001F}, 0.5F)),
      UnitShape(1), course::InputMode::kUnit));
  EXPECT_THROWS(course::ReadImages(
      SaveImages(directory, "byte_low.f32", ImageValues({-0.5F}, 128.0F)),
      UnitShape(1), course::InputMode::kByte));
  EXPECT_THROWS(course::ReadImages(
      SaveImages(directory, "byte_high.f32", ImageValues({255.5F}, 128.0F)),
      UnitShape(1), course::InputMode::kByte));
}

TEST_CASE(ReadImages_rejects_nan_and_infinity_in_every_mode) {
  const std::string directory =
      MakeScratchDirectory("read_images_nonfinite");
  const course::InputMode modes[] = {course::InputMode::kUnit,
                                     course::InputMode::kByte,
                                     course::InputMode::kNormalized};
  const float nonfinite[] = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()};
  int case_index = 0;
  for (course::InputMode mode : modes) {
    for (float value : nonfinite) {
      const std::string name =
          "nonfinite_i" + std::to_string(case_index++) + ".f32";
      EXPECT_THROWS(course::ReadImages(
          SaveImages(directory, name.c_str(), ImageValues({value}, 0.0F)),
          UnitShape(1), mode));
    }
  }
}

TEST_CASE(ReadData_decodes_known_little_endian_fixtures) {
  const std::string directory =
      MakeScratchDirectory("read_data_little_endian_fixtures");

  // Explicit little-endian binary32 patterns (not host memcpy): 1.0f is
  // 00 00 80 3f, -0.0f is 00 00 00 80, 0.1f is cd cc cc 3d, and 0.5f is
  // 00 00 00 3f. This proves the supported decode on the little-endian target.
  std::vector<unsigned char> image_bytes;
  const std::uint32_t patterns[] = {0x3f800000U, 0x80000000U, 0x3dcccccdU,
                                    0x3f000000U};
  for (std::uint32_t bits : patterns) {
    image_bytes.push_back(static_cast<unsigned char>(bits & 0xffu));
    image_bytes.push_back(static_cast<unsigned char>((bits >> 8) & 0xffu));
    image_bytes.push_back(static_cast<unsigned char>((bits >> 16) & 0xffu));
    image_bytes.push_back(static_cast<unsigned char>((bits >> 24) & 0xffu));
  }
  while (image_bytes.size() < kImageValueCount * sizeof(float)) {
    image_bytes.push_back(0x00u);
    image_bytes.push_back(0x00u);
    image_bytes.push_back(0x00u);
    image_bytes.push_back(0x3fu);
  }
  const std::string images_path = directory + "/images_le.f32";
  WriteBytes(images_path, image_bytes);

  const std::vector<float> images =
      course::ReadImages(images_path, UnitShape(1), course::InputMode::kUnit);
  EXPECT_EQ(kImageValueCount, images.size());
  if (images.size() == kImageValueCount) {
    EXPECT_TRUE(SameBits(1.0F, images[0]));
    EXPECT_TRUE(SameBits(-0.0F, images[1]));
    EXPECT_TRUE(SameBits(0.1F, images[2]));
    EXPECT_TRUE(SameBits(0.5F, images[3]));
    for (std::size_t index = 4; index < images.size(); ++index) {
      EXPECT_TRUE(SameBits(0.5F, images[index]));
    }
  }

  // int32 little-endian labels 0 and 9.
  const std::string labels_path = directory + "/labels_le.i32";
  WriteBytes(labels_path, std::vector<unsigned char>{0x00u, 0x00u, 0x00u,
                                                     0x00u, 0x09u, 0x00u,
                                                     0x00u, 0x00u});
  const std::vector<std::uint8_t> labels =
      course::ReadLabels(labels_path, UnitShape(2));
  EXPECT_EQ(std::vector<std::uint8_t>({0, 9}), labels);
}

TEST_CASE(ReadImages_rejects_short_and_long_files) {
  const std::string directory =
      MakeScratchDirectory("read_images_size_mismatch");
  const std::vector<float> short_values(kImageValueCount - 1, 0.25F);
  const std::vector<float> long_values(kImageValueCount + 1, 0.25F);
  EXPECT_THROWS(course::ReadImages(
      SaveImages(directory, "short.f32", short_values), UnitShape(1),
      course::InputMode::kUnit));
  EXPECT_THROWS(course::ReadImages(
      SaveImages(directory, "long.f32", long_values), UnitShape(1),
      course::InputMode::kUnit));
}

TEST_CASE(ReadImages_rejects_oversized_declarations_without_allocating) {
  const std::string directory =
      MakeScratchDirectory("read_images_oversized_declarations");
  const std::string path =
      SaveImages(directory, "tiny.f32", std::vector<float>(1, 0.25F));

  // Declared extents of roughly 3 GiB and 12 TiB: the exact file size must be
  // rejected before any result vector is requested.
  EXPECT_THROWS(
      course::ReadImages(path, UnitShape(1000000), course::InputMode::kUnit));
  EXPECT_THROWS(course::ReadImages(
      path, UnitShape(std::numeric_limits<std::uint32_t>::max()),
      course::InputMode::kUnit));
}

TEST_CASE(ReadImages_rejects_unsupported_and_zero_batch_shapes) {
  const std::string directory =
      MakeScratchDirectory("read_images_bad_shapes");
  const std::string path =
      SaveImages(directory, "images.f32", ImageValues({}, 0.25F));

  EXPECT_THROWS(course::ReadImages(
      path, course::Shape{0, 1, 28, 28, 10}, course::InputMode::kUnit));
  EXPECT_THROWS(course::ReadImages(
      path, course::Shape{1, 2, 28, 28, 10}, course::InputMode::kUnit));
  EXPECT_THROWS(course::ReadImages(
      path, course::Shape{1, 1, 27, 28, 10}, course::InputMode::kUnit));
  EXPECT_THROWS(course::ReadImages(
      path, course::Shape{1, 1, 28, 29, 10}, course::InputMode::kUnit));
  EXPECT_THROWS(course::ReadImages(
      path, course::Shape{1, 1, 28, 28, 9}, course::InputMode::kUnit));
}

TEST_CASE(ReadImages_reads_symlink_alias_without_mutating_files) {
  if (!COURSE_TEST_HAS_SYMLINKS) {
    return;
  }
  const std::string directory = MakeScratchDirectory("read_images_symlink");
  const std::vector<float> values = ImageValues({0.0F, 1.0F, 1.0F / 3.0F},
                                                0.25F);
  const std::string real_path = SaveImages(directory, "images.f32", values);
  const std::string alias_path = directory + "/images_alias.f32";
  CreateSymlink(real_path, alias_path);
  const std::vector<unsigned char> before = ReadBytes(real_path);

  const std::vector<float> loaded =
      course::ReadImages(alias_path, UnitShape(1), course::InputMode::kUnit);
  EXPECT_EQ(values.size(), loaded.size());
  if (values.size() == loaded.size()) {
    for (std::size_t index = 0; index < values.size(); ++index) {
      EXPECT_TRUE(SameBits(values[index], loaded[index]));
    }
  }
  EXPECT_TRUE(before == ReadBytes(real_path));
  EXPECT_TRUE(IsSymlink(alias_path));
}

// ---------------------------------------------------------------------------
// Raw label reading
// ---------------------------------------------------------------------------

TEST_CASE(ReadLabels_returns_lossless_uint8_values) {
  const std::string directory = MakeScratchDirectory("read_labels_valid");
  const std::string path =
      SaveLabels(directory, "labels.i32", std::vector<std::int32_t>{0, 9});
  const std::vector<unsigned char> before = ReadBytes(path);

  const std::vector<std::uint8_t> labels =
      course::ReadLabels(path, UnitShape(2));
  EXPECT_EQ(std::vector<std::uint8_t>({0, 9}), labels);
  EXPECT_TRUE(before == ReadBytes(path));
}

TEST_CASE(ReadLabels_rejects_out_of_range_values_before_narrowing) {
  const std::string directory =
      MakeScratchDirectory("read_labels_out_of_range");
  const std::int32_t invalid[] = {
      -1,
      10,
      11,
      256,
      std::numeric_limits<std::int32_t>::min(),
      std::numeric_limits<std::int32_t>::max(),
  };
  int case_index = 0;
  for (std::int32_t value : invalid) {
    const std::string name =
        "labels_i" + std::to_string(case_index++) + ".i32";
    EXPECT_THROWS(course::ReadLabels(
        SaveLabels(directory, name.c_str(),
                   std::vector<std::int32_t>{3, value}),
        UnitShape(2)));
  }
}

TEST_CASE(ReadLabels_rejects_short_and_long_files) {
  const std::string directory =
      MakeScratchDirectory("read_labels_size_mismatch");
  EXPECT_THROWS(course::ReadLabels(
      SaveLabels(directory, "short.i32", std::vector<std::int32_t>{3}),
      UnitShape(2)));
  EXPECT_THROWS(course::ReadLabels(
      SaveLabels(directory, "long.i32",
                 std::vector<std::int32_t>{3, 4, 5}),
      UnitShape(2)));
}

// ---------------------------------------------------------------------------
// Model bundle publication and loading
// ---------------------------------------------------------------------------

TEST_CASE(TestCrc32_matches_zlib_reference_vector) {
  const std::string text = "123456789";
  const std::vector<unsigned char> bytes(text.begin(), text.end());
  EXPECT_EQ(static_cast<std::uint32_t>(0xcbf43926u), Crc32(bytes));
}

TEST_CASE(CourseModel_round_trips_unit_bundle) {
  const std::string directory = MakeScratchDirectory("model_unit");
  const Checkpoint checkpoint = SampleCheckpoint();
  course::SaveCourseModel(directory, checkpoint, course::InputMode::kUnit);

  const std::vector<unsigned char> checkpoint_before =
      ReadBytes(directory + "/lenet.bin");
  const std::string metadata_before = ReadText(directory + "/course.meta");

  const course::CourseModel model = course::LoadCourseModel(directory);
  EXPECT_EQ(course::InputMode::kUnit, model.input_mode);
  ExpectEquivalentCheckpoints(checkpoint, model.checkpoint);
  EXPECT_TRUE(checkpoint_before == ReadBytes(directory + "/lenet.bin"));
  EXPECT_TRUE(metadata_before == ReadText(directory + "/course.meta"));
}

TEST_CASE(CourseModel_round_trips_normalized_bundle) {
  const std::string directory = MakeScratchDirectory("model_normalized");
  const Checkpoint checkpoint = SampleCheckpoint();
  course::SaveCourseModel(directory, checkpoint,
                          course::InputMode::kNormalized);

  const course::CourseModel model = course::LoadCourseModel(directory);
  EXPECT_EQ(course::InputMode::kNormalized, model.input_mode);
  ExpectEquivalentCheckpoints(checkpoint, model.checkpoint);
}

TEST_CASE(CourseModel_replaces_previous_valid_bundle) {
  const std::string directory = MakeScratchDirectory("model_replace");
  course::SaveCourseModel(directory, SampleCheckpoint(),
                          course::InputMode::kUnit);
  const Checkpoint replacement = MutatedCheckpoint();
  course::SaveCourseModel(directory, replacement, course::InputMode::kByte);

  const course::CourseModel model = course::LoadCourseModel(directory);
  EXPECT_EQ(course::InputMode::kByte, model.input_mode);
  ExpectEquivalentCheckpoints(replacement, model.checkpoint);
}

TEST_CASE(CourseModel_writes_exact_four_line_metadata) {
  const std::string directory = MakeScratchDirectory("model_metadata_bytes");
  course::SaveCourseModel(directory, SampleCheckpoint(),
                          course::InputMode::kUnit);
  const std::string metadata = ReadText(directory + "/course.meta");

  EXPECT_EQ(MetadataText("unit", BundleCrc(directory)), metadata);
  EXPECT_TRUE(metadata.size() <= 256);
  EXPECT_FALSE(metadata.empty());
  if (!metadata.empty()) {
    EXPECT_EQ('\n', metadata[metadata.size() - 1]);
  }
  std::size_t lines = 0;
  for (char character : metadata) {
    if (character == '\n') {
      ++lines;
    }
  }
  EXPECT_EQ(static_cast<std::size_t>(4), lines);
}

TEST_CASE(CourseModel_rejects_structural_metadata_corruption) {
  const std::string base = MetadataText("unit", "%CRC%");

  ExpectMetadataRejected("extra_field", base + "extra=1\n");
  ExpectMetadataRejected("unknown_field",
                         "course_lenet_v1\ninput_mode=unit\n"
                         "shape=1,28,28,10\nepochs=20\n"
                         "checkpoint_crc32=%CRC%\n");
  ExpectMetadataRejected("duplicate_mode",
                         "course_lenet_v1\ninput_mode=unit\n"
                         "input_mode=unit\nshape=1,28,28,10\n"
                         "checkpoint_crc32=%CRC%\n");
  ExpectMetadataRejected("wrong_version",
                         "course_lenet_v2\ninput_mode=unit\n"
                         "shape=1,28,28,10\ncheckpoint_crc32=%CRC%\n");
  ExpectMetadataRejected("missing_version",
                         "input_mode=unit\nshape=1,28,28,10\n"
                         "checkpoint_crc32=%CRC%\n");
  ExpectMetadataRejected("missing_shape",
                         "course_lenet_v1\ninput_mode=unit\n"
                         "checkpoint_crc32=%CRC%\n");
  ExpectMetadataRejected("reordered_fields",
                         "course_lenet_v1\nshape=1,28,28,10\n"
                         "input_mode=unit\ncheckpoint_crc32=%CRC%\n");
  ExpectMetadataRejected("unknown_mode",
                         "course_lenet_v1\ninput_mode=float\n"
                         "shape=1,28,28,10\ncheckpoint_crc32=%CRC%\n");
  ExpectMetadataRejected("bad_shape",
                         "course_lenet_v1\ninput_mode=unit\n"
                         "shape=1,28,28,9\ncheckpoint_crc32=%CRC%\n");
  ExpectMetadataRejected("short_crc",
                         "course_lenet_v1\ninput_mode=unit\n"
                         "shape=1,28,28,10\ncheckpoint_crc32=abc\n");
  ExpectMetadataRejected("empty", "");
  ExpectMetadataRejected("oversized", base + std::string(300, 'x') + "\n");

  std::string missing_final_newline = base;
  missing_final_newline.erase(missing_final_newline.size() - 1);
  ExpectMetadataRejected("missing_final_newline", missing_final_newline);

  std::string carriage_returns = base;
  ReplaceAll(&carriage_returns, "\n", "\r\n");
  ExpectMetadataRejected("carriage_returns", carriage_returns);
}

TEST_CASE(CourseModel_rejects_crc_mismatch_and_nonlowercase_crc) {
  const std::string directory = MakeScratchDirectory("model_crc_checks");
  course::SaveCourseModel(directory, SampleCheckpoint(),
                          course::InputMode::kUnit);
  const std::string crc = BundleCrc(directory);
  const std::string metadata_path = directory + "/course.meta";

  const std::string uppercase = ToUpper(crc);
  if (uppercase != crc) {
    WriteText(metadata_path, MetadataText("unit", uppercase));
    EXPECT_THROWS(course::LoadCourseModel(directory));
  } else {
    // An all-digit CRC has no case distinction; use a same-length non-CRC
    // token so the file is still rejected.
    WriteText(metadata_path, MetadataText("unit", "ABCDEF12"));
    EXPECT_THROWS(course::LoadCourseModel(directory));
  }

  std::string wrong = crc;
  wrong[7] = (wrong[7] == '0' ? '1' : '0');
  WriteText(metadata_path, MetadataText("unit", wrong));
  EXPECT_THROWS(course::LoadCourseModel(directory));
}

TEST_CASE(CourseModel_rejects_mixed_checkpoint_and_metadata_pair) {
  const std::string directory = MakeScratchDirectory("model_mixed_pair");
  course::SaveCourseModel(directory, SampleCheckpoint(),
                          course::InputMode::kUnit);

  // Simulate a new checkpoint published with the previous metadata: the CRC
  // now describes different bytes, so the pair must not be accepted.
  SaveCheckpoint(directory + "/lenet.bin", MutatedCheckpoint());
  EXPECT_THROWS(course::LoadCourseModel(directory));
}

TEST_CASE(CourseModel_failed_save_preserves_previous_bundle) {
  const std::string directory = MakeScratchDirectory("model_failed_save");
  const Checkpoint checkpoint = SampleCheckpoint();
  course::SaveCourseModel(directory, checkpoint, course::InputMode::kUnit);
  const std::vector<unsigned char> checkpoint_before =
      ReadBytes(directory + "/lenet.bin");
  const std::string metadata_before = ReadText(directory + "/course.meta");

  Checkpoint broken = SampleCheckpoint();
  broken.parameters[7].values[0] =
      std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROWS(course::SaveCourseModel(directory, broken,
                                        course::InputMode::kNormalized));

  EXPECT_TRUE(checkpoint_before == ReadBytes(directory + "/lenet.bin"));
  EXPECT_TRUE(metadata_before == ReadText(directory + "/course.meta"));
  const course::CourseModel model = course::LoadCourseModel(directory);
  EXPECT_EQ(course::InputMode::kUnit, model.input_mode);
  ExpectEquivalentCheckpoints(checkpoint, model.checkpoint);
}

TEST_CASE(CourseModel_preserves_unrelated_work_directory_entries) {
  const std::string directory =
      MakeScratchDirectory("model_unrelated_entries");
  WriteText(directory + "/notes.txt", "keep");
  MakeDirectory(directory + "/keepdir");
  WriteText(directory + "/keepdir/inside.txt", "keep inside");

  const Checkpoint checkpoint = SampleCheckpoint();
  course::SaveCourseModel(directory, checkpoint, course::InputMode::kUnit);
  const course::CourseModel model = course::LoadCourseModel(directory);
  EXPECT_EQ(course::InputMode::kUnit, model.input_mode);
  ExpectEquivalentCheckpoints(checkpoint, model.checkpoint);
  EXPECT_EQ(std::string("keep"), ReadText(directory + "/notes.txt"));
  EXPECT_EQ(std::string("keep inside"),
            ReadText(directory + "/keepdir/inside.txt"));
}

TEST_CASE(CourseModel_refuses_malformed_reserved_artifacts) {
  const std::string checkpoint_directory =
      MakeScratchDirectory("model_reserved_checkpoint_directory");
  MakeDirectory(checkpoint_directory + "/lenet.bin");
  EXPECT_THROWS(course::SaveCourseModel(checkpoint_directory,
                                        SampleCheckpoint(),
                                        course::InputMode::kUnit));
  EXPECT_TRUE(PathExists(checkpoint_directory + "/lenet.bin"));
  EXPECT_FALSE(PathExists(checkpoint_directory + "/course.meta"));

  const std::string metadata_directory =
      MakeScratchDirectory("model_reserved_metadata_directory");
  MakeDirectory(metadata_directory + "/course.meta");
  EXPECT_THROWS(course::SaveCourseModel(metadata_directory, SampleCheckpoint(),
                                        course::InputMode::kUnit));
  EXPECT_TRUE(PathExists(metadata_directory + "/course.meta"));
  EXPECT_FALSE(PathExists(metadata_directory + "/lenet.bin"));
}

TEST_CASE(CourseModel_rejects_lone_regular_reserved_artifacts) {
  const std::string checkpoint_directory =
      MakeScratchDirectory("model_lone_junk_checkpoint");
  const std::vector<unsigned char> junk_checkpoint{'j', 'u', 'n', 'k'};
  WriteBytes(checkpoint_directory + "/lenet.bin", junk_checkpoint);
  WriteText(checkpoint_directory + "/notes.txt", "keep");
  EXPECT_THROWS_AS(course::SaveCourseModel(checkpoint_directory,
                                           SampleCheckpoint(),
                                           course::InputMode::kUnit),
                   std::runtime_error);
  EXPECT_TRUE(junk_checkpoint ==
              ReadBytes(checkpoint_directory + "/lenet.bin"));
  EXPECT_FALSE(PathExists(checkpoint_directory + "/course.meta"));
  EXPECT_EQ(std::string("keep"),
            ReadText(checkpoint_directory + "/notes.txt"));

  const std::string metadata_directory =
      MakeScratchDirectory("model_lone_junk_metadata");
  WriteText(metadata_directory + "/course.meta", "not metadata\n");
  EXPECT_THROWS_AS(course::SaveCourseModel(metadata_directory,
                                           SampleCheckpoint(),
                                           course::InputMode::kUnit),
                   std::runtime_error);
  EXPECT_EQ(std::string("not metadata\n"),
            ReadText(metadata_directory + "/course.meta"));
  EXPECT_FALSE(PathExists(metadata_directory + "/lenet.bin"));
}

TEST_CASE(CourseModel_rejects_junk_regular_reserved_pairs) {
  const std::string junk_pair_directory =
      MakeScratchDirectory("model_junk_regular_pair");
  WriteText(junk_pair_directory + "/lenet.bin", "junk checkpoint");
  WriteText(junk_pair_directory + "/course.meta", "junk metadata\n");
  EXPECT_THROWS_AS(course::SaveCourseModel(junk_pair_directory,
                                           SampleCheckpoint(),
                                           course::InputMode::kUnit),
                   std::runtime_error);
  EXPECT_EQ(std::string("junk checkpoint"),
            ReadText(junk_pair_directory + "/lenet.bin"));
  EXPECT_EQ(std::string("junk metadata\n"),
            ReadText(junk_pair_directory + "/course.meta"));

  // Metadata whose CRC matches the junk checkpoint bytes: the pair must still
  // be rejected because the reserved checkpoint itself is not a valid model.
  const std::string matching_crc_directory =
      MakeScratchDirectory("model_junk_checkpoint_matching_crc");
  const std::vector<unsigned char> junk_bytes{'n', 'o', 't', 'c', 'k', 'p',
                                              't'};
  WriteBytes(matching_crc_directory + "/lenet.bin", junk_bytes);
  WriteText(matching_crc_directory + "/course.meta",
            MetadataText("unit", Hex8(Crc32(junk_bytes))));
  EXPECT_THROWS_AS(course::SaveCourseModel(matching_crc_directory,
                                           SampleCheckpoint(),
                                           course::InputMode::kUnit),
                   std::runtime_error);
  EXPECT_TRUE(junk_bytes ==
              ReadBytes(matching_crc_directory + "/lenet.bin"));

  const std::string junk_metadata_directory =
      MakeScratchDirectory("model_valid_checkpoint_junk_metadata");
  SaveCheckpoint(junk_metadata_directory + "/lenet.bin", SampleCheckpoint());
  const std::vector<unsigned char> checkpoint_before =
      ReadBytes(junk_metadata_directory + "/lenet.bin");
  WriteText(junk_metadata_directory + "/course.meta", "junk metadata\n");
  EXPECT_THROWS_AS(course::SaveCourseModel(junk_metadata_directory,
                                           SampleCheckpoint(),
                                           course::InputMode::kUnit),
                   std::runtime_error);
  EXPECT_TRUE(checkpoint_before ==
              ReadBytes(junk_metadata_directory + "/lenet.bin"));
  EXPECT_EQ(std::string("junk metadata\n"),
            ReadText(junk_metadata_directory + "/course.meta"));
}

TEST_CASE(CourseModel_refuses_symlink_checkpoint_destination) {
  if (!COURSE_TEST_HAS_SYMLINKS) {
    return;
  }
  const std::string target_directory =
      MakeScratchDirectory("model_symlink_target");
  const std::string target_path = target_directory + "/checkpoint.bin";
  SaveCheckpoint(target_path, SampleCheckpoint());
  const std::vector<unsigned char> target_before = ReadBytes(target_path);

  const std::string directory = MakeScratchDirectory("model_symlink_bundle");
  const std::string alias_path = directory + "/lenet.bin";
  CreateSymlink(target_path, alias_path);

  EXPECT_THROWS(course::SaveCourseModel(directory, MutatedCheckpoint(),
                                        course::InputMode::kUnit));
  EXPECT_TRUE(target_before == ReadBytes(target_path));
  EXPECT_TRUE(IsSymlink(alias_path));
}

TEST_CASE(CourseModel_refuses_hardlink_checkpoint_destination) {
  if (!COURSE_TEST_HAS_SYMLINKS) {
    return;
  }
  const std::string target_directory =
      MakeScratchDirectory("model_hardlink_target");
  const std::string target_path = target_directory + "/checkpoint.bin";
  SaveCheckpoint(target_path, SampleCheckpoint());
  const std::vector<unsigned char> target_before = ReadBytes(target_path);

  const std::string directory = MakeScratchDirectory("model_hardlink_bundle");
  const std::string alias_path = directory + "/lenet.bin";
  CreateHardLink(target_path, alias_path);

  // Writing through the alias would corrupt the caller's checkpoint bytes.
  EXPECT_THROWS(course::SaveCourseModel(directory, MutatedCheckpoint(),
                                        course::InputMode::kUnit));
  EXPECT_TRUE(target_before == ReadBytes(target_path));
}

TEST_CASE(CourseModel_rejects_missing_artifacts) {
  const std::string empty = MakeScratchDirectory("model_missing_empty");
  EXPECT_THROWS(course::LoadCourseModel(empty));
  EXPECT_THROWS(course::LoadCourseModel(ScratchRoot() + "/model_absent"));

  const std::string checkpoint_only =
      MakeScratchDirectory("model_missing_metadata");
  SaveCheckpoint(checkpoint_only + "/lenet.bin", SampleCheckpoint());
  EXPECT_THROWS(course::LoadCourseModel(checkpoint_only));

  const std::string metadata_only =
      MakeScratchDirectory("model_missing_checkpoint");
  WriteText(metadata_only + "/course.meta", MetadataText("unit", "00000000"));
  EXPECT_THROWS(course::LoadCourseModel(metadata_only));
}

}  // namespace
