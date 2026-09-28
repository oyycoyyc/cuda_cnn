#include "checkpoint.h"
#include "test_harness.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::size_t kHeaderSize = 40;
const std::size_t kTensorMetadataSize = 60;
const std::size_t kCheckpointSize = 178344;
const std::array<std::size_t, 10> kMetadataOffsets{{
    40, 700, 784, 10444, 10568, 133508, 134048, 174428, 174824,
    178244,
}};
const std::array<std::size_t, 10> kPayloadOffsets{{
    100, 760, 844, 10504, 10628, 133568, 134108, 174488, 174884,
    178304,
}};
const std::array<const char*, 10> kNames{{
    "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
    "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight",
    "fc3.bias",
}};
const std::array<std::uint32_t, 10> kRanks{{4, 1, 4, 1, 2, 1, 2, 1, 2, 1}};
const std::array<std::array<std::uint32_t, 4>, 10> kDimensions{{
    {{6, 1, 5, 5}}, {{6, 1, 1, 1}}, {{16, 6, 5, 5}},
    {{16, 1, 1, 1}}, {{120, 256, 1, 1}}, {{120, 1, 1, 1}},
    {{84, 120, 1, 1}}, {{84, 1, 1, 1}}, {{10, 84, 1, 1}},
    {{10, 1, 1, 1}},
}};
const std::array<std::uint64_t, 10> kElementCounts{{
    150, 6, 2400, 16, 30720, 120, 10080, 84, 840, 10,
}};

std::vector<unsigned char> ReadBytes(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("test could not read " + path);
  }
  return std::vector<unsigned char>(std::istreambuf_iterator<char>(input),
                                    std::istreambuf_iterator<char>());
}

void WriteBytes(const std::string& path,
                const std::vector<unsigned char>& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    throw std::runtime_error("test could not write " + path);
  }
}

std::uint32_t ReadU32(const std::vector<unsigned char>& bytes,
                      std::size_t offset) {
  return static_cast<std::uint32_t>(bytes[offset]) |
         (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
         (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

std::uint64_t ReadU64(const std::vector<unsigned char>& bytes,
                      std::size_t offset) {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < 8; ++index) {
    value |= static_cast<std::uint64_t>(bytes[offset + index]) << (index * 8);
  }
  return value;
}

void PutU32(std::vector<unsigned char>* bytes, std::size_t offset,
            std::uint32_t value) {
  for (std::size_t index = 0; index < 4; ++index) {
    (*bytes)[offset + index] =
        static_cast<unsigned char>(value >> (index * 8));
  }
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

Checkpoint ValidCheckpoint() {
  Checkpoint checkpoint{{7, 0.75F, 0.1307F, 0.3081F},
                        CreateLenetParameters()};
  std::uint32_t ordinal = 1;
  for (ParameterTensor& tensor : checkpoint.parameters) {
    for (float& value : tensor.values) {
      value = static_cast<float>(static_cast<int>(ordinal % 257) - 128) /
              64.0F;
      ++ordinal;
    }
  }
  checkpoint.parameters[0].values[0] = -0.0F;
  checkpoint.parameters[0].values[1] =
      std::numeric_limits<float>::denorm_min();
  return checkpoint;
}

void ExpectFloatBitsEqual(float expected, float actual) {
  EXPECT_EQ(FloatBits(expected), FloatBits(actual));
}

void ExpectCheckpointsBitEqual(const Checkpoint& expected,
                               const Checkpoint& actual) {
  EXPECT_EQ(expected.metadata.best_epoch, actual.metadata.best_epoch);
  ExpectFloatBitsEqual(expected.metadata.validation_accuracy,
                       actual.metadata.validation_accuracy);
  ExpectFloatBitsEqual(expected.metadata.normalization_mean,
                       actual.metadata.normalization_mean);
  ExpectFloatBitsEqual(expected.metadata.normalization_stddev,
                       actual.metadata.normalization_stddev);
  EXPECT_EQ(expected.parameters.size(), actual.parameters.size());
  for (std::size_t tensor = 0; tensor < expected.parameters.size(); ++tensor) {
    EXPECT_EQ(expected.parameters[tensor].name, actual.parameters[tensor].name);
    EXPECT_EQ(expected.parameters[tensor].rank, actual.parameters[tensor].rank);
    EXPECT_EQ(expected.parameters[tensor].dimensions,
              actual.parameters[tensor].dimensions);
    EXPECT_EQ(expected.parameters[tensor].values.size(),
              actual.parameters[tensor].values.size());
    for (std::size_t value = 0;
         value < expected.parameters[tensor].values.size(); ++value) {
      ExpectFloatBitsEqual(expected.parameters[tensor].values[value],
                           actual.parameters[tensor].values[value]);
    }
  }
}

void ExpectLoadRejected(const std::vector<unsigned char>& bytes,
                        const std::string& case_name,
                        const std::string& invariant) {
  const std::string path = "build/checkpoint_reject_" + case_name + ".bin";
  WriteBytes(path, bytes);
  try {
    LoadCheckpoint(path);
    std::remove(path.c_str());
    test_harness::Fail(__FILE__, __LINE__, "malformed checkpoint loaded");
  } catch (const std::exception& error) {
    const std::string message(error.what());
    std::remove(path.c_str());
    EXPECT_TRUE(message.find(path) != std::string::npos);
    EXPECT_TRUE(message.find(invariant) != std::string::npos);
  }
}

void ExpectFailedSavePreservesDestination(const Checkpoint& checkpoint,
                                          const std::string& case_name) {
  const std::string path = "build/checkpoint_preserve_" + case_name + ".bin";
  const std::vector<unsigned char> sentinel{{0x53, 0x41, 0x46, 0x45}};
  WriteBytes(path, sentinel);
  bool threw = false;
  try {
    SaveCheckpoint(path, checkpoint);
  } catch (const std::exception& error) {
    threw = true;
    EXPECT_TRUE(std::string(error.what()).find(path) != std::string::npos);
  }
  EXPECT_TRUE(threw);
  EXPECT_EQ(sentinel, ReadBytes(path));
  std::remove(path.c_str());
}

}  // namespace

TEST_CASE(checkpoint_round_trip_is_bit_identical_and_layout_is_exact) {
  const std::string first_path = "build/checkpoint_round_trip_a.bin";
  const std::string second_path = "build/checkpoint_round_trip_b.bin";
  WriteBytes(first_path, std::vector<unsigned char>{{0x4f, 0x4c, 0x44}});
  const Checkpoint expected = ValidCheckpoint();
  SaveCheckpoint(first_path, expected);

  const std::vector<unsigned char> bytes = ReadBytes(first_path);
  EXPECT_EQ(kCheckpointSize, bytes.size());
  const std::array<unsigned char, 8> magic{{'L', 'N', 'E', 'T', 'C', '0', '1',
                                            0}};
  EXPECT_TRUE(std::equal(magic.begin(), magic.end(), bytes.begin()));
  EXPECT_EQ(UINT32_C(1), ReadU32(bytes, 8));
  EXPECT_EQ(UINT32_C(1), ReadU32(bytes, 12));
  EXPECT_EQ(UINT32_C(10), ReadU32(bytes, 16));
  EXPECT_EQ(UINT32_C(7), ReadU32(bytes, 20));
  EXPECT_EQ(FloatBits(0.75F), ReadU32(bytes, 24));
  EXPECT_EQ(FloatBits(0.1307F), ReadU32(bytes, 28));
  EXPECT_EQ(FloatBits(0.3081F), ReadU32(bytes, 32));
  EXPECT_EQ(UINT32_C(0), ReadU32(bytes, 36));
  EXPECT_EQ(kHeaderSize, kMetadataOffsets.front());

  for (std::size_t tensor = 0; tensor < kMetadataOffsets.size(); ++tensor) {
    const std::size_t metadata = kMetadataOffsets[tensor];
    const std::size_t name_size = std::strlen(kNames[tensor]);
    for (std::size_t index = 0; index < 32; ++index) {
      const unsigned char expected_byte = index < name_size
                                              ? kNames[tensor][index]
                                              : static_cast<unsigned char>(0);
      EXPECT_EQ(expected_byte, bytes[metadata + index]);
    }
    EXPECT_EQ(kRanks[tensor], ReadU32(bytes, metadata + 32));
    for (std::size_t dimension = 0; dimension < 4; ++dimension) {
      EXPECT_EQ(kDimensions[tensor][dimension],
                ReadU32(bytes, metadata + 36 + dimension * 4));
    }
    EXPECT_EQ(kElementCounts[tensor], ReadU64(bytes, metadata + 52));
    EXPECT_EQ(metadata + kTensorMetadataSize, kPayloadOffsets[tensor]);
    EXPECT_EQ(FloatBits(expected.parameters[tensor].values.front()),
              ReadU32(bytes, kPayloadOffsets[tensor]));
  }

  const Checkpoint actual = LoadCheckpoint(first_path);
  ExpectCheckpointsBitEqual(expected, actual);
  SaveCheckpoint(second_path, actual);
  EXPECT_EQ(bytes, ReadBytes(second_path));
  std::remove(first_path.c_str());
  std::remove(second_path.c_str());
}

TEST_CASE(checkpoint_loader_rejects_every_header_invariant) {
  const std::string valid_path = "build/checkpoint_valid_header.bin";
  SaveCheckpoint(valid_path, ValidCheckpoint());
  const std::vector<unsigned char> valid = ReadBytes(valid_path);
  std::remove(valid_path.c_str());

  std::vector<unsigned char> changed = valid;
  changed[0] = 'X';
  ExpectLoadRejected(changed, "magic", "magic");
  changed = valid;
  PutU32(&changed, 8, 2);
  ExpectLoadRejected(changed, "version", "version");
  changed = valid;
  PutU32(&changed, 12, 2);
  ExpectLoadRejected(changed, "architecture", "architecture");
  changed = valid;
  PutU32(&changed, 16, 9);
  ExpectLoadRejected(changed, "tensor_count", "tensor count");
  changed = valid;
  PutU32(&changed, 20, 0);
  ExpectLoadRejected(changed, "epoch", "epoch");
  changed = valid;
  PutU32(&changed, 24, FloatBits(-0.01F));
  ExpectLoadRejected(changed, "accuracy_negative", "accuracy");
  changed = valid;
  PutU32(&changed, 24, FloatBits(1.01F));
  ExpectLoadRejected(changed, "accuracy_high", "accuracy");
  changed = valid;
  PutU32(&changed, 24, UINT32_C(0x7fc00000));
  ExpectLoadRejected(changed, "accuracy_nan", "accuracy");
  changed = valid;
  PutU32(&changed, 28, FloatBits(0.1308F));
  ExpectLoadRejected(changed, "mean", "mean");
  changed = valid;
  PutU32(&changed, 32, FloatBits(0.3082F));
  ExpectLoadRejected(changed, "stddev", "standard deviation");
  changed = valid;
  PutU32(&changed, 36, 1);
  ExpectLoadRejected(changed, "reserved", "reserved");
}

TEST_CASE(checkpoint_loader_rejects_names_order_duplicates_and_shapes) {
  const std::string valid_path = "build/checkpoint_valid_schema.bin";
  SaveCheckpoint(valid_path, ValidCheckpoint());
  const std::vector<unsigned char> valid = ReadBytes(valid_path);
  std::remove(valid_path.c_str());

  std::vector<unsigned char> changed = valid;
  changed[kMetadataOffsets[0]] = 'x';
  ExpectLoadRejected(changed, "unknown_name", "name");
  changed = valid;
  std::copy_n(changed.begin() + kMetadataOffsets[0], 32,
              changed.begin() + kMetadataOffsets[1]);
  ExpectLoadRejected(changed, "duplicate_name", "name");
  changed = valid;
  std::array<unsigned char, 32> first_name;
  std::copy_n(changed.begin() + kMetadataOffsets[0], 32, first_name.begin());
  std::copy_n(changed.begin() + kMetadataOffsets[1], 32,
              changed.begin() + kMetadataOffsets[0]);
  std::copy(first_name.begin(), first_name.end(),
            changed.begin() + kMetadataOffsets[1]);
  ExpectLoadRejected(changed, "name_order", "name");
  changed = valid;
  PutU32(&changed, kMetadataOffsets[0] + 32, 3);
  ExpectLoadRejected(changed, "rank", "rank");
  changed = valid;
  PutU32(&changed, kMetadataOffsets[0] + 36, 7);
  ExpectLoadRejected(changed, "dimension", "dimensions");
  changed = valid;
  PutU32(&changed, kMetadataOffsets[1] + 40, 2);
  ExpectLoadRejected(changed, "unused_dimension", "dimensions");
  changed = valid;
  changed[kMetadataOffsets[0] + 12] = 1;
  ExpectLoadRejected(changed, "name_padding", "name");
  changed = valid;
  PutU32(&changed, kMetadataOffsets[0] + 52, 149);
  ExpectLoadRejected(changed, "element_count", "element count");
}

TEST_CASE(checkpoint_loader_rejects_truncation_trailing_and_nonfinite_values) {
  const std::string valid_path = "build/checkpoint_valid_payload.bin";
  SaveCheckpoint(valid_path, ValidCheckpoint());
  const std::vector<unsigned char> valid = ReadBytes(valid_path);
  std::remove(valid_path.c_str());

  std::vector<unsigned char> changed(valid.begin(), valid.begin() + 7);
  ExpectLoadRejected(changed, "truncated_header", "header");
  changed.assign(valid.begin(), valid.begin() + 99);
  ExpectLoadRejected(changed, "truncated_metadata", "metadata");
  changed.assign(valid.begin(), valid.end() - 1);
  ExpectLoadRejected(changed, "truncated_payload", "parameter value");
  changed = valid;
  changed.push_back(0);
  ExpectLoadRejected(changed, "trailing", "trailing");
  changed = valid;
  PutU32(&changed, kPayloadOffsets[0], UINT32_C(0x7f800000));
  ExpectLoadRejected(changed, "infinite_parameter", "finite");
  changed = valid;
  PutU32(&changed, kPayloadOffsets[9], UINT32_C(0x7fc00000));
  ExpectLoadRejected(changed, "nan_parameter", "finite");
}

TEST_CASE(checkpoint_save_validates_everything_before_replacing_destination) {
  Checkpoint invalid = ValidCheckpoint();
  invalid.metadata.best_epoch = 0;
  ExpectFailedSavePreservesDestination(invalid, "metadata");

  invalid = ValidCheckpoint();
  invalid.parameters[3].rank = 2;
  ExpectFailedSavePreservesDestination(invalid, "schema");

  invalid = ValidCheckpoint();
  invalid.parameters.back().values.back() =
      std::numeric_limits<float>::infinity();
  ExpectFailedSavePreservesDestination(invalid, "nonfinite");
}
