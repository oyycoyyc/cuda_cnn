// MNISTC1 dataset fixtures, little-endian binary IO, malformed/truncated/trailing
// input rejection, label range checks, and official-count enforcement.
#include "binary_io.h"
#include "dataset.h"
#include "test_harness.h"

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Fixed header size and one 28x28 image extent for the fixture builder.
const std::size_t kHeaderSize = 24;
const std::size_t kImageSize = 28 * 28;

// Removes a fixture path when the enclosing test scope exits.
class ScopedFile {
 public:
  explicit ScopedFile(const std::string& path) : path_(path) {}
  ~ScopedFile() { std::remove(path_.c_str()); }

 private:
  std::string path_;
};

// Appends one little-endian 32-bit header field.
void AppendU32(std::vector<std::uint8_t>* bytes, std::uint32_t value) {
  bytes->push_back(static_cast<std::uint8_t>(value));
  bytes->push_back(static_cast<std::uint8_t>(value >> 8));
  bytes->push_back(static_cast<std::uint8_t>(value >> 16));
  bytes->push_back(static_cast<std::uint8_t>(value >> 24));
}

// Overwrites one little-endian header field to synthesize malformed inputs.
void SetU32(std::vector<std::uint8_t>* bytes, std::size_t offset,
            std::uint32_t value) {
  for (std::size_t byte = 0; byte < 4; ++byte) {
    (*bytes)[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8));
  }
}

// Builds a valid three-sample MNISTC1 image with distinct label values.
std::vector<std::uint8_t> ThreeRowFixture() {
  const char magic[8] = {'M', 'N', 'I', 'S', 'T', 'C', '1', '\0'};
  std::vector<std::uint8_t> bytes(magic, magic + 8);
  AppendU32(&bytes, 1);
  AppendU32(&bytes, 3);
  AppendU32(&bytes, 28);
  AppendU32(&bytes, 28);
  bytes.insert(bytes.end(), kImageSize, 0);
  bytes.insert(bytes.end(), kImageSize, 127);
  for (std::size_t index = 0; index < kImageSize; ++index) {
    bytes.push_back(static_cast<std::uint8_t>(index % 256));
  }
  bytes.push_back(0);
  bytes.push_back(5);
  bytes.push_back(9);
  return bytes;
}

// Writes fixture bytes and fails the test if the file cannot be created.
void WriteFile(const std::string& path,
               const std::vector<std::uint8_t>& bytes) {
  std::ofstream output(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!output || (!bytes.empty() &&
                  !output.write(reinterpret_cast<const char*>(bytes.data()),
                                bytes.size()))) {
    throw std::runtime_error("could not write test fixture " + path);
  }
}

// Requires the callable to throw with both the path and the invariant named.
template <typename Function>
void ExpectError(Function function, const std::string& path,
                 const std::string& invariant) {
  bool threw = false;
  try {
    function();
  } catch (const std::exception& error) {
    threw = true;
    const std::string message(error.what());
    EXPECT_TRUE(message.find(path) != std::string::npos);
    EXPECT_TRUE(message.find(invariant) != std::string::npos);
  }
  EXPECT_TRUE(threw);
}

// Writes a named malformed fixture and requires the loader to reject it.
void ExpectLoadError(const std::string& name,
                     const std::vector<std::uint8_t>& bytes,
                     const std::string& invariant) {
  const std::string path = "build/" + name + ".bin";
  ScopedFile cleanup(path);
  WriteFile(path, bytes);
  ExpectError([&path]() { LoadMnistDataset(path); }, path, invariant);
}

}  // namespace

// Happy path: counts, dimensions, pixels, labels, and image pointer arithmetic.
TEST_CASE(loads_exact_three_row_fixture) {
  const std::string path = "build/dataset-valid.bin";
  ScopedFile cleanup(path);
  WriteFile(path, ThreeRowFixture());

  const MnistDataset dataset = LoadMnistDataset(path);

  EXPECT_EQ(std::uint32_t{3}, dataset.sample_count);
  EXPECT_EQ(std::uint32_t{28}, dataset.rows);
  EXPECT_EQ(std::uint32_t{28}, dataset.columns);
  EXPECT_EQ(std::size_t{3 * 784}, dataset.images.size());
  EXPECT_EQ(std::size_t{3}, dataset.labels.size());
  EXPECT_EQ(std::uint8_t{0}, dataset.labels[0]);
  EXPECT_EQ(std::uint8_t{5}, dataset.labels[1]);
  EXPECT_EQ(std::uint8_t{9}, dataset.labels[2]);
  EXPECT_TRUE(dataset.Image(1) == dataset.images.data() + 784);
}

// Little-endian scalar writers produce the exact expected byte sequence.
TEST_CASE(binary_io_round_trips_little_endian_scalars) {
  const std::string path = "memory fixture";
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  binary_io::WriteU32LE(stream, 0x78563412U, path, "u32 is writable");
  binary_io::WriteU64LE(stream, UINT64_C(0x0102030405060708), path,
                        "u64 is writable");
  binary_io::WriteF32LE(stream, 1.0F, path, "f32 is writable");
  const std::string expected(
      "\x12\x34\x56\x78\x08\x07\x06\x05\x04\x03\x02\x01"
      "\x00\x00\x80\x3f",
      16);
  EXPECT_EQ(expected, stream.str());

  stream.seekg(0);
  EXPECT_EQ(std::uint32_t{0x78563412U},
            binary_io::ReadU32LE(stream, path, "u32 is readable"));
  EXPECT_EQ(UINT64_C(0x0102030405060708),
            binary_io::ReadU64LE(stream, path, "u64 is readable"));
  EXPECT_EQ(1.0F, binary_io::ReadF32LE(stream, path, "f32 is readable"));
}

TEST_CASE(binary_io_errors_name_path_and_invariant) {
  const std::string path = "empty memory fixture";
  std::istringstream input(std::string(), std::ios::binary);
  ExpectError(
      [&input, &path]() {
        binary_io::ReadU32LE(input, path, "u32 requires four bytes");
      },
      path, "u32 requires four bytes");
}

// Header validation: magic and format version must match exactly.
TEST_CASE(rejects_bad_magic_and_version) {
  std::vector<std::uint8_t> bytes = ThreeRowFixture();
  bytes[0] = 'X';
  ExpectLoadError("dataset-bad-magic", bytes, "magic must equal MNISTC1");

  bytes = ThreeRowFixture();
  SetU32(&bytes, 8, 2);
  ExpectLoadError("dataset-bad-version", bytes, "version must equal 1");
}

TEST_CASE(rejects_zero_count_and_non_28_dimensions) {
  std::vector<std::uint8_t> bytes = ThreeRowFixture();
  SetU32(&bytes, 12, 0);
  ExpectLoadError("dataset-zero-count", bytes,
                  "sample count must be nonzero");

  bytes = ThreeRowFixture();
  SetU32(&bytes, 16, 27);
  ExpectLoadError("dataset-bad-rows", bytes, "rows must equal 28");

  bytes = ThreeRowFixture();
  SetU32(&bytes, 20, 29);
  ExpectLoadError("dataset-bad-columns", bytes, "columns must equal 28");
}

// An implausibly large sample count rejected by checked size and exact-size
// validation.
TEST_CASE(rejects_count_whose_32_bit_size_calculation_wraps) {
  std::vector<std::uint8_t> bytes = ThreeRowFixture();
  bytes.resize(kHeaderSize + 1);
  SetU32(&bytes, 12, UINT32_C(1006718449));
  ExpectLoadError("dataset-wrapped-count", bytes,
                  "file size must match header count");
}

// Truncated header, image, or label sections fail the exact-size check.
TEST_CASE(rejects_truncated_header_images_and_labels) {
  const std::vector<std::uint8_t> complete = ThreeRowFixture();
  ExpectLoadError("dataset-short-header",
                  std::vector<std::uint8_t>(complete.begin(),
                                            complete.begin() + 10),
                  "header version");
  ExpectLoadError("dataset-short-images",
                  std::vector<std::uint8_t>(complete.begin(),
                                            complete.begin() + kHeaderSize +
                                                3 * kImageSize - 1),
                  "file size must match header count");
  ExpectLoadError("dataset-short-labels",
                  std::vector<std::uint8_t>(complete.begin(),
                                            complete.end() - 1),
                  "file size must match header count");
}

// Trailing bytes after the declared payload are rejected.
TEST_CASE(rejects_trailing_bytes) {
  std::vector<std::uint8_t> bytes = ThreeRowFixture();
  bytes.push_back(0);
  ExpectLoadError("dataset-trailing-byte", bytes,
                  "file size must match header count");
}

// Labels must be decimal digits; values 10 and 255 are rejected.
TEST_CASE(rejects_labels_outside_decimal_digit_range) {
  for (const std::uint8_t label : {std::uint8_t{10}, std::uint8_t{255}}) {
    std::vector<std::uint8_t> bytes = ThreeRowFixture();
    bytes[kHeaderSize + 3 * kImageSize] = label;
    ExpectLoadError("dataset-bad-label-" + std::to_string(label), bytes,
                    "label values must be in [0, 9]");
  }
}

TEST_CASE(rejects_unreadable_paths) {
  const std::string path = "build/dataset-does-not-exist.bin";
  std::remove(path.c_str());
  ExpectError([&path]() { LoadMnistDataset(path); }, path,
              "file must be readable");
}

TEST_CASE(rejects_out_of_range_image_access) {
  const std::string path = "build/dataset-image-range.bin";
  ScopedFile cleanup(path);
  WriteFile(path, ThreeRowFixture());
  const MnistDataset dataset = LoadMnistDataset(path);
  EXPECT_THROW_CONTAINS(dataset.Image(3),
                        "image index must be below sample_count");
}

// RequireDatasetCount enforces 60000 unless the bypass flag is set.
TEST_CASE(enforces_official_count_unless_explicitly_bypassed) {
  MnistDataset dataset{};
  dataset.sample_count = 3;
  EXPECT_THROW_CONTAINS(
      RequireDatasetCount(dataset, 60000, false, "training dataset"),
      "training dataset count must equal 60000");
  RequireDatasetCount(dataset, 60000, true, "training dataset");
  RequireDatasetCount(dataset, 3, false, "training dataset");
}
