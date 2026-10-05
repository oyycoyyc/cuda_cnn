#include "dataset.h"

#include "binary_io.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>

namespace {

// MNISTC1 layout: an 8-byte magic, a 4-byte version, three little-endian u32
// header fields, then count * (784 image bytes + 1 label byte) of payload.
const std::uint64_t kHeaderSize = 24;
const std::uint64_t kImageSize = 28 * 28;
const std::uint64_t kBytesPerSample = kImageSize + 1;

// Reports a dataset invariant failure prefixed with the offending path.
[[noreturn]] void DatasetError(const std::string& path,
                               const std::string& invariant) {
  throw std::runtime_error(path + ": " + invariant);
}

}  // namespace

// Returns a pointer to one 28x28 image; index must be below sample_count.
const std::uint8_t* MnistDataset::Image(std::uint32_t index) const {
  if (index >= sample_count) {
    throw std::out_of_range("image index must be below sample_count");
  }
  const std::size_t image_size =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
  return images.data() + static_cast<std::size_t>(index) * image_size;
}

// Loads a complete dataset, validating magic, version, shape, checked size
// arithmetic, exact file size, label range, and absence of trailing bytes.
MnistDataset LoadMnistDataset(const std::string& path) {
  std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
  if (!input.is_open()) {
    DatasetError(path, "file must be readable");
  }

  const std::streamoff end = input.tellg();
  if (end < 0) {
    DatasetError(path, "file size must be readable");
  }
  const std::uint64_t file_size = static_cast<std::uint64_t>(end);
  input.seekg(0, std::ios::beg);
  if (!input) {
    DatasetError(path, "file start must be seekable");
  }

  // The 24-byte header opens with an 8-byte magic identifying MNISTC1.
  const char expected_magic[8] = {'M', 'N', 'I', 'S', 'T', 'C', '1', '\0'};
  char magic[8];
  binary_io::ReadExact(input, magic, sizeof(magic), path,
                       "header magic requires 8 bytes");
  for (std::size_t index = 0; index < sizeof(magic); ++index) {
    if (magic[index] != expected_magic[index]) {
      DatasetError(path, "magic must equal MNISTC1\\0");
    }
  }

  // Version, count, rows, and columns follow as little-endian u32 fields.
  const std::uint32_t version = binary_io::ReadU32LE(
      input, path, "header version requires four bytes");
  if (version != 1) {
    DatasetError(path, "version must equal 1");
  }

  MnistDataset dataset{};
  dataset.sample_count = binary_io::ReadU32LE(
      input, path, "header sample count requires four bytes");
  dataset.rows = binary_io::ReadU32LE(
      input, path, "header rows requires four bytes");
  dataset.columns = binary_io::ReadU32LE(
      input, path, "header columns requires four bytes");

  if (dataset.sample_count == 0) {
    DatasetError(path, "sample count must be nonzero");
  }
  if (dataset.rows != 28) {
    DatasetError(path, "rows must equal 28");
  }
  if (dataset.columns != 28) {
    DatasetError(path, "columns must equal 28");
  }

  // Reject counts whose expected size would overflow 64-bit arithmetic.
  if (dataset.sample_count >
      (std::numeric_limits<std::uint64_t>::max() - kHeaderSize) /
          kBytesPerSample) {
    DatasetError(path, "sample count must not overflow expected file size");
  }
  // The physical file must equal header plus count * 785 bytes exactly.
  const std::uint64_t expected_size =
      kHeaderSize + static_cast<std::uint64_t>(dataset.sample_count) *
                        kBytesPerSample;
  if (file_size != expected_size) {
    DatasetError(path, "file size must match header count");
  }
  // Reject counts whose packed image bytes cannot fit addressable memory.
  if (dataset.sample_count >
      std::numeric_limits<std::size_t>::max() / kImageSize) {
    DatasetError(path, "image byte count must fit addressable memory");
  }

  // Size the image and label storage from the validated sample count.
  const std::size_t image_bytes =
      static_cast<std::size_t>(dataset.sample_count) *
      static_cast<std::size_t>(kImageSize);
  try {
    dataset.images.resize(image_bytes);
    dataset.labels.resize(dataset.sample_count);
  } catch (const std::bad_alloc&) {
    DatasetError(path, "dataset storage must be allocatable");
  } catch (const std::length_error&) {
    DatasetError(path, "dataset storage must fit container limits");
  }

  // Read the dense image payload, then the per-sample label payload.
  binary_io::ReadExact(input, dataset.images.data(), dataset.images.size(),
                       path,
                       "image payload must contain count * 784 bytes");
  binary_io::ReadExact(input, dataset.labels.data(), dataset.labels.size(),
                       path, "label payload must contain count bytes");
  // Every label must be a class index in [0, 9]; name the first offender.
  for (std::size_t index = 0; index < dataset.labels.size(); ++index) {
    if (dataset.labels[index] > 9) {
      std::ostringstream invariant;
      invariant << "label values must be in [0, 9]; label " << index
                << " is " << static_cast<unsigned int>(dataset.labels[index]);
      DatasetError(path, invariant.str());
    }
  }

  // Require a clean end-of-stream with no trailing bytes.
  if (input.peek() != std::char_traits<char>::eof()) {
    DatasetError(path, "file must contain no trailing bytes");
  }
  if (input.bad()) {
    DatasetError(path, "file end must be readable");
  }
  return dataset;
}

// Enforces a role's exact sample count unless nonstandard counts are allowed;
// loading never enforces a role, only this separate check does.
void RequireDatasetCount(const MnistDataset& dataset, std::uint32_t expected,
                         bool allow_nonstandard, const std::string& role) {
  if (allow_nonstandard || dataset.sample_count == expected) {
    return;
  }
  std::ostringstream message;
  message << role << " count must equal " << expected << "; got "
          << dataset.sample_count;
  throw std::runtime_error(message.str());
}
