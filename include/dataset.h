#ifndef INCLUDE_DATASET_H_
#define INCLUDE_DATASET_H_

#include <cstdint>
#include <string>
#include <vector>

// Owns a validated MNIST split. Images are contiguous uint8 pixels in
// sample-major, row-major [sample_count][rows][columns] layout; labels has one
// class ID per image. The vectors own their storage and do not alias external
// input buffers.
struct MnistDataset {
  std::uint32_t sample_count;
  std::uint32_t rows;
  std::uint32_t columns;
  std::vector<std::uint8_t> images;
  std::vector<std::uint8_t> labels;

  // Returns a non-owning pointer to the first pixel of image index. The
  // dataset must remain alive and images must not be modified while the
  // pointer is used. Requires the public fields to retain their validated
  // shape; throws std::out_of_range when index is not below sample_count.
  const std::uint8_t* Image(std::uint32_t index) const;
};

// Loads and owns any nonempty, exactly sized MNISTC1 v1 file with 28x28
// row-major images followed by labels in [0, 9]. The returned vectors do not
// alias file or parser storage. Throws std::runtime_error on open, I/O, format,
// size, or label errors; every such message names path and the failed
// invariant.
MnistDataset LoadMnistDataset(const std::string& path);

// Enforces the caller's role-specific sample count. When allow_nonstandard is
// true, only this count check is bypassed; the dataset must already have been
// fully validated by LoadMnistDataset. Throws std::runtime_error naming role
// and expected count on mismatch.
void RequireDatasetCount(const MnistDataset& dataset, std::uint32_t expected,
                         bool allow_nonstandard, const std::string& role);

#endif  // INCLUDE_DATASET_H_
