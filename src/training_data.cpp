#include "training_data.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

// Builds the workflow split: 60000 samples always yield 55000/5000, while
// other counts require the nonstandard allowance and at least two samples.
DatasetSplit MakeWorkflowSplit(std::uint32_t sample_count, std::uint64_t seed,
                               bool allow_nonstandard_count) {
  // Official counts fix validation at 5000; nonstandard counts use one fifth.
  std::uint32_t validation_count = 5000;
  if (sample_count != 60000) {
    if (!allow_nonstandard_count) {
      throw std::invalid_argument(
          "sample_count must equal 60000 unless nonstandard counts are allowed");
    }
    if (sample_count < 2) {
      throw std::invalid_argument(
          "nonstandard sample_count must contain at least 2 samples");
    }
    validation_count = std::max(UINT32_C(1), sample_count / 5);
  }
  return MakeTrainValidationSplit(sample_count, validation_count, seed);
}

// Packs up to capacity entries starting at offset into reusable batch storage,
// returning the actual, possibly partial, batch size.
std::uint32_t PackBatch(const MnistDataset& dataset,
                        const std::vector<std::uint32_t>& order,
                        std::uint32_t offset, std::uint32_t capacity,
                        HostBatch* batch) {
  if (batch == nullptr) {
    throw std::invalid_argument("batch must not be null");
  }
  if (capacity == 0) {
    throw std::invalid_argument("capacity must be nonzero");
  }
  if (static_cast<std::size_t>(offset) >= order.size()) {
    throw std::out_of_range("offset must be below order size");
  }

  // Validate dataset dimensions and storage before any copy.
  const std::size_t rows = dataset.rows;
  const std::size_t columns = dataset.columns;
  if (rows == 0 || columns == 0 ||
      rows > std::numeric_limits<std::size_t>::max() / columns) {
    throw std::invalid_argument("dataset image dimensions must be nonzero");
  }
  const std::size_t image_size = rows * columns;
  if (dataset.sample_count >
      std::numeric_limits<std::size_t>::max() / image_size ||
      dataset.images.size() !=
          static_cast<std::size_t>(dataset.sample_count) * image_size) {
    throw std::invalid_argument(
        "dataset image storage must match sample_count and dimensions");
  }
  if (dataset.labels.size() != dataset.sample_count) {
    throw std::invalid_argument(
        "dataset label storage must match sample_count");
  }

  // The final batch may be partial, so copy only the remaining entries.
  const std::size_t remaining = order.size() - offset;
  const std::size_t actual =
      std::min(remaining, static_cast<std::size_t>(capacity));
  if (actual > std::numeric_limits<std::size_t>::max() / image_size) {
    throw std::invalid_argument(
        "packed image byte count must fit size_t");
  }
  // Check every selected order index before touching destination storage.
  for (std::size_t packed = 0; packed < actual; ++packed) {
    if (order[static_cast<std::size_t>(offset) + packed] >=
        dataset.sample_count) {
      throw std::out_of_range("order index must be below sample_count");
    }
  }

  // Reuse the caller's vectors, then copy images, labels, and original indices
  // together so their correspondence never drifts.
  batch->images.resize(actual * image_size);
  batch->labels.resize(actual);
  batch->original_indices.resize(actual);
  for (std::size_t packed = 0; packed < actual; ++packed) {
    const std::uint32_t original =
        order[static_cast<std::size_t>(offset) + packed];
    const std::uint8_t* source =
        dataset.images.data() + static_cast<std::size_t>(original) * image_size;
    std::copy(source, source + image_size,
              batch->images.begin() + packed * image_size);
    batch->labels[packed] = dataset.labels[original];
    batch->original_indices[packed] = original;
  }
  batch->size = static_cast<std::uint32_t>(actual);
  return batch->size;
}
