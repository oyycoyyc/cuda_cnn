#ifndef INCLUDE_TRAINING_DATA_H_
#define INCLUDE_TRAINING_DATA_H_

#include "dataset.h"
#include "random.h"

#include <cstdint>
#include <vector>

// Owns a packed host batch. Images use contiguous sample-major, row-major
// [size][dataset.rows][dataset.columns] uint8 layout; labels and
// original_indices each contain size entries. All vectors own their storage,
// do not alias the source dataset or order, and may retain capacity for reuse.
struct HostBatch {
  std::vector<std::uint8_t> images;
  std::vector<std::uint8_t> labels;
  std::vector<std::uint32_t> original_indices;
  std::uint32_t size;
};

// Builds the deterministic train/validation split used by the workflow. An
// official sample_count of 60000 always yields 55000 training and 5000
// validation indices. Other counts require allow_nonstandard_count and at
// least two samples; validation then receives max(1, sample_count / 5).
// The returned vectors own their storage and use MakeTrainValidationSplit's
// canonical deterministic ordering. Throws std::invalid_argument when the
// count is not permitted.
DatasetSplit MakeWorkflowSplit(std::uint32_t sample_count, std::uint64_t seed,
                               bool allow_nonstandard_count);

// Packs up to capacity entries beginning at offset in order and returns the
// actual batch size, also storing it in batch->size. dataset must have
// consistent image and label storage; offset must identify an order entry;
// capacity must be nonzero; every selected order index must be below
// dataset.sample_count; and batch must be non-null. Source and destination
// buffers must not alias. The function resizes and reuses caller-owned batch
// vectors, copies complete images, labels, and original indices without
// correspondence drift, and throws std::invalid_argument or std::out_of_range
// when a precondition fails.
std::uint32_t PackBatch(const MnistDataset& dataset,
                        const std::vector<std::uint32_t>& order,
                        std::uint32_t offset, std::uint32_t capacity,
                        HostBatch* batch);

#endif  // INCLUDE_TRAINING_DATA_H_
