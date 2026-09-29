#include "random.h"
#include "test_harness.h"
#include "training_data.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

const std::size_t kImageSize = 28 * 28;

MnistDataset MakeDataset(std::uint32_t sample_count) {
  MnistDataset dataset{};
  dataset.sample_count = sample_count;
  dataset.rows = 28;
  dataset.columns = 28;
  dataset.images.resize(static_cast<std::size_t>(sample_count) * kImageSize);
  dataset.labels.resize(sample_count);
  for (std::uint32_t sample = 0; sample < sample_count; ++sample) {
    dataset.labels[sample] = static_cast<std::uint8_t>(sample % 10);
    for (std::size_t pixel = 0; pixel < kImageSize; ++pixel) {
      dataset.images[static_cast<std::size_t>(sample) * kImageSize + pixel] =
          static_cast<std::uint8_t>((sample * 17 + pixel) % 256);
    }
  }
  return dataset;
}

}  // namespace

TEST_CASE(workflow_split_uses_official_task4_ordering) {
  const DatasetSplit actual = MakeWorkflowSplit(60000, 1337, false);
  const DatasetSplit canonical = MakeTrainValidationSplit(60000, 5000, 1337);
  EXPECT_EQ(std::size_t{55000}, actual.training_indices.size());
  EXPECT_EQ(std::size_t{5000}, actual.validation_indices.size());
  EXPECT_EQ(canonical.training_indices, actual.training_indices);
  EXPECT_EQ(canonical.validation_indices, actual.validation_indices);
}

TEST_CASE(workflow_split_gates_and_sizes_nonstandard_counts) {
  EXPECT_THROW_CONTAINS(MakeWorkflowSplit(1280, 9, false),
                        "sample_count must equal 60000");
  EXPECT_THROW_CONTAINS(MakeWorkflowSplit(0, 9, true), "at least 2");
  EXPECT_THROW_CONTAINS(MakeWorkflowSplit(1, 9, true), "at least 2");

  const DatasetSplit actual = MakeWorkflowSplit(1280, 9, true);
  const DatasetSplit canonical = MakeTrainValidationSplit(1280, 256, 9);
  EXPECT_EQ(std::size_t{1024}, actual.training_indices.size());
  EXPECT_EQ(std::size_t{256}, actual.validation_indices.size());
  EXPECT_EQ(canonical.training_indices, actual.training_indices);
  EXPECT_EQ(canonical.validation_indices, actual.validation_indices);

  const DatasetSplit two = MakeWorkflowSplit(2, 9, true);
  EXPECT_EQ(std::size_t{1}, two.training_indices.size());
  EXPECT_EQ(std::size_t{1}, two.validation_indices.size());
}

TEST_CASE(pack_batch_makes_128_128_1_without_correspondence_drift) {
  const MnistDataset dataset = MakeDataset(257);
  std::vector<std::uint32_t> order(257);
  std::iota(order.begin(), order.end(), UINT32_C(0));
  std::reverse(order.begin(), order.end());

  HostBatch batch{};
  const std::uint32_t expected_sizes[] = {128, 128, 1};
  std::uint32_t offset = 0;
  std::size_t retained_image_capacity = 0;
  for (const std::uint32_t expected_size : expected_sizes) {
    EXPECT_EQ(expected_size, PackBatch(dataset, order, offset, 128, &batch));
    EXPECT_EQ(expected_size, batch.size);
    EXPECT_EQ(static_cast<std::size_t>(expected_size) * kImageSize,
              batch.images.size());
    EXPECT_EQ(static_cast<std::size_t>(expected_size), batch.labels.size());
    EXPECT_EQ(static_cast<std::size_t>(expected_size),
              batch.original_indices.size());
    for (std::uint32_t packed = 0; packed < expected_size; ++packed) {
      const std::uint32_t original = order[offset + packed];
      EXPECT_EQ(original, batch.original_indices[packed]);
      EXPECT_EQ(dataset.labels[original], batch.labels[packed]);
      for (std::size_t pixel = 0; pixel < kImageSize; ++pixel) {
        EXPECT_EQ(dataset.images[static_cast<std::size_t>(original) *
                                     kImageSize +
                                 pixel],
                  batch.images[static_cast<std::size_t>(packed) * kImageSize +
                               pixel]);
      }
    }
    if (offset == 0) {
      retained_image_capacity = batch.images.capacity();
    }
    offset += expected_size;
  }
  EXPECT_EQ(std::uint32_t{257}, offset);
  EXPECT_TRUE(batch.images.capacity() >= retained_image_capacity);
}

TEST_CASE(pack_batch_rejects_invalid_arguments_before_copying) {
  MnistDataset dataset = MakeDataset(3);
  const std::vector<std::uint32_t> order{2, 0, 1};
  HostBatch batch{};
  EXPECT_THROW_CONTAINS(PackBatch(dataset, order, 0, 1, nullptr),
                        "batch must not be null");
  EXPECT_THROW_CONTAINS(PackBatch(dataset, order, 0, 0, &batch),
                        "capacity must be nonzero");
  EXPECT_THROW_CONTAINS(PackBatch(dataset, order, 3, 1, &batch),
                        "offset must be below order size");

  const std::vector<std::uint32_t> bad_order{0, 3};
  EXPECT_THROW_CONTAINS(PackBatch(dataset, bad_order, 0, 2, &batch),
                        "order index must be below sample_count");

  dataset.images.pop_back();
  EXPECT_THROW_CONTAINS(PackBatch(dataset, order, 0, 1, &batch),
                        "dataset image storage");
}
