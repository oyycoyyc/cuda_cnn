#include "random.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

const std::uint64_t kSplitDomain = UINT64_C(0x53504C49545F5631);
const std::uint64_t kShuffleDomain = UINT64_C(0x53485546464C4531);

void Shuffle(std::vector<std::uint32_t>* indices, std::uint64_t stream_seed) {
  SplitMix64 random(stream_seed);
  for (std::size_t remaining = indices->size(); remaining > 1; --remaining) {
    const std::size_t other = static_cast<std::size_t>(
        random.UniformBounded(static_cast<std::uint64_t>(remaining)));
    std::swap((*indices)[remaining - 1], (*indices)[other]);
  }
}

}  // namespace

DatasetSplit MakeTrainValidationSplit(std::uint32_t sample_count,
                                      std::uint32_t validation_count,
                                      std::uint64_t seed) {
  if (validation_count > sample_count) {
    throw std::invalid_argument(
        "validation_count must not exceed sample_count");
  }

  std::vector<std::uint32_t> indices(sample_count);
  std::iota(indices.begin(), indices.end(), UINT32_C(0));
  const std::uint64_t stream_seed = lenet_random_internal::Derive(
      seed, kSplitDomain, sample_count, validation_count);
  Shuffle(&indices, stream_seed);

  const std::size_t training_count =
      static_cast<std::size_t>(sample_count - validation_count);
  DatasetSplit split;
  split.training_indices.assign(indices.begin(),
                                indices.begin() + training_count);
  split.validation_indices.assign(indices.begin() + training_count,
                                  indices.end());
  return split;
}

std::vector<std::uint32_t> ShuffledTrainingIndices(
    const std::vector<std::uint32_t>& canonical_indices, std::uint64_t seed,
    std::uint32_t one_based_epoch) {
  if (one_based_epoch == 0) {
    throw std::invalid_argument("epoch must be one-based");
  }
  std::vector<std::uint32_t> shuffled = canonical_indices;
  const std::uint64_t stream_seed = lenet_random_internal::Derive(
      seed, kShuffleDomain, one_based_epoch,
      static_cast<std::uint64_t>(canonical_indices.size()));
  Shuffle(&shuffled, stream_seed);
  return shuffled;
}
