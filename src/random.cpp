#include "random.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

// Domain tags keep the split and shuffle streams distinct for a single seed.
const std::uint64_t kSplitDomain = UINT64_C(0x53504C49545F5631);
const std::uint64_t kShuffleDomain = UINT64_C(0x53485546464C4531);

// In-place Fisher-Yates over the supplied index vector using a derived stream.
void Shuffle(std::vector<std::uint32_t>* indices, std::uint64_t stream_seed) {
  SplitMix64 random(stream_seed);
  for (std::size_t remaining = indices->size(); remaining > 1; --remaining) {
    // Draw a uniform swap slot; bounded sampling avoids modulo bias.
    const std::size_t other = static_cast<std::size_t>(
        random.UniformBounded(static_cast<std::uint64_t>(remaining)));
    std::swap((*indices)[remaining - 1], (*indices)[other]);
  }
}

}  // namespace

// Builds the canonical deterministic split: the first sample_count minus
// validation_count indices train and the remainder validate.
DatasetSplit MakeTrainValidationSplit(std::uint32_t sample_count,
                                      std::uint32_t validation_count,
                                      std::uint64_t seed) {
  if (validation_count > sample_count) {
    throw std::invalid_argument(
        "validation_count must not exceed sample_count");
  }

  // Begin from the canonical 0..sample_count-1 order before shuffling.
  std::vector<std::uint32_t> indices(sample_count);
  std::iota(indices.begin(), indices.end(), UINT32_C(0));
  // Bind the stream to both counts so partitions of different sizes diverge.
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

// Replaces output with a deterministic permutation of canonical_indices for
// the given one-based epoch; output must not alias the input.
void ShuffledTrainingIndices(
    const std::vector<std::uint32_t>& canonical_indices, std::uint64_t seed,
    std::uint32_t one_based_epoch, std::vector<std::uint32_t>* output) {
  if (output == nullptr) {
    throw std::invalid_argument("shuffle output must not be null");
  }
  if (output == &canonical_indices) {
    throw std::invalid_argument("shuffle output must not alias input");
  }
  // Epochs are one-based, so epoch zero is always rejected.
  if (one_based_epoch == 0) {
    throw std::invalid_argument("epoch must be one-based");
  }
  output->assign(canonical_indices.begin(), canonical_indices.end());
  // Fold the epoch and index count into the shuffle stream.
  const std::uint64_t stream_seed = lenet_random_internal::Derive(
      seed, kShuffleDomain, one_based_epoch,
      static_cast<std::uint64_t>(canonical_indices.size()));
  Shuffle(output, stream_seed);
}

// Convenience overload that returns a freshly allocated shuffled copy.
std::vector<std::uint32_t> ShuffledTrainingIndices(
    const std::vector<std::uint32_t>& canonical_indices, std::uint64_t seed,
    std::uint32_t one_based_epoch) {
  std::vector<std::uint32_t> shuffled;
  shuffled.reserve(canonical_indices.size());
  ShuffledTrainingIndices(canonical_indices, seed, one_based_epoch, &shuffled);
  return shuffled;
}
