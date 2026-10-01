#ifndef INCLUDE_RANDOM_H_
#define INCLUDE_RANDOM_H_

#include <cstdint>
#include <vector>

#if defined(__CUDACC__)
#define LENET_HOST_DEVICE __host__ __device__
#else
#define LENET_HOST_DEVICE
#endif

namespace lenet_random_internal {

// Applies the SplitMix64 finalizer to one value with only unsigned integer
// arithmetic, making the result identical in host and device code.
LENET_HOST_DEVICE inline std::uint64_t Mix64(std::uint64_t value) {
  value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
  return value ^ (value >> 31);
}

// Derives one deterministic random word from a seed, domain, and two indices;
// it owns no state and has no failure path.
LENET_HOST_DEVICE inline std::uint64_t Derive(std::uint64_t seed,
                                               std::uint64_t domain,
                                               std::uint64_t a,
                                               std::uint64_t b) {
  std::uint64_t value = Mix64(seed ^ domain);
  value = Mix64(value ^ a);
  return Mix64(value ^ b);
}

}  // namespace lenet_random_internal

// Owns one deterministic SplitMix64 stream. Instances have no external
// storage, aliases, or lifetime dependencies and are usable from host and
// device code. This integer generator performs no allocation and reports no
// errors.
class SplitMix64 {
 public:
  // Starts a stream at seed; the first Next() increments state before mixing.
  LENET_HOST_DEVICE explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

  // Advances and returns the next full-width SplitMix64 output.
  LENET_HOST_DEVICE std::uint64_t Next() {
    state_ += UINT64_C(0x9E3779B97F4A7C15);
    return lenet_random_internal::Mix64(state_);
  }

  // Returns an unbiased value in [0, exclusive_upper_bound) by rejection
  // sampling. exclusive_upper_bound must be nonzero; callers, including host
  // APIs, must establish this precondition. Device code never throws.
  LENET_HOST_DEVICE std::uint64_t UniformBounded(
      std::uint64_t exclusive_upper_bound) {
    const std::uint64_t threshold =
        (UINT64_C(0) - exclusive_upper_bound) % exclusive_upper_bound;
    std::uint64_t value;
    do {
      value = Next();
    } while (value < threshold);
    return value % exclusive_upper_bound;
  }

 private:
  std::uint64_t state_;
};

// Owns disjoint original dataset indices. Both vectors are flat index lists;
// they own their storage and do not alias each other or caller storage.
struct DatasetSplit {
  std::vector<std::uint32_t> training_indices;
  std::vector<std::uint32_t> validation_indices;
};

// Describes a non-owning integer translation in x/y pixel coordinates.
struct TranslationOffset {
  std::int32_t dx;
  std::int32_t dy;
};

// Deterministically shuffles [0, sample_count), returning the first
// sample_count - validation_count indices for training and the remainder for
// validation. The result owns all storage and aliases nothing. Throws
// std::invalid_argument if validation_count exceeds sample_count.
DatasetSplit MakeTrainValidationSplit(std::uint32_t sample_count,
                                      std::uint32_t validation_count,
                                      std::uint64_t seed);

// Fills caller-owned output with a Fisher-Yates permutation of
// canonical_indices, reusing its allocation when capacity is sufficient.
// output must be non-null and must not alias canonical_indices;
// one_based_epoch must be at least one. Invalid arguments throw
// std::invalid_argument before output is modified. Empty input is valid.
void ShuffledTrainingIndices(
    const std::vector<std::uint32_t>& canonical_indices, std::uint64_t seed,
    std::uint32_t one_based_epoch, std::vector<std::uint32_t>* output);

// Returns an owning Fisher-Yates permutation of canonical_indices without
// modifying or aliasing the input. This convenience wrapper delegates to the
// in-place overload. one_based_epoch must be at least one; invalid arguments
// throw std::invalid_argument. Empty input is valid.
std::vector<std::uint32_t> ShuffledTrainingIndices(
    const std::vector<std::uint32_t>& canonical_indices, std::uint64_t seed,
    std::uint32_t one_based_epoch);

// Returns independent deterministic dx/dy offsets in [-2, 2] for the original
// dataset index, not its shuffled position. one_based_epoch must be nonzero.
// The value owns no resources, has no aliasing or lifetime requirements, and
// the host/device implementation does not throw; host callers validate the
// epoch before entering device-callable paths.
LENET_HOST_DEVICE inline TranslationOffset TranslationOffsetForSample(
    std::uint64_t seed, std::uint32_t one_based_epoch,
    std::uint32_t original_index) {
  const std::uint64_t x_seed = lenet_random_internal::Derive(
      seed, UINT64_C(0x5452414E535F5831), one_based_epoch, original_index);
  const std::uint64_t y_seed = lenet_random_internal::Derive(
      seed, UINT64_C(0x5452414E535F5931), one_based_epoch, original_index);
  SplitMix64 x_random(x_seed);
  SplitMix64 y_random(y_seed);
  TranslationOffset offset;
  offset.dx = static_cast<std::int32_t>(x_random.UniformBounded(5)) - 2;
  offset.dy = static_cast<std::int32_t>(y_random.UniformBounded(5)) - 2;
  return offset;
}

#endif  // INCLUDE_RANDOM_H_
