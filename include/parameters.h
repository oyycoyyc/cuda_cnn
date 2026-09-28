#ifndef INCLUDE_PARAMETERS_H_
#define INCLUDE_PARAMETERS_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Describes one immutable canonical tensor layout. name points to static
// null-terminated storage owned for the program lifetime. dimensions is the
// four-slot row-major shape with unused slots set to one; element_count is its
// product, and fan_in is nonzero only for weights. The descriptor owns no
// dynamic storage and callers must not modify the pointed-to name.
struct ParameterSpec {
  const char* name;
  std::uint32_t rank;
  std::array<std::uint32_t, 4> dimensions;
  std::uint64_t element_count;
  std::uint64_t fan_in;
  bool is_bias;
};

// Owns one parameter tensor. values is contiguous row-major FP32 storage in
// dimensions order; name and values own their memory and do not alias schema
// descriptors or other tensors. rank identifies the used dimensions, whose
// remaining slots are one.
struct ParameterTensor {
  std::string name;
  std::uint32_t rank;
  std::array<std::uint32_t, 4> dimensions;
  std::vector<float> values;
};

// Owns the ten tensors in canonical model/checkpoint order. Tensors and their
// value buffers do not alias one another.
using ParameterSet = std::vector<ParameterTensor>;

// Returns the immutable ten-tensor LeNet schema in canonical order. The
// returned reference and all name pointers remain valid for program lifetime;
// callers receive no ownership. This function has no preconditions or errors.
const std::array<ParameterSpec, 10>& LenetParameterSpecs();

// Allocates and returns a canonical ten-tensor ParameterSet with positive-zero
// values. The caller owns all returned storage. Allocation failures propagate
// as standard container exceptions.
ParameterSet CreateLenetParameters();

// Verifies exact tensor count, canonical order/names, ranks, four-slot shapes,
// and value counts. It reads but never retains or aliases parameters. Throws
// std::invalid_argument naming the first violated schema invariant.
void ValidateLenetParameters(const ParameterSet& parameters);

// Reinitializes canonical parameters deterministically: each weight receives
// its own He-normal stream and each bias becomes positive zero without
// consuming random values. parameters must be non-null and must already match
// LenetParameterSpecs; storage remains caller-owned and is modified in place.
// Throws std::invalid_argument for a null pointer or schema mismatch.
void InitializeLenetParameters(std::uint64_t seed, ParameterSet* parameters);

#endif  // INCLUDE_PARAMETERS_H_
