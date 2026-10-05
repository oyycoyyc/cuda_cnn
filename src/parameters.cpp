#include "parameters.h"

#include "random.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

// Domain tag separating parameter initialization streams from the split,
// shuffle, and translation streams derived from the same caller seed.
const std::uint64_t kInitializationDomain =
    UINT64_C(0x494E49545F563031);
const double kPi = 3.141592653589793238462643383279502884;
const double kInverseTwoTo52 = 1.0 / 4503599627370496.0;

// Maps a 64-bit draw into the open unit interval (0, 1); the half-step offset
// keeps the logarithm well-defined.
double OpenUniform(std::uint64_t raw) {
  return (static_cast<double>(raw >> 12) + 0.5) * kInverseTwoTo52;
}

// Reports which canonical tensor slot violates the expected schema.
[[noreturn]] void SchemaError(std::size_t index,
                              const std::string& invariant) {
  std::ostringstream message;
  message << "parameter tensor " << index << ' ' << invariant;
  throw std::invalid_argument(message.str());
}

}  // namespace

// The sole canonical ten-tensor schema: ordered names, ranks, four-slot
// shapes, element counts, fan-in values, and bias flags.
const std::array<ParameterSpec, 10>& LenetParameterSpecs() {
  static const std::array<ParameterSpec, 10> specs{{
      {"conv1.weight", 4, {{6, 1, 5, 5}}, 150, 25, false},
      {"conv1.bias", 1, {{6, 1, 1, 1}}, 6, 0, true},
      {"conv2.weight", 4, {{16, 6, 5, 5}}, 2400, 150, false},
      {"conv2.bias", 1, {{16, 1, 1, 1}}, 16, 0, true},
      {"fc1.weight", 2, {{120, 256, 1, 1}}, 30720, 256, false},
      {"fc1.bias", 1, {{120, 1, 1, 1}}, 120, 0, true},
      {"fc2.weight", 2, {{84, 120, 1, 1}}, 10080, 120, false},
      {"fc2.bias", 1, {{84, 1, 1, 1}}, 84, 0, true},
      {"fc3.weight", 2, {{10, 84, 1, 1}}, 840, 84, false},
      {"fc3.bias", 1, {{10, 1, 1, 1}}, 10, 0, true},
  }};
  return specs;
}

// Allocates the canonical parameter set in schema order with zeroed storage.
ParameterSet CreateLenetParameters() {
  const std::array<ParameterSpec, 10>& specs = LenetParameterSpecs();
  ParameterSet parameters;
  parameters.reserve(specs.size());
  for (const ParameterSpec& spec : specs) {
    ParameterTensor tensor;
    tensor.name = spec.name;
    tensor.rank = spec.rank;
    tensor.dimensions = spec.dimensions;
    tensor.values.assign(static_cast<std::size_t>(spec.element_count), 0.0F);
    parameters.push_back(std::move(tensor));
  }
  return parameters;
}

// Requires exactly ten tensors matching the canonical name, rank, shape, and
// element-count schema in order.
void ValidateLenetParameters(const ParameterSet& parameters) {
  const std::array<ParameterSpec, 10>& specs = LenetParameterSpecs();
  if (parameters.size() != specs.size()) {
    throw std::invalid_argument("parameter tensor count must equal 10");
  }
  for (std::size_t index = 0; index < specs.size(); ++index) {
    const ParameterTensor& tensor = parameters[index];
    const ParameterSpec& spec = specs[index];
    if (tensor.name != spec.name) {
      SchemaError(index, "name or canonical order is invalid");
    }
    if (tensor.rank != spec.rank) {
      SchemaError(index, "rank is invalid");
    }
    if (tensor.dimensions != spec.dimensions) {
      SchemaError(index, "dimensions are invalid");
    }
    if (tensor.values.size() != spec.element_count) {
      SchemaError(index, "value count is invalid");
    }
  }
}

// Fills a validated parameter set from one seed: each weight tensor draws from
// its own derived stream and each bias is zeroed.
void InitializeLenetParameters(std::uint64_t seed, ParameterSet* parameters) {
  if (parameters == nullptr) {
    throw std::invalid_argument("parameters must be non-null");
  }
  ValidateLenetParameters(*parameters);
  const std::array<ParameterSpec, 10>& specs = LenetParameterSpecs();
  for (std::size_t ordinal = 0; ordinal < specs.size(); ++ordinal) {
    const ParameterSpec& spec = specs[ordinal];
    std::vector<float>& values = (*parameters)[ordinal].values;
    // Bias tensors are fixed at zero and consume no random stream.
    if (spec.is_bias) {
      std::fill(values.begin(), values.end(), 0.0F);
      continue;
    }

    // Derive a per-ordinal initialization stream so tensors stay independent
    // and reproducible from the single caller seed.
    const std::uint64_t stream_seed = lenet_random_internal::Derive(
        seed, kInitializationDomain, static_cast<std::uint64_t>(ordinal),
        spec.fan_in);
    SplitMix64 random(stream_seed);
    // He scaling sizes weight values by sqrt(2 / fan_in).
    const double standard_deviation =
        std::sqrt(2.0 / static_cast<double>(spec.fan_in));
    // Transform two open-unit draws into paired normal values, writing the
    // second only when another element remains.
    std::size_t index = 0;
    while (index < values.size()) {
      const double first = OpenUniform(random.Next());
      const double second = OpenUniform(random.Next());
      const double magnitude = std::sqrt(-2.0 * std::log(first));
      const double angle = 2.0 * kPi * second;
      values[index++] = static_cast<float>(
          magnitude * std::cos(angle) * standard_deviation);
      if (index < values.size()) {
        values[index++] = static_cast<float>(
            magnitude * std::sin(angle) * standard_deviation);
      }
    }
  }
}
