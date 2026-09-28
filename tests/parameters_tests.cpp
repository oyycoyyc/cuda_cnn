#include "parameters.h"
#include "test_harness.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::map<std::string, std::string> LoadVectors() {
  std::ifstream input("tests/repro_vectors.txt");
  if (!input) {
    throw std::runtime_error("tests/repro_vectors.txt must be readable");
  }
  std::map<std::string, std::string> vectors;
  std::string line;
  while (std::getline(input, line)) {
    const std::string::size_type separator = line.find('=');
    if (separator == std::string::npos || separator == 0) {
      throw std::runtime_error("invalid reproducibility vector line");
    }
    vectors[line.substr(0, separator)] = line.substr(separator + 1);
  }
  return vectors;
}

std::vector<double> ParseDoubleList(const std::string& text) {
  std::vector<double> values;
  std::istringstream input(text);
  std::string token;
  while (std::getline(input, token, ',')) {
    values.push_back(std::stod(token));
  }
  return values;
}

struct ExpectedSpec {
  const char* name;
  std::uint32_t rank;
  std::array<std::uint32_t, 4> dimensions;
  std::uint64_t count;
  std::uint64_t fan_in;
  bool is_bias;
};

const std::array<ExpectedSpec, 10> kExpected{{
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

}  // namespace

TEST_CASE(parameter_specs_and_storage_match_canonical_schema) {
  const std::array<ParameterSpec, 10>& specs = LenetParameterSpecs();
  const ParameterSet parameters = CreateLenetParameters();
  EXPECT_EQ(std::size_t{10}, parameters.size());
  std::uint64_t total = 0;
  for (std::size_t index = 0; index < kExpected.size(); ++index) {
    EXPECT_EQ(std::string(kExpected[index].name), std::string(specs[index].name));
    EXPECT_EQ(kExpected[index].rank, specs[index].rank);
    EXPECT_EQ(kExpected[index].dimensions, specs[index].dimensions);
    EXPECT_EQ(kExpected[index].count, specs[index].element_count);
    EXPECT_EQ(kExpected[index].fan_in, specs[index].fan_in);
    EXPECT_EQ(kExpected[index].is_bias, specs[index].is_bias);
    EXPECT_EQ(std::string(kExpected[index].name), parameters[index].name);
    EXPECT_EQ(kExpected[index].rank, parameters[index].rank);
    EXPECT_EQ(kExpected[index].dimensions, parameters[index].dimensions);
    EXPECT_EQ(static_cast<std::size_t>(kExpected[index].count),
              parameters[index].values.size());
    total += parameters[index].values.size();
  }
  EXPECT_EQ(UINT64_C(44426), total);
  ValidateLenetParameters(parameters);
}

TEST_CASE(validation_rejects_every_schema_mismatch_class) {
  ParameterSet parameters = CreateLenetParameters();
  ParameterSet changed = parameters;
  changed.pop_back();
  EXPECT_THROW_CONTAINS(ValidateLenetParameters(changed), "tensor count");

  changed = parameters;
  changed[1].name = changed[0].name;
  EXPECT_THROW_CONTAINS(ValidateLenetParameters(changed), "name");
  changed = parameters;
  changed[0].name = "unknown.weight";
  EXPECT_THROW_CONTAINS(ValidateLenetParameters(changed), "name");
  changed = parameters;
  std::swap(changed[0], changed[1]);
  EXPECT_THROW_CONTAINS(ValidateLenetParameters(changed), "name");
  changed = parameters;
  changed[0].rank = 3;
  EXPECT_THROW_CONTAINS(ValidateLenetParameters(changed), "rank");
  changed = parameters;
  changed[0].dimensions[2] = 4;
  EXPECT_THROW_CONTAINS(ValidateLenetParameters(changed), "dimensions");
  changed = parameters;
  changed[0].values.pop_back();
  EXPECT_THROW_CONTAINS(ValidateLenetParameters(changed), "value count");
}

TEST_CASE(initialization_matches_vectors_and_produces_positive_zero_biases) {
  const std::map<std::string, std::string> vectors = LoadVectors();
  ParameterSet parameters = CreateLenetParameters();
  InitializeLenetParameters(1337, &parameters);
  const std::array<ParameterSpec, 10>& specs = LenetParameterSpecs();
  for (std::size_t tensor = 0; tensor < parameters.size(); ++tensor) {
    if (specs[tensor].is_bias) {
      for (float value : parameters[tensor].values) {
        EXPECT_EQ(0.0F, value);
        EXPECT_TRUE(!std::signbit(value));
      }
      continue;
    }
    const std::vector<double> expected =
        ParseDoubleList(vectors.at("initial." + parameters[tensor].name));
    EXPECT_EQ(std::size_t{5}, expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
      EXPECT_NEAR(expected[index], parameters[tensor].values[index], 1e-6);
    }
    for (float value : parameters[tensor].values) {
      EXPECT_TRUE(std::isfinite(value));
    }
  }
}

TEST_CASE(initialization_is_bit_identical_within_one_binary) {
  ParameterSet first = CreateLenetParameters();
  ParameterSet second = CreateLenetParameters();
  InitializeLenetParameters(1337, &first);
  InitializeLenetParameters(1337, &second);
  for (std::size_t index = 0; index < first.size(); ++index) {
    EXPECT_EQ(first[index].values, second[index].values);
  }
}

TEST_CASE(initialization_validates_pointer_and_schema_before_use) {
  EXPECT_THROW_CONTAINS(InitializeLenetParameters(1, nullptr), "parameters");
  ParameterSet missing = CreateLenetParameters();
  missing.pop_back();
  EXPECT_THROW_CONTAINS(InitializeLenetParameters(1, &missing), "tensor count");
}
