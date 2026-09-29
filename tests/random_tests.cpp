#include "random.h"
#include "test_harness.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
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

std::vector<std::uint64_t> ParseU64List(const std::string& text) {
  std::vector<std::uint64_t> values;
  std::istringstream input(text);
  std::string token;
  while (std::getline(input, token, ',')) {
    values.push_back(static_cast<std::uint64_t>(std::stoull(token)));
  }
  return values;
}

std::vector<std::int32_t> ParseI32List(const std::string& text) {
  std::vector<std::int32_t> values;
  std::istringstream input(text);
  std::string token;
  while (std::getline(input, token, ',')) {
    values.push_back(static_cast<std::int32_t>(std::stol(token)));
  }
  return values;
}

void ExpectPrefix(const std::vector<std::uint32_t>& actual,
                  const std::string& expected_text) {
  const std::vector<std::uint64_t> expected = ParseU64List(expected_text);
  EXPECT_TRUE(actual.size() >= expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(expected[index], static_cast<std::uint64_t>(actual[index]));
  }
}

void ExpectPermutation(const std::vector<std::uint32_t>& expected,
                       std::vector<std::uint32_t> actual) {
  std::vector<std::uint32_t> sorted_expected = expected;
  std::sort(sorted_expected.begin(), sorted_expected.end());
  std::sort(actual.begin(), actual.end());
  EXPECT_EQ(sorted_expected, actual);
}

}  // namespace

TEST_CASE(splitmix64_matches_independent_raw_vectors) {
  const std::map<std::string, std::string> vectors = LoadVectors();
  const std::vector<std::uint64_t> expected = ParseU64List(vectors.at("raw"));
  SplitMix64 random(1337);
  for (std::uint64_t value : expected) {
    EXPECT_EQ(value, random.Next());
  }
}

TEST_CASE(uniform_bounded_handles_one_and_rejection) {
  SplitMix64 bound_one(77);
  for (int index = 0; index < 20; ++index) {
    EXPECT_EQ(UINT64_C(0), bound_one.UniformBounded(1));
  }

  const std::map<std::string, std::string> vectors = LoadVectors();
  const std::vector<std::uint64_t> rejection =
      ParseU64List(vectors.at("rejection"));
  EXPECT_EQ(std::size_t{5}, rejection.size());
  SplitMix64 raw(rejection[0]);
  EXPECT_EQ(rejection[2], raw.Next());
  EXPECT_EQ(rejection[3], raw.Next());
  SplitMix64 bounded(rejection[0]);
  EXPECT_EQ(rejection[4], bounded.UniformBounded(rejection[1]));
}

TEST_CASE(split_matches_fixed_prefixes_and_is_complete) {
  const std::map<std::string, std::string> vectors = LoadVectors();
  const DatasetSplit split = MakeTrainValidationSplit(60000, 5000, 1337);
  EXPECT_EQ(std::size_t{55000}, split.training_indices.size());
  EXPECT_EQ(std::size_t{5000}, split.validation_indices.size());
  ExpectPrefix(split.training_indices, vectors.at("split_training_prefix"));
  ExpectPrefix(split.validation_indices,
               vectors.at("split_validation_prefix"));

  std::vector<std::uint32_t> combined = split.training_indices;
  combined.insert(combined.end(), split.validation_indices.begin(),
                  split.validation_indices.end());
  std::sort(combined.begin(), combined.end());
  for (std::uint32_t index = 0; index < 60000; ++index) {
    EXPECT_EQ(index, combined[index]);
  }
}

TEST_CASE(epoch_shuffle_matches_prefix_and_preserves_training_set) {
  const std::map<std::string, std::string> vectors = LoadVectors();
  const DatasetSplit split = MakeTrainValidationSplit(60000, 5000, 1337);
  const std::vector<std::uint32_t> epoch_one =
      ShuffledTrainingIndices(split.training_indices, 1337, 1);
  const std::vector<std::uint32_t> epoch_two =
      ShuffledTrainingIndices(split.training_indices, 1337, 2);
  ExpectPrefix(epoch_one, vectors.at("epoch1_shuffle_prefix"));
  ExpectPermutation(split.training_indices, epoch_one);
  ExpectPermutation(split.training_indices, epoch_two);
  EXPECT_TRUE(epoch_one != epoch_two);
}

TEST_CASE(epoch_shuffle_in_place_reuses_caller_capacity) {
  std::vector<std::uint32_t> canonical(257);
  for (std::uint32_t index = 0; index < canonical.size(); ++index) {
    canonical[index] = index;
  }
  std::vector<std::uint32_t> shuffled;
  shuffled.reserve(canonical.size());
  std::uint32_t* const reserved_storage = shuffled.data();

  ShuffledTrainingIndices(canonical, 1337, 1, &shuffled);
  EXPECT_EQ(reserved_storage, shuffled.data());
  EXPECT_EQ(ShuffledTrainingIndices(canonical, 1337, 1), shuffled);
  ShuffledTrainingIndices(canonical, 1337, 2, &shuffled);
  EXPECT_EQ(reserved_storage, shuffled.data());
  ExpectPermutation(canonical, shuffled);
  EXPECT_THROW_CONTAINS(
      ShuffledTrainingIndices(canonical, 1337, 1, nullptr), "output");
}

TEST_CASE(host_random_apis_reject_invalid_arguments) {
  EXPECT_THROW_CONTAINS(MakeTrainValidationSplit(4, 5, 1),
                        "validation_count");
  const std::vector<std::uint32_t> indices{1, 2, 3};
  EXPECT_THROW_CONTAINS(ShuffledTrainingIndices(indices, 1, 0), "epoch");
}

TEST_CASE(translation_uses_original_index_and_fixed_domains) {
  const std::map<std::string, std::string> vectors = LoadVectors();
  for (const std::uint32_t index : {0U, 1U, 12345U, 59999U}) {
    const std::vector<std::int32_t> expected =
        ParseI32List(vectors.at("translation." + std::to_string(index)));
    const TranslationOffset offset = TranslationOffsetForSample(1337, 1, index);
    EXPECT_EQ(expected[0], offset.dx);
    EXPECT_EQ(expected[1], offset.dy);
    EXPECT_TRUE(offset.dx >= -2 && offset.dx <= 2);
    EXPECT_TRUE(offset.dy >= -2 && offset.dy <= 2);
  }
}
