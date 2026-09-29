#include "cli.h"
#include "test_harness.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

namespace {

bool Parse(std::initializer_list<const char*> arguments, CliOptions* options,
           std::string* error) {
  const std::vector<const char*> argv(arguments);
  return ParseCli(static_cast<int>(argv.size()), argv.data(), options, error);
}

void ExpectParseError(std::initializer_list<const char*> arguments,
                      const std::string& expected_text) {
  CliOptions options{};
  std::string error;
  EXPECT_TRUE(!Parse(arguments, &options, &error));
  EXPECT_TRUE(!error.empty());
  EXPECT_TRUE(error.find(expected_text) != std::string::npos);
}

}  // namespace

TEST_CASE(train_parses_required_paths_and_defaults) {
  CliOptions options{};
  std::string error("old error");
  EXPECT_TRUE(Parse({"lenet_cuda", "train", "--train", "train.bin", "--test",
                     "test.bin", "--output", "weights/model.bin"},
                    &options, &error));
  EXPECT_TRUE(error.empty());
  EXPECT_TRUE(options.command == Command::kTrain);
  EXPECT_EQ(std::string("train.bin"), options.train.train_path);
  EXPECT_EQ(std::string("test.bin"), options.train.test_path);
  EXPECT_EQ(std::string("weights/model.bin"), options.train.output_path);
  EXPECT_EQ(std::uint32_t{20}, options.train.epochs);
  EXPECT_EQ(std::uint32_t{128}, options.train.batch_size);
  EXPECT_EQ(UINT64_C(1337), options.train.seed);
  EXPECT_EQ(0, options.train.device);
  EXPECT_TRUE(!options.train.allow_nonstandard_count);
}

TEST_CASE(train_accepts_inclusive_numeric_bounds_and_flag) {
  CliOptions options{};
  std::string error;
  EXPECT_TRUE(Parse(
      {"lenet_cuda", "train", "--train", "a", "--test", "b", "--output",
       "c", "--epochs", "1000", "--batch-size", "1024", "--seed",
       "18446744073709551615", "--device", "2147483647",
       "--allow-nonstandard-count"},
      &options, &error));
  EXPECT_EQ(std::uint32_t{1000}, options.train.epochs);
  EXPECT_EQ(std::uint32_t{1024}, options.train.batch_size);
  EXPECT_EQ(std::numeric_limits<std::uint64_t>::max(), options.train.seed);
  EXPECT_EQ(std::numeric_limits<int>::max(), options.train.device);
  EXPECT_TRUE(options.train.allow_nonstandard_count);

  EXPECT_TRUE(Parse({"lenet_cuda", "train", "--train", "a", "--test", "b",
                     "--output", "c", "--epochs", "1", "--batch-size",
                     "1"},
                    &options, &error));
  EXPECT_EQ(std::uint32_t{1}, options.train.epochs);
  EXPECT_EQ(std::uint32_t{1}, options.train.batch_size);
}

TEST_CASE(evaluate_parses_defaults_endpoints_and_flag) {
  CliOptions options{};
  std::string error;
  EXPECT_TRUE(Parse({"lenet_cuda", "evaluate", "--data", "test.bin",
                     "--weights", "model.bin"},
                    &options, &error));
  EXPECT_TRUE(options.command == Command::kEvaluate);
  EXPECT_EQ(0.0F, options.evaluate.minimum_accuracy);
  EXPECT_EQ(0, options.evaluate.device);
  EXPECT_TRUE(!options.evaluate.allow_nonstandard_count);

  EXPECT_TRUE(Parse({"lenet_cuda", "evaluate", "--data", "test.bin",
                     "--weights", "model.bin", "--min-accuracy", "1",
                     "--device", "7", "--allow-nonstandard-count"},
                    &options, &error));
  EXPECT_EQ(1.0F, options.evaluate.minimum_accuracy);
  EXPECT_EQ(7, options.evaluate.device);
  EXPECT_TRUE(options.evaluate.allow_nonstandard_count);
}

TEST_CASE(infer_requires_and_parses_uint64_index) {
  CliOptions options{};
  std::string error;
  EXPECT_TRUE(Parse({"lenet_cuda", "infer", "--data", "test.bin",
                     "--weights", "model.bin", "--index",
                     "18446744073709551615"},
                    &options, &error));
  EXPECT_TRUE(options.command == Command::kInfer);
  EXPECT_EQ(std::numeric_limits<std::uint64_t>::max(), options.infer.index);
  EXPECT_EQ(0, options.infer.device);
}

TEST_CASE(required_options_are_enforced_without_filesystem_access) {
  ExpectParseError({"lenet_cuda", "train", "--test", "b", "--output", "c"},
                   "--train");
  ExpectParseError({"lenet_cuda", "train", "--train", "a", "--output", "c"},
                   "--test");
  ExpectParseError({"lenet_cuda", "train", "--train", "a", "--test", "b"},
                   "--output");
  ExpectParseError({"lenet_cuda", "evaluate", "--weights", "w"}, "--data");
  ExpectParseError({"lenet_cuda", "evaluate", "--data", "d"}, "--weights");
  ExpectParseError({"lenet_cuda", "infer", "--weights", "w", "--index", "0"},
                   "--data");
  ExpectParseError({"lenet_cuda", "infer", "--data", "d", "--index", "0"},
                   "--weights");
  ExpectParseError({"lenet_cuda", "infer", "--data", "d", "--weights", "w"},
                   "--index");

  CliOptions options{};
  std::string error;
  EXPECT_TRUE(Parse({"lenet_cuda", "train", "--train", "missing-a",
                     "--test", "missing-b", "--output", "missing/c"},
                    &options, &error));
}

TEST_CASE(rejects_missing_unknown_duplicate_and_cross_command_syntax) {
  ExpectParseError({"lenet_cuda"}, "command");
  ExpectParseError({"lenet_cuda", "serve"}, "serve");
  ExpectParseError({"lenet_cuda", "train", "--train"}, "--train");
  ExpectParseError({"lenet_cuda", "train", "--train", "a", "--train", "b",
                    "--test", "t", "--output", "o"},
                   "duplicate --train");
  ExpectParseError({"lenet_cuda", "evaluate", "--data", "d", "--weights",
                    "w", "--device", "0", "--device", "1"},
                   "duplicate --device");
  ExpectParseError({"lenet_cuda", "train", "--train", "a", "--test", "t",
                    "--output", "o", "--bogus"},
                   "unknown option --bogus");
  ExpectParseError({"lenet_cuda", "train", "--train", "a", "--test", "t",
                    "--output", "o", "--min-accuracy", "0.9"},
                   "--min-accuracy is not valid for train");
  ExpectParseError({"lenet_cuda", "evaluate", "--data", "d", "--weights",
                    "w", "--epochs", "2"},
                   "--epochs is not valid for evaluate");
  ExpectParseError({"lenet_cuda", "infer", "--data", "d", "--weights", "w",
                    "--index", "0", "--allow-nonstandard-count"},
                   "--allow-nonstandard-count is not valid for infer");
}

TEST_CASE(rejects_invalid_unsigned_and_integer_values) {
  for (const char* value : {"0", "1001", "-1", "1x", "4294967296"}) {
    ExpectParseError({"lenet_cuda", "train", "--train", "a", "--test", "b",
                      "--output", "c", "--epochs", value},
                     "--epochs");
  }
  for (const char* value : {"0", "1025", "-1", "2.0", "4294967296"}) {
    ExpectParseError({"lenet_cuda", "train", "--train", "a", "--test", "b",
                      "--output", "c", "--batch-size", value},
                     "--batch-size");
  }
  for (const char* value : {"-1", "18446744073709551616", "1x", ""}) {
    ExpectParseError({"lenet_cuda", "train", "--train", "a", "--test", "b",
                      "--output", "c", "--seed", value},
                     "--seed");
  }
  for (const char* value : {"-1", "2147483648", "1x", "+1"}) {
    ExpectParseError({"lenet_cuda", "infer", "--data", "d", "--weights", "w",
                      "--index", "0", "--device", value},
                     "--device");
  }
  for (const char* value : {"-1", "18446744073709551616", "0x", "+1"}) {
    ExpectParseError({"lenet_cuda", "infer", "--data", "d", "--weights", "w",
                      "--index", value},
                     "--index");
  }
}

TEST_CASE(rejects_out_of_range_nonfinite_and_trailing_accuracy) {
  for (const char* value : {"-0.1", "1.1", "nan", "NaN", "inf", "-inf",
                            "0.5x", "", " 0.5"}) {
    ExpectParseError({"lenet_cuda", "evaluate", "--data", "d", "--weights",
                      "w", "--min-accuracy", value},
                     "--min-accuracy");
  }
}

TEST_CASE(parse_failure_leaves_options_unchanged_and_reports_bad_pointers) {
  CliOptions options{};
  options.command = Command::kInfer;
  options.infer.index = 42;
  std::string error;
  EXPECT_TRUE(!Parse({"lenet_cuda", "infer", "--data", "d"}, &options,
                     &error));
  EXPECT_TRUE(options.command == Command::kInfer);
  EXPECT_EQ(UINT64_C(42), options.infer.index);

  const char* argv[] = {"lenet_cuda", "train"};
  EXPECT_TRUE(!ParseCli(2, argv, nullptr, &error));
  EXPECT_TRUE(error.find("options") != std::string::npos);
  EXPECT_TRUE(!ParseCli(2, argv, &options, nullptr));
  EXPECT_TRUE(!ParseCli(2, nullptr, &options, &error));
  EXPECT_TRUE(error.find("argv") != std::string::npos);
}

TEST_CASE(usage_is_stable_and_lists_each_command) {
  EXPECT_EQ(
      std::string(
          "Usage:\n"
          "  lenet_cuda train --train PATH --test PATH --output PATH "
          "[--epochs N] [--batch-size N] [--seed N] [--device N] "
          "[--allow-nonstandard-count]\n"
          "  lenet_cuda evaluate --data PATH --weights PATH "
          "[--min-accuracy X] [--device N] [--allow-nonstandard-count]\n"
          "  lenet_cuda infer --data PATH --weights PATH --index N "
          "[--device N]\n"),
      Usage());
}
