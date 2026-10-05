#include "cli.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
#include <string>

namespace {

// Records a parse error in the caller's slot and returns false to propagate.
bool Fail(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

// Lists every option the parser recognizes, regardless of command.
bool IsKnownOption(const std::string& option) {
  return option == "--train" || option == "--test" ||
         option == "--output" || option == "--epochs" ||
         option == "--batch-size" || option == "--seed" ||
         option == "--device" || option == "--data" ||
         option == "--weights" || option == "--min-accuracy" ||
         option == "--index" || option == "--allow-nonstandard-count";
}

// Maps a command to its stable lowercase name for diagnostics.
const char* CommandName(Command command) {
  switch (command) {
    case Command::kTrain:
      return "train";
    case Command::kEvaluate:
      return "evaluate";
    case Command::kInfer:
      return "infer";
  }
  return "unknown";
}

// Enforces the per-command allowlist; --device is accepted by every command.
bool IsAllowedOption(Command command, const std::string& option) {
  if (option == "--device") {
    return true;
  }
  switch (command) {
    case Command::kTrain:
      return option == "--train" || option == "--test" ||
             option == "--output" || option == "--epochs" ||
             option == "--batch-size" || option == "--seed" ||
             option == "--allow-nonstandard-count";
    case Command::kEvaluate:
      return option == "--data" || option == "--weights" ||
             option == "--min-accuracy" ||
             option == "--allow-nonstandard-count";
    case Command::kInfer:
      return option == "--data" || option == "--weights" ||
             option == "--index";
  }
  return false;
}

// Parses a decimal unsigned value up to an inclusive maximum, rejecting empty
// input, non-digits, overflow, and out-of-range values.
bool ParseUnsigned(const std::string& text, std::uint64_t maximum,
                   std::uint64_t* value) {
  if (text.empty()) {
    return false;
  }
  for (const char character : text) {
    if (character < '0' || character > '9') {
      return false;
    }
  }

  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
      parsed > maximum) {
    return false;
  }
  *value = static_cast<std::uint64_t>(parsed);
  return true;
}

// Parses a finite unit-interval accuracy, rejecting leading whitespace and
// out-of-range values.
bool ParseAccuracy(const std::string& text, float* value) {
  if (text.empty() || text[0] == ' ' || text[0] == '\t' ||
      text[0] == '\n' || text[0] == '\r' || text[0] == '\f' ||
      text[0] == '\v') {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const double parsed = std::strtod(text.c_str(), &end);
  if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
      !std::isfinite(parsed) || parsed < 0.0 || parsed > 1.0) {
    return false;
  }
  *value = static_cast<float>(parsed);
  return true;
}

// Fails unless the given required option was seen during parsing.
bool RequireOption(const std::set<std::string>& seen,
                   const std::string& option, std::string* error) {
  if (seen.count(option) == 0) {
    return Fail(error, "missing required option " + option);
  }
  return true;
}

}  // namespace

// Parses argv into caller-owned options with no side effects; the result is
// committed only after every command-specific requirement is satisfied.
bool ParseCli(int argc, const char* const* argv, CliOptions* options,
              std::string* error) {
  if (error == nullptr) {
    return false;
  }
  error->clear();
  if (options == nullptr) {
    return Fail(error, "options output must not be null");
  }
  if (argc < 0 || argv == nullptr) {
    return Fail(error, "argv must not be null and argc must be nonnegative");
  }
  if (argc < 2) {
    return Fail(error, "missing command: expected train, evaluate, or infer");
  }
  if (argv[1] == nullptr) {
    return Fail(error, "command argument must not be null");
  }

  // Apply documented defaults so omitted optional options stay well-defined.
  CliOptions parsed{};
  parsed.train.epochs = 20;
  parsed.train.batch_size = 128;
  parsed.train.seed = UINT64_C(1337);
  parsed.train.device = 0;
  parsed.train.allow_nonstandard_count = false;
  parsed.evaluate.minimum_accuracy = 0.0F;
  parsed.evaluate.device = 0;
  parsed.evaluate.allow_nonstandard_count = false;
  parsed.infer.index = 0;
  parsed.infer.device = 0;

  const std::string command(argv[1]);
  if (command == "train") {
    parsed.command = Command::kTrain;
  } else if (command == "evaluate") {
    parsed.command = Command::kEvaluate;
  } else if (command == "infer") {
    parsed.command = Command::kInfer;
  } else {
    return Fail(error, "unknown command " + command);
  }

  // Track seen options to reject duplicates and verify required options.
  std::set<std::string> seen;
  for (int argument = 2; argument < argc; ++argument) {
    if (argv[argument] == nullptr) {
      return Fail(error, "option argument must not be null");
    }
    const std::string option(argv[argument]);
    if (!IsKnownOption(option)) {
      return Fail(error, "unknown option " + option);
    }
    if (!IsAllowedOption(parsed.command, option)) {
      return Fail(error, option + " is not valid for " +
                             CommandName(parsed.command));
    }
    if (!seen.insert(option).second) {
      return Fail(error, "duplicate " + option);
    }

    if (option == "--allow-nonstandard-count") {
      if (parsed.command == Command::kTrain) {
        parsed.train.allow_nonstandard_count = true;
      } else {
        parsed.evaluate.allow_nonstandard_count = true;
      }
      continue;
    }

    // Reject a missing value and any token that looks like the next option.
    if (argument + 1 >= argc || argv[argument + 1] == nullptr ||
        std::string(argv[argument + 1]).compare(0, 2, "--") == 0) {
      return Fail(error, option + " requires a value");
    }
    const std::string value(argv[++argument]);
    if (value.empty()) {
      return Fail(error, option + " requires a nonempty value");
    }

    if (option == "--train") {
      parsed.train.train_path = value;
    } else if (option == "--test") {
      parsed.train.test_path = value;
    } else if (option == "--output") {
      parsed.train.output_path = value;
    } else if (option == "--data") {
      if (parsed.command == Command::kEvaluate) {
        parsed.evaluate.data_path = value;
      } else {
        parsed.infer.data_path = value;
      }
    } else if (option == "--weights") {
      if (parsed.command == Command::kEvaluate) {
        parsed.evaluate.weights_path = value;
      } else {
        parsed.infer.weights_path = value;
      }
    } else if (option == "--min-accuracy") {
      if (!ParseAccuracy(value, &parsed.evaluate.minimum_accuracy)) {
        return Fail(error,
                    "invalid value for --min-accuracy: expected finite [0, 1]");
      }
    } else {
      // Bound each numeric option by its documented inclusive maximum.
      std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
      if (option == "--epochs") {
        maximum = 1000;
      } else if (option == "--batch-size") {
        maximum = 1024;
      } else if (option == "--device") {
        maximum = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
      }
      std::uint64_t numeric = 0;
      if (!ParseUnsigned(value, maximum, &numeric) ||
          ((option == "--epochs" || option == "--batch-size") &&
           numeric == 0)) {
        return Fail(error, "invalid value for " + option);
      }
      if (option == "--epochs") {
        parsed.train.epochs = static_cast<std::uint32_t>(numeric);
      } else if (option == "--batch-size") {
        parsed.train.batch_size = static_cast<std::uint32_t>(numeric);
      } else if (option == "--seed") {
        parsed.train.seed = numeric;
      } else if (option == "--device") {
        if (parsed.command == Command::kTrain) {
          parsed.train.device = static_cast<int>(numeric);
        } else if (parsed.command == Command::kEvaluate) {
          parsed.evaluate.device = static_cast<int>(numeric);
        } else {
          parsed.infer.device = static_cast<int>(numeric);
        }
      } else if (option == "--index") {
        parsed.infer.index = numeric;
      }
    }
  }

  // Require each command's mandatory options before committing the result.
  if (parsed.command == Command::kTrain) {
    if (!RequireOption(seen, "--train", error) ||
        !RequireOption(seen, "--test", error) ||
        !RequireOption(seen, "--output", error)) {
      return false;
    }
  } else if (parsed.command == Command::kEvaluate) {
    if (!RequireOption(seen, "--data", error) ||
        !RequireOption(seen, "--weights", error)) {
      return false;
    }
  } else if (!RequireOption(seen, "--data", error) ||
             !RequireOption(seen, "--weights", error) ||
             !RequireOption(seen, "--index", error)) {
    return false;
  }

  // Commit the parsed result only after all validation has succeeded.
  *options = parsed;
  return true;
}

std::string Usage() {
  return "Usage:\n"
         "  lenet_cuda train --train PATH --test PATH --output PATH "
         "[--epochs N] [--batch-size N] [--seed N] [--device N] "
         "[--allow-nonstandard-count]\n"
         "  lenet_cuda evaluate --data PATH --weights PATH "
         "[--min-accuracy X] [--device N] [--allow-nonstandard-count]\n"
         "  lenet_cuda infer --data PATH --weights PATH --index N "
         "[--device N]\n";
}
