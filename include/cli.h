#ifndef INCLUDE_CLI_H_
#define INCLUDE_CLI_H_

#include <cstdint>
#include <string>

// Identifies the workflow selected by the first command-line argument.
enum class Command {
  kTrain,
  kEvaluate,
  kInfer,
};

// Owns all syntax-validated train command values. Paths are retained exactly
// as supplied and are not opened by command-line parsing. Numeric values are
// range checked, and the structure owns its strings without aliasing argv.
struct TrainOptions {
  std::string train_path;
  std::string test_path;
  std::string output_path;
  std::uint32_t epochs;
  std::uint32_t batch_size;
  std::uint64_t seed;
  int device;
  bool allow_nonstandard_count;
};

// Owns all syntax-validated evaluate command values. Paths are retained
// without filesystem access; minimum_accuracy is finite and in [0, 1], and
// the strings do not alias argv.
struct EvaluateOptions {
  std::string data_path;
  std::string weights_path;
  float minimum_accuracy;
  int device;
  bool allow_nonstandard_count;
};

// Owns all syntax-validated infer command values. Paths are retained without
// filesystem access, index is a syntactically valid uint64 value, and the
// strings do not alias argv.
struct InferOptions {
  std::string data_path;
  std::string weights_path;
  std::uint64_t index;
  int device;
};

// Owns the selected command and its corresponding typed options. Only the
// member named by command is meaningful after successful parsing.
struct CliOptions {
  Command command;
  TrainOptions train;
  EvaluateOptions evaluate;
  InferOptions infer;
};

// Parses argv without filesystem or CUDA side effects. argc must be
// nonnegative; argv and every consumed entry, options, and error must be
// non-null. On success, writes an owning CliOptions value, clears error, and
// returns true. On invalid command syntax, duplicate/unknown/cross-command
// options, missing required values, or numeric range errors, leaves options
// unchanged, writes a specific nonempty error when possible, and returns
// false. This function does not validate path readability, output parents,
// CUDA availability, loaded dataset counts, or an index against loaded data.
bool ParseCli(int argc, const char* const* argv, CliOptions* options,
              std::string* error);

// Returns the stable, concise usage text for train, evaluate, and infer. The
// returned string owns its storage and has no lifetime or aliasing constraints.
std::string Usage();

#endif  // INCLUDE_CLI_H_
