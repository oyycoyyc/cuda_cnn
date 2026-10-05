// Entry point: parses the command line, dispatches the selected workflow, and
// maps syntax and runtime failures to their exit codes.
#include "cli.h"
#include "train.h"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  // Syntax errors print the parse message and usage text before exiting.
  CliOptions options{};
  std::string parse_error;
  if (!ParseCli(argc, const_cast<const char* const*>(argv), &options,
                &parse_error)) {
    std::cerr << "error: " << parse_error << '\n' << Usage();
    return kUsageError;
  }

  // Runtime failures from any dispatched workflow share one error path.
  try {
    // Dispatch the parsed command to its workflow entry point.
    switch (options.command) {
      case Command::kTrain:
        return RunTrain(options.train, std::cout, std::cerr);
      case Command::kEvaluate:
        return RunEvaluate(options.evaluate, std::cout, std::cerr);
      case Command::kInfer:
        return RunInfer(options.infer, std::cout, std::cerr);
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n' << Usage();
    return kRuntimeError;
  }
  // Defensive default for an unknown command enumerator.
  std::cerr << "error: invalid parsed command\n" << Usage();
  return kRuntimeError;
}
