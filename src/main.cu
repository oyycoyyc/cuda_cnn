#include "cli.h"
#include "train.h"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  CliOptions options{};
  std::string parse_error;
  if (!ParseCli(argc, const_cast<const char* const*>(argv), &options,
                &parse_error)) {
    std::cerr << "error: " << parse_error << '\n' << Usage();
    return kUsageError;
  }

  try {
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
  std::cerr << "error: invalid parsed command\n" << Usage();
  return kRuntimeError;
}
