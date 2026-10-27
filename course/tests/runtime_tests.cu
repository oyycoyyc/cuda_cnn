// Tests-first contract for course/runtime.h and the official train/infer
// executables.
//
// These cases are written before course/runtime.cu, root main.cu, and the CMake
// targets exist. Until then the TU fails at link time naming RunCourseInfer; once
// the runtime exists but an executable is missing, the first subprocess case
// fails naming the absent build-adjacent `train`/`infer` entry point. Hardware
// access is never used as a stand-in for a missing entry point.
//
// Every subprocess runs the real official executable with an explicitly built
// environment (all inherited COURSE_* variables removed), a build-owned scratch
// working directory, and absolute fixture paths. Fixtures are tiny raw
// FP32/int32 arrays and real course model bundles; no mounted dataset, test
// label file, pretrained weights, CPU neural-network computation, mock, or
// test-only production hook is used. The train case runs exactly one bounded
// epoch in its own work directory and never the default twenty-epoch schedule.
//
// The executable directory is resolved from /proc/self/exe, so `train` and
// `infer` are located next to this test binary at acceptance. No personal or
// machine-specific path is embedded. The deployment target is Linux; Windows is
// rejected explicitly because these cases require fork/execve subprocesses.

#if defined(_WIN32)
#error "course/runtime_tests.cu requires the POSIX subprocess runner"
#endif

#include "checkpoint.h"
#include "course/interface.h"
#include "course/model_io.h"
#include "course/runtime.h"
#include "course/tests/test_support.h"
#include "parameters.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

extern "C" {
extern char** environ;
}

namespace {

using course_test_support::Fail;
using course_test_support::MakeDirectory;
using course_test_support::MakeScratchDirectory;
using course_test_support::PathExists;
using course_test_support::ReadBytes;
using course_test_support::ReadText;
using course_test_support::SameBits;
using course_test_support::WriteBytes;
using course_test_support::WriteFloats;
using course_test_support::WriteInt32s;
using course_test_support::WriteText;

constexpr std::uint32_t kChannelCount = 1;
constexpr std::uint32_t kImageSide = 28;
constexpr std::uint32_t kClassCount = 10;
constexpr std::uint32_t kImagePixels = kImageSide * kImageSide;
constexpr std::uint32_t kDefaultInferBatch = 1024;
constexpr std::uint32_t kMaximumInferBatch = 16384;
constexpr std::uint64_t kFixtureSeed = UINT64_C(1337);
constexpr float kNormalizationMean = 0.1307F;
constexpr float kNormalizationStddev = 0.3081F;

// The supported course shape for a supplied sample count.
course::Shape FixtureShape(std::uint32_t sample_count) {
  return course::Shape{sample_count, kChannelCount, kImageSide, kImageSide,
                       kClassCount};
}

// Returns the persisted spelling for one input mode.
const char* InputModeName(course::InputMode input_mode) {
  switch (input_mode) {
    case course::InputMode::kUnit:
      return "unit";
    case course::InputMode::kByte:
      return "byte";
    case course::InputMode::kNormalized:
      return "normalized";
  }
  return "";
}

// Resolves this test binary through the Linux kernel, without argv or a
// machine-specific compile-time path.
std::string SelfExecutablePath() {
  char buffer[4096];
  const ssize_t length =
      ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (length <= 0) {
    throw std::runtime_error(
        "could not resolve /proc/self/exe for the runtime test binary");
  }
  buffer[length] = '\0';
  return std::string(buffer);
}

// Directory containing this test binary. CMake places the official `train` and
// `infer` executables in the same build directory at acceptance.
std::string SelfExecutableDirectory() {
  const std::string path = SelfExecutablePath();
  const std::string::size_type separator = path.find_last_of('/');
  if (separator == std::string::npos) {
    return ".";
  }
  return path.substr(0, separator);
}

// Requires the named official executable to exist next to this test binary.
// The absence is reported by name before any hardware access.
std::string OfficialExecutable(const char* name) {
  const std::string path = SelfExecutableDirectory() + "/" + name;
  if (!PathExists(path)) {
    throw std::runtime_error(
        std::string("missing official entry point '") + name +
        "' at build-adjacent path " + path +
        "; build the official train/infer targets next to the runtime tests");
  }
  return path;
}

// One explicit child environment entry. No parent COURSE_* variable survives.
struct EnvVar {
  std::string name;
  std::string value;
};

struct ProcessResult {
  int status;
  std::string out;
  std::string err;
};

// Best-effort async-signal-safe child diagnostic writer. It consumes and
// checks each write result (satisfying warn_unused_result without suppressing
// warnings) and retries short writes; it allocates nothing between fork and
// exec and ignores an unrecoverable error because the child is about to exit.
void WriteChildDiagnostic(const char* message, std::size_t size) {
  while (size > 0) {
    const ssize_t written = ::write(STDERR_FILENO, message, size);
    if (written <= 0) {
      return;
    }
    message += static_cast<std::size_t>(written);
    size -= static_cast<std::size_t>(written);
  }
}

// Copies the parent environment without any COURSE_* entries, then appends the
// case's exact overrides. The parent environment is never modified.
std::vector<std::string> BuildEnvironment(
    const std::vector<EnvVar>& overrides) {
  std::vector<std::string> environment;
  for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
    const std::string variable(*entry);
    if (variable.compare(0, 7, "COURSE_") == 0) {
      continue;
    }
    environment.push_back(variable);
  }
  for (const EnvVar& override_value : overrides) {
    environment.push_back(override_value.name + "=" + override_value.value);
  }
  return environment;
}

// Runs one official executable with an exact argv, an exact COURSE_* environment,
// and an optional working directory. Both output streams are captured through
// pipes with poll, so even a noisy failure cannot deadlock the harness.
ProcessResult RunProcess(const std::string& executable,
                         const std::vector<std::string>& arguments,
                         const std::vector<EnvVar>& environment,
                         const std::string& working_directory) {
  std::vector<std::string> argument_storage;
  argument_storage.push_back(executable);
  argument_storage.insert(argument_storage.end(), arguments.begin(),
                          arguments.end());
  std::vector<char*> argument_pointers;
  for (std::string& argument : argument_storage) {
    argument_pointers.push_back(&argument[0]);
  }
  argument_pointers.push_back(nullptr);

  std::vector<std::string> environment_storage =
      BuildEnvironment(environment);
  std::vector<char*> environment_pointers;
  for (std::string& variable : environment_storage) {
    environment_pointers.push_back(&variable[0]);
  }
  environment_pointers.push_back(nullptr);

  int output_pipe[2] = {-1, -1};
  int error_pipe[2] = {-1, -1};
  if (::pipe(output_pipe) != 0 || ::pipe(error_pipe) != 0) {
    if (output_pipe[0] >= 0) {
      ::close(output_pipe[0]);
      ::close(output_pipe[1]);
    }
    throw std::runtime_error("could not create subprocess capture pipes");
  }

  const pid_t child = ::fork();
  if (child < 0) {
    ::close(output_pipe[0]);
    ::close(output_pipe[1]);
    ::close(error_pipe[0]);
    ::close(error_pipe[1]);
    throw std::runtime_error("could not fork an official executable");
  }

  if (child == 0) {
    if (!working_directory.empty() &&
        ::chdir(working_directory.c_str()) != 0) {
      WriteChildDiagnostic("could not enter the test working directory\n",
                           sizeof("could not enter the test working "
                                  "directory\n") -
                               1);
      ::_exit(126);
    }
    if (::dup2(output_pipe[1], STDOUT_FILENO) < 0 ||
        ::dup2(error_pipe[1], STDERR_FILENO) < 0) {
      WriteChildDiagnostic("could not redirect subprocess output\n",
                           sizeof("could not redirect subprocess output\n") -
                               1);
      ::_exit(126);
    }
    ::close(output_pipe[0]);
    ::close(output_pipe[1]);
    ::close(error_pipe[0]);
    ::close(error_pipe[1]);

    ::execve(executable.c_str(), argument_pointers.data(),
             environment_pointers.data());
    WriteChildDiagnostic("could not execute the official executable\n",
                         sizeof("could not execute the official executable\n") -
                             1);
    ::_exit(127);
  }

  ::close(output_pipe[1]);
  ::close(error_pipe[1]);

  ProcessResult result;
  result.status = 255;
  int descriptors[2] = {output_pipe[0], error_pipe[0]};
  std::string* destinations[2] = {&result.out, &result.err};

  while (descriptors[0] >= 0 || descriptors[1] >= 0) {
    struct pollfd watched[2];
    int watched_index[2] = {-1, -1};
    nfds_t watched_count = 0;
    for (int index = 0; index < 2; ++index) {
      if (descriptors[index] < 0) {
        continue;
      }
      watched[watched_count].fd = descriptors[index];
      watched[watched_count].events = POLLIN | POLLHUP | POLLERR;
      watched[watched_count].revents = 0;
      watched_index[watched_count] = index;
      ++watched_count;
    }

    const int ready = ::poll(watched, watched_count, -1);
    if (ready < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    for (nfds_t slot = 0; slot < watched_count; ++slot) {
      if (watched[slot].revents == 0) {
        continue;
      }
      const int index = watched_index[slot];
      char buffer[4096];
      const ssize_t received =
          ::read(descriptors[index], buffer, sizeof(buffer));
      if (received > 0) {
        destinations[index]->append(buffer,
                                    static_cast<std::size_t>(received));
      } else if (received == 0 ||
                 (received < 0 && errno != EINTR)) {
        ::close(descriptors[index]);
        descriptors[index] = -1;
      }
    }
  }
  if (descriptors[0] >= 0) {
    ::close(descriptors[0]);
  }
  if (descriptors[1] >= 0) {
    ::close(descriptors[1]);
  }

  int wait_status = 0;
  pid_t waited = -1;
  do {
    waited = ::waitpid(child, &wait_status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited < 0) {
    throw std::runtime_error("could not wait for an official executable");
  }
  if (WIFEXITED(wait_status)) {
    result.status = WEXITSTATUS(wait_status);
  } else if (WIFSIGNALED(wait_status)) {
    result.status = 128 + WTERMSIG(wait_status);
  }
  return result;
}

void ExpectExit(const ProcessResult& result, int expected,
                const std::string& context) {
  if (result.status != expected) {
    std::ostringstream message;
    message << context << ": expected exit " << expected << " but observed "
            << result.status << "; stdout=[" << result.out << "] stderr=["
            << result.err << ']';
    Fail(__FILE__, __LINE__, message.str());
  }
}

void ExpectEmptyStdout(const ProcessResult& result,
                       const std::string& context) {
  if (!result.out.empty()) {
    Fail(__FILE__, __LINE__,
         context + ": negative result wrote to stdout: [" + result.out + ']');
  }
}

void ExpectNonEmptyDiagnostics(const ProcessResult& result,
                               const std::string& context) {
  if (result.err.empty()) {
    Fail(__FILE__, __LINE__,
         context + ": expected diagnostics on stderr, observed none");
  }
}

// Writes one raw FP32 image file and returns its path.
std::string WriteImages(const std::string& directory, const char* name,
                        const std::vector<float>& values) {
  MakeDirectory(directory);
  const std::string path = directory + "/" + name;
  WriteFloats(path, values);
  return path;
}

// Writes one raw int32 label file and returns its path.
std::string WriteLabels(const std::string& directory, const char* name,
                        const std::vector<std::int32_t>& values) {
  MakeDirectory(directory);
  const std::string path = directory + "/" + name;
  WriteInt32s(path, values);
  return path;
}

// Deterministic in-range raw images. kByte values keep fractional pixels and
// kNormalized values include negative samples, so a wrongly selected unit mode
// cannot consume them.
std::vector<float> PatternImages(std::uint32_t sample_count,
                                 course::InputMode input_mode) {
  std::vector<float> values(
      static_cast<std::size_t>(sample_count) * kImagePixels);
  for (std::size_t index = 0; index < values.size(); ++index) {
    switch (input_mode) {
      case course::InputMode::kUnit:
        values[index] =
            static_cast<float>((index * 29u + 7u) % 1001u) / 1000.0F;
        break;
      case course::InputMode::kByte:
        // Inclusive maximum 2550 / 10 == 255.0F; the modulus deliberately
        // spans every tenth so fractional pixels and the upper bound both
        // occur in a small fixture.
        values[index] =
            static_cast<float>((index * 37u + 13u) % 2551u) / 10.0F;
        break;
      case course::InputMode::kNormalized: {
        const std::int64_t centered =
            static_cast<std::int64_t>((index * 53u + 17u) % 1001u) - 500;
        values[index] = static_cast<float>(centered) / 1000.0F;
        break;
      }
    }
  }
  return values;
}

// Pins the generated fixture's mode bounds and coverage: unit/byte must stay
// inside their inclusive ranges with fractional values present, and normalized
// must include negative finite values.
void ExpectPatternBounds(const std::vector<float>& values,
                         course::InputMode input_mode,
                         const std::string& context) {
  bool fractional = false;
  bool negative = false;
  for (const float value : values) {
    if (!std::isfinite(value)) {
      Fail(__FILE__, __LINE__, context + ": fixture value is not finite");
    }
    switch (input_mode) {
      case course::InputMode::kUnit:
        if (value < 0.0F || value > 1.0F) {
          Fail(__FILE__, __LINE__,
               context + ": unit fixture value is outside [0, 1]");
        }
        fractional = fractional || value != std::trunc(value);
        break;
      case course::InputMode::kByte:
        if (value < 0.0F || value > 255.0F) {
          Fail(__FILE__, __LINE__,
               context + ": byte fixture value is outside [0, 255]");
        }
        fractional = fractional || value != std::trunc(value);
        break;
      case course::InputMode::kNormalized:
        if (value < -0.5F || value > 0.5F) {
          Fail(__FILE__, __LINE__,
               context + ": normalized fixture value is outside [-0.5, 0.5]");
        }
        negative = negative || value < 0.0F;
        break;
    }
  }
  if (input_mode == course::InputMode::kUnit ||
      input_mode == course::InputMode::kByte) {
    EXPECT_TRUE(fractional);
  }
  if (input_mode == course::InputMode::kNormalized) {
    EXPECT_TRUE(negative);
  }
}

// Cyclic valid labels for the bounded one-epoch training fixture.
std::vector<std::int32_t> CyclicLabels(std::uint32_t sample_count) {
  std::vector<std::int32_t> labels(sample_count);
  for (std::uint32_t index = 0; index < sample_count; ++index) {
    labels[index] = static_cast<std::int32_t>(index % kClassCount);
  }
  return labels;
}

Checkpoint BaseCheckpoint() {
  Checkpoint checkpoint;
  checkpoint.metadata = CheckpointMetadata{1, 0.0F, kNormalizationMean,
                                           kNormalizationStddev};
  checkpoint.parameters = CreateLenetParameters();
  return checkpoint;
}

// Canonical finite checkpoint using the original deterministic initializer.
Checkpoint InitializedCheckpoint() {
  Checkpoint checkpoint = BaseCheckpoint();
  InitializeLenetParameters(kFixtureSeed, &checkpoint.parameters);
  return checkpoint;
}

// Canonical checkpoint with every weight at `weight` and every bias at `bias`.
// The poison case supplies finite weights of one and zero biases.
Checkpoint UniformCheckpoint(float weight, float bias) {
  Checkpoint checkpoint = BaseCheckpoint();
  const std::array<ParameterSpec, 10>& specs = LenetParameterSpecs();
  for (std::size_t index = 0; index < checkpoint.parameters.size(); ++index) {
    const float value = specs[index].is_bias ? bias : weight;
    std::fill(checkpoint.parameters[index].values.begin(),
              checkpoint.parameters[index].values.end(), value);
  }
  return checkpoint;
}

// Deterministic batch-sensitive fixture with finite, format-valid tensors.
// Hidden convolutions/fc1/fc2 have weight 1 and bias 0. fc3 weights are 1 only
// on the class-0 row; fc3 class-1 bias is 1e8F. For normalized all-zero and
// all-one images the literal logits give predictions [1, 0]; a batch slice that
// starts one float late turns the second image into 783 zeros plus one 1 and
// changes its prediction. No host neural-network computation is performed.
Checkpoint BatchSensitiveCheckpoint() {
  Checkpoint checkpoint = BaseCheckpoint();
  const std::array<ParameterSpec, 10>& specs = LenetParameterSpecs();
  for (std::size_t index = 0; index < checkpoint.parameters.size(); ++index) {
    std::vector<float>& values = checkpoint.parameters[index].values;
    if (specs[index].is_bias) {
      std::fill(values.begin(), values.end(), 0.0F);
    } else {
      std::fill(values.begin(), values.end(), 1.0F);
    }
  }

  // fc3.weight: only the class-0 row (first 84 values) is 1; all other class
  // rows remain 0.
  std::vector<float>& fc3_weights = checkpoint.parameters[8].values;
  std::fill(fc3_weights.begin() + 84, fc3_weights.end(), 0.0F);
  // fc3.bias: class 1 gets 1e8F; every other class bias is 0.
  std::vector<float>& fc3_bias = checkpoint.parameters[9].values;
  fc3_bias[1] = 1.0e8F;
  return checkpoint;
}

// Publishes a real course model bundle in a build-owned directory.
void PublishModel(const std::string& directory, course::InputMode input_mode,
                  const Checkpoint& checkpoint) {
  MakeDirectory(directory);
  course::SaveCourseModel(directory, checkpoint, input_mode);
}

// Runs the official infer executable with exact positional arguments and the
// requested model/environment overrides.
ProcessResult RunInfer(const std::string& images_path,
                       std::uint32_t sample_count,
                       const std::string& model_dir,
                       const std::vector<EnvVar>& extra_environment,
                       const std::string& working_directory) {
  std::vector<EnvVar> environment;
  environment.push_back(EnvVar{"COURSE_MODEL_DIR", model_dir});
  environment.insert(environment.end(), extra_environment.begin(),
                     extra_environment.end());
  return RunProcess(OfficialExecutable("infer"),
                    {images_path, std::to_string(sample_count), "1", "28", "28",
                     "10"},
                    environment, working_directory);
}

// Splits captured diagnostics into individual lines without a global token
// search that could cross record boundaries.
std::vector<std::string> DiagnosticLines(const std::string& text) {
  std::vector<std::string> lines;
  std::string::size_type start = 0;
  while (start < text.size()) {
    const std::string::size_type end = text.find('\n', start);
    if (end == std::string::npos) {
      lines.push_back(text.substr(start));
      break;
    }
    lines.push_back(text.substr(start, end - start));
    start = end + 1;
  }
  return lines;
}

// Extracts the value of an exact whitespace-delimited key=value token from one
// line. Using complete tokens avoids matching "samples=" inside
// "batch_samples=".
bool TokenValue(const std::string& line, const std::string& key,
                std::string* value) {
  std::string::size_type start = 0;
  while (start < line.size()) {
    const std::string::size_type end = line.find(' ', start);
    const std::string token =
        line.substr(start, end == std::string::npos ? std::string::npos
                                                    : end - start);
    if (token.compare(0, key.size(), key) == 0) {
      *value = token.substr(key.size());
      return true;
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return false;
}

void ExpectToken(const std::string& line, const std::string& key,
                 const std::string& expected, const std::string& context) {
  std::string actual;
  if (!TokenValue(line, key, &actual)) {
    Fail(__FILE__, __LINE__,
         context + ": missing token " + key + " in line [" + line + ']');
  }
  if (actual != expected) {
    Fail(__FILE__, __LINE__, context + ": token " + key + " expected [" +
                               expected + "] but observed [" + actual +
                               "] in line [" + line + ']');
  }
}

double ParseDoubleToken(const std::string& line, const std::string& key,
                        const std::string& context) {
  std::string token;
  if (!TokenValue(line, key, &token)) {
    Fail(__FILE__, __LINE__,
         context + ": missing numeric token " + key + " in line [" + line +
             ']');
  }
  char* end = nullptr;
  const double value = std::strtod(token.c_str(), &end);
  if (end == token.c_str() || end == nullptr || *end != '\0' ||
      !std::isfinite(value) || value < 0.0) {
    Fail(__FILE__, __LINE__, context + ": invalid numeric token " + key +
                               " value [" + token + "] in line [" + line +
                               ']');
  }
  return value;
}

// Returns the unique line beginning with prefix, or fails with the entire
// captured diagnostics.
std::string SingleLineStarting(const std::string& text,
                               const std::string& prefix,
                               const std::string& context) {
  std::string found;
  std::size_t count = 0;
  for (const std::string& line : DiagnosticLines(text)) {
    if (line.compare(0, prefix.size(), prefix) == 0) {
      found = line;
      ++count;
    }
  }
  if (count != 1) {
    std::ostringstream message;
    message << context << ": expected exactly one line starting [" << prefix
            << "] but observed " << count << "; diagnostics=[" << text << ']';
    Fail(__FILE__, __LINE__, message.str());
  }
  return found;
}

std::vector<std::string> LinesStarting(const std::string& text,
                                       const std::string& prefix) {
  std::vector<std::string> matches;
  for (const std::string& line : DiagnosticLines(text)) {
    if (line.compare(0, prefix.size(), prefix) == 0) {
      matches.push_back(line);
    }
  }
  return matches;
}

// Requires the exact successful inference completion record.
void ExpectSuccessRecord(const std::string& diagnostics,
                         std::uint32_t sample_count, std::uint32_t batches,
                         course::InputMode input_mode,
                         const std::string& context) {
  const std::string line =
      SingleLineStarting(diagnostics, "event=course_infer status=pass ",
                         context);
  ExpectToken(line, "samples=", std::to_string(sample_count), context);
  ExpectToken(line, "batches=", std::to_string(batches), context);
  ExpectToken(line, "input_mode=", InputModeName(input_mode), context);
}

// Parses a successful stdout payload as exactly N canonical decimal class
// integers, one per line, each in [0, 10), with a final newline and nothing else.
std::vector<int> ParsePredictions(const std::string& text,
                                  std::size_t expected_count,
                                  const std::string& context) {
  if (text.empty()) {
    Fail(__FILE__, __LINE__, context + ": expected predictions, stdout empty");
  }
  if (text[text.size() - 1] != '\n') {
    Fail(__FILE__, __LINE__,
         context + ": prediction stdout has no final newline: [" + text + ']');
  }

  std::vector<int> predictions;
  std::string::size_type start = 0;
  while (start < text.size()) {
    const std::string::size_type end = text.find('\n', start);
    if (end == std::string::npos) {
      Fail(__FILE__, __LINE__,
           context + ": prediction line has no terminator: [" + text + ']');
    }
    const std::string line = text.substr(start, end - start);
    if (line.size() != 1 || line[0] < '0' || line[0] > '9') {
      Fail(__FILE__, __LINE__, context +
                                 ": non-canonical prediction line [" + line +
                                 "] in stdout [" + text + ']');
    }
    predictions.push_back(line[0] - '0');
    start = end + 1;
  }
  if (predictions.size() != expected_count) {
    Fail(__FILE__, __LINE__,
         context + ": expected " + std::to_string(expected_count) +
             " prediction lines but observed " +
             std::to_string(predictions.size()));
  }
  return predictions;
}

// Returns exactly the first count newline-terminated lines of text.
std::string FirstLines(const std::string& text, std::size_t count,
                       const std::string& context) {
  std::string::size_type start = 0;
  for (std::size_t line = 0; line < count; ++line) {
    const std::string::size_type end = text.find('\n', start);
    if (end == std::string::npos) {
      Fail(__FILE__, __LINE__, context + ": stdout has fewer than " +
                                 std::to_string(count) + " lines");
    }
    start = end + 1;
  }
  return text.substr(0, start);
}

// Builds a fully explicit direct InferOptions value.
course::InferOptions DirectOptions(const std::string& images_path,
                                   const std::string& model_dir,
                                   std::uint32_t sample_count,
                                   std::uint32_t batch_size) {
  course::InferOptions options;
  options.images_path = images_path;
  options.model_dir = model_dir;
  options.shape = FixtureShape(sample_count);
  options.batch_size = batch_size;
  options.profile = false;
  return options;
}

// Rewrites the persisted CRC field to a different eight-digit value without
// touching a valid checkpoint payload.
void TamperMetadataCrc(const std::string& directory) {
  const std::string path = directory + "/course.meta";
  std::string text = ReadText(path);
  const std::string key = "checkpoint_crc32=";
  const std::string::size_type position = text.find(key);
  if (position == std::string::npos || position + key.size() + 8 > text.size()) {
    Fail(__FILE__, __LINE__, "fixture metadata has no CRC field: " + path);
  }
  const std::string current = text.substr(position + key.size(), 8);
  const std::string replacement =
      current == "deadbeef" ? "feedface" : "deadbeef";
  text.replace(position + key.size(), 8, replacement);
  WriteText(path, text);
}

// Flips one payload byte so the checkpoint bytes no longer match the metadata
// CRC and may also violate the checkpoint format.
void CorruptCheckpointPayload(const std::string& directory) {
  const std::string path = directory + "/lenet.bin";
  std::vector<unsigned char> bytes = ReadBytes(path);
  if (bytes.empty()) {
    Fail(__FILE__, __LINE__, "fixture checkpoint is empty: " + path);
  }
  bytes[0] = static_cast<unsigned char>(bytes[0] ^ 0xffu);
  WriteBytes(path, bytes);
}

// ---------------------------------------------------------------------------
// Official train executable and persisted input mode
// ---------------------------------------------------------------------------

TEST_CASE(train_cli_bounded_epoch_persists_normalized_mode_for_infer) {
  const std::string root = MakeScratchDirectory("runtime_train_persist");
  const std::uint32_t sample_count = 4;
  const std::string images_path =
      WriteImages(root + "/data", "images.f32",
                  PatternImages(sample_count, course::InputMode::kNormalized));
  const std::string labels_path =
      WriteLabels(root + "/data", "labels.i32", CyclicLabels(sample_count));
  const std::string work_dir = root + "/work";
  EXPECT_FALSE(PathExists(work_dir));

  const ProcessResult trained = RunProcess(
      OfficialExecutable("train"),
      {images_path, labels_path, "4", "1", "28", "28", "10", work_dir},
      {EnvVar{"COURSE_EPOCHS", "1"},
       EnvVar{"COURSE_INPUT_MODE", "normalized"}},
      root);
  ExpectExit(trained, 0, "bounded one-epoch train");
  ExpectEmptyStdout(trained, "bounded one-epoch train");
  ExpectNonEmptyDiagnostics(trained, "bounded one-epoch train");

  const std::string completion = SingleLineStarting(
      trained.err, "event=course_train status=pass ", "bounded one-epoch train");
  ExpectToken(completion, "epochs=", "1", "bounded one-epoch train");
  ExpectToken(completion, "input_mode=", "normalized",
              "bounded one-epoch train");
  const std::vector<std::string> epoch_lines =
      LinesStarting(trained.err, "event=course_train_epoch ");
  EXPECT_EQ(static_cast<std::size_t>(1), epoch_lines.size());
  if (epoch_lines.size() == 1) {
    ExpectToken(epoch_lines[0], "epoch=", "1", "bounded one-epoch train");
  }

  EXPECT_TRUE(PathExists(work_dir + "/lenet.bin"));
  EXPECT_TRUE(PathExists(work_dir + "/course.meta"));
  const course::CourseModel bundle = course::LoadCourseModel(work_dir);
  EXPECT_EQ(course::InputMode::kNormalized, bundle.input_mode);
  EXPECT_EQ(static_cast<std::size_t>(10), bundle.checkpoint.parameters.size());
  for (const ParameterTensor& tensor : bundle.checkpoint.parameters) {
    for (const float value : tensor.values) {
      EXPECT_TRUE(std::isfinite(value));
    }
  }

  // Inference must use the mode persisted by training. The fixture contains
  // negative normalized pixels that unit mode would reject, and conflicting
  // training variables must be ignored entirely.
  const ProcessResult inferred =
      RunInfer(images_path, sample_count, work_dir,
               {EnvVar{"COURSE_INPUT_MODE", "unit"},
                EnvVar{"COURSE_EPOCHS", "1000"}},
               root);
  ExpectExit(inferred, 0, "infer from trained normalized bundle");
  ParsePredictions(inferred.out, sample_count,
                   "infer from trained normalized bundle");
  ExpectSuccessRecord(inferred.err, sample_count, 1,
                      course::InputMode::kNormalized,
                      "infer from trained normalized bundle");
}

TEST_CASE(train_cli_usage_errors_exit_two_without_stdout) {
  const std::string root = MakeScratchDirectory("runtime_train_usage");
  const std::string train = OfficialExecutable("train");

  const ProcessResult no_arguments = RunProcess(train, {}, {}, root);
  ExpectExit(no_arguments, 2, "train with no arguments");
  ExpectEmptyStdout(no_arguments, "train with no arguments");
  ExpectNonEmptyDiagnostics(no_arguments, "train with no arguments");

  // argc 8: images, labels, N, C, H, W, K present; work_dir missing.
  const ProcessResult short_argv =
      RunProcess(train,
                 {"images.f32", "labels.i32", "2", "1", "28", "28", "10"},
                 {}, root);
  ExpectExit(short_argv, 2, "train with argc 8");
  ExpectEmptyStdout(short_argv, "train with argc 8");
  ExpectNonEmptyDiagnostics(short_argv, "train with argc 8");

  const ProcessResult single_sample = RunProcess(
      train,
      {"images.f32", "labels.i32", "1", "1", "28", "28", "10", "work"},
      {EnvVar{"COURSE_EPOCHS", "1"}}, root);
  ExpectExit(single_sample, 2, "train with N=1");
  ExpectEmptyStdout(single_sample, "train with N=1");
  ExpectNonEmptyDiagnostics(single_sample, "train with N=1");

  // argc 10: every official field is present plus one trailing argument.
  const ProcessResult extra_argv = RunProcess(
      train,
      {"images.f32", "labels.i32", "2", "1", "28", "28", "10", "work",
       "extra"},
      {EnvVar{"COURSE_EPOCHS", "1"}}, root);
  ExpectExit(extra_argv, 2, "train with an extra argument");
  ExpectEmptyStdout(extra_argv, "train with an extra argument");
  ExpectNonEmptyDiagnostics(extra_argv, "train with an extra argument");
}

// ---------------------------------------------------------------------------
// Official infer executable: usage, model validation, and input modes
// ---------------------------------------------------------------------------

TEST_CASE(infer_requires_exact_argv_and_course_model_dir) {
  const std::string root = MakeScratchDirectory("runtime_infer_usage");
  const std::string infer = OfficialExecutable("infer");

  const ProcessResult no_arguments = RunProcess(infer, {}, {}, root);
  ExpectExit(no_arguments, 2, "infer with no arguments");
  ExpectEmptyStdout(no_arguments, "infer with no arguments");
  ExpectNonEmptyDiagnostics(no_arguments, "infer with no arguments");

  const std::vector<std::string> exact_argv{"images.f32", "1", "1", "28", "28",
                                            "10"};
  const ProcessResult missing_model =
      RunProcess(infer, exact_argv, {}, root);
  ExpectExit(missing_model, 2, "infer without COURSE_MODEL_DIR");
  ExpectEmptyStdout(missing_model, "infer without COURSE_MODEL_DIR");
  ExpectNonEmptyDiagnostics(missing_model, "infer without COURSE_MODEL_DIR");

  const ProcessResult empty_model =
      RunProcess(infer, exact_argv, {EnvVar{"COURSE_MODEL_DIR", ""}}, root);
  ExpectExit(empty_model, 2, "infer with empty COURSE_MODEL_DIR");
  ExpectEmptyStdout(empty_model, "infer with empty COURSE_MODEL_DIR");
  ExpectNonEmptyDiagnostics(empty_model, "infer with empty COURSE_MODEL_DIR");

  const ProcessResult extra_argument = RunProcess(
      infer,
      {"images.f32", "1", "1", "28", "28", "10", "labels.i32"},
      {EnvVar{"COURSE_MODEL_DIR", root + "/model"}}, root);
  ExpectExit(extra_argument, 2, "infer with a labels argument");
  ExpectEmptyStdout(extra_argument, "infer with a labels argument");
  ExpectNonEmptyDiagnostics(extra_argument, "infer with a labels argument");
}

TEST_CASE(infer_missing_malformed_and_bad_crc_models_write_no_stdout) {
  const std::string root = MakeScratchDirectory("runtime_infer_bad_models");
  const std::string images_path =
      WriteImages(root + "/data", "images.f32",
                  PatternImages(2, course::InputMode::kUnit));

  const std::string missing_model = root + "/missing_model";
  PublishModel(root + "/malformed_meta", course::InputMode::kUnit,
               InitializedCheckpoint());
  const std::string malformed_model = root + "/malformed_meta";
  WriteText(malformed_model + "/course.meta", "not_a_course_model\n");

  PublishModel(root + "/bad_crc", course::InputMode::kUnit,
               InitializedCheckpoint());
  const std::string bad_crc_model = root + "/bad_crc";
  TamperMetadataCrc(bad_crc_model);

  PublishModel(root + "/corrupt_checkpoint", course::InputMode::kUnit,
               InitializedCheckpoint());
  const std::string corrupt_model = root + "/corrupt_checkpoint";
  CorruptCheckpointPayload(corrupt_model);

  const std::vector<std::string> models{missing_model, malformed_model,
                                        bad_crc_model, corrupt_model};
  const std::vector<std::string> labels{"missing model directory",
                                        "malformed metadata", "bad CRC",
                                        "corrupt checkpoint bytes"};
  for (std::size_t index = 0; index < models.size(); ++index) {
    const ProcessResult result =
        RunInfer(images_path, 2, models[index],
                 {EnvVar{"COURSE_INFER_BATCH", "1"}}, root);
    ExpectExit(result, 1, labels[index]);
    ExpectEmptyStdout(result, labels[index]);
    ExpectNonEmptyDiagnostics(result, labels[index]);
  }
}

TEST_CASE(infer_uses_persisted_mode_and_ignores_course_input_mode) {
  const std::string root = MakeScratchDirectory("runtime_infer_modes");
  struct ModeCase {
    course::InputMode mode;
    const char* spelling;
    const char* conflicting_environment;
  };
  const ModeCase cases[] = {
      {course::InputMode::kUnit, "unit", "normalized"},
      {course::InputMode::kByte, "byte", "unit"},
      {course::InputMode::kNormalized, "normalized", "byte"},
  };

  for (const ModeCase& mode_case : cases) {
    const std::string case_root =
        root + "/" + std::string(mode_case.spelling);
    // MakeScratchDirectory created only root; create this owned case directory
    // before PublishModel creates its single model directory level.
    MakeDirectory(case_root);
    const std::vector<float> images =
        PatternImages(5, mode_case.mode);
    ExpectPatternBounds(images, mode_case.mode,
                        std::string("fixture ") + mode_case.spelling);
    const std::string model_dir = case_root + "/model";
    PublishModel(model_dir, mode_case.mode, InitializedCheckpoint());
    const std::string images_path =
        WriteImages(case_root + "/data", "images.f32", images);

    const ProcessResult baseline =
        RunInfer(images_path, 5, model_dir, {}, root);
    ExpectExit(baseline, 0,
               std::string("infer with persisted ") + mode_case.spelling);
    ParsePredictions(baseline.out, 5,
                     std::string("infer with persisted ") +
                         mode_case.spelling);
    ExpectSuccessRecord(baseline.err, 5, 1, mode_case.mode,
                        std::string("infer with persisted ") +
                            mode_case.spelling);

    // A conflicting training-mode variable must not change the selected mode
    // or the resulting bytes.
    const ProcessResult conflicted = RunInfer(
        images_path, 5, model_dir,
        {EnvVar{"COURSE_INPUT_MODE", mode_case.conflicting_environment},
         EnvVar{"COURSE_EPOCHS", "999"}},
        root);
    ExpectExit(conflicted, 0,
               std::string("infer ignores COURSE_INPUT_MODE for ") +
                   mode_case.spelling);
    ParsePredictions(conflicted.out, 5,
                     std::string("infer ignores COURSE_INPUT_MODE for ") +
                         mode_case.spelling);
    ExpectSuccessRecord(conflicted.err, 5, 1, mode_case.mode,
                        std::string("infer ignores COURSE_INPUT_MODE for ") +
                            mode_case.spelling);
    EXPECT_EQ(baseline.out, conflicted.out);
  }
}

// ---------------------------------------------------------------------------
// N=1 / N=17 / N=1025 output contract, ordering, and batching
// ---------------------------------------------------------------------------

TEST_CASE(infer_n1_17_1025_pure_stdout_preserves_input_order) {
  const std::string root = MakeScratchDirectory("runtime_infer_counts");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kNormalized,
               InitializedCheckpoint());

  const std::uint32_t full_count = 1025;
  const std::uint32_t prefix_count = 17;
  const std::vector<float> full_images =
      PatternImages(full_count, course::InputMode::kNormalized);
  const std::string full_path =
      WriteImages(root + "/data", "full.f32", full_images);
  const std::vector<float> prefix_images(
      full_images.begin(),
      full_images.begin() + static_cast<std::size_t>(prefix_count) *
                                kImagePixels);
  const std::string prefix_path =
      WriteImages(root + "/data", "prefix.f32", prefix_images);
  const std::vector<float> single_images(
      full_images.begin(),
      full_images.begin() + static_cast<std::size_t>(kImagePixels));
  const std::string single_path =
      WriteImages(root + "/data", "single.f32", single_images);

  // Raw files carry exactly the declared sample count, so the N=1 request
  // cannot pass by accepting a longer 1025-image file.
  EXPECT_EQ(static_cast<std::size_t>(1) * kImagePixels * sizeof(float),
            ReadBytes(single_path).size());
  EXPECT_EQ(static_cast<std::size_t>(prefix_count) * kImagePixels *
                sizeof(float),
            ReadBytes(prefix_path).size());
  EXPECT_EQ(static_cast<std::size_t>(full_count) * kImagePixels *
                sizeof(float),
            ReadBytes(full_path).size());

  const ProcessResult one = RunInfer(single_path, 1, model_dir, {}, root);
  ExpectExit(one, 0, "infer N=1");
  ParsePredictions(one.out, 1, "infer N=1");
  ExpectSuccessRecord(one.err, 1, 1, course::InputMode::kNormalized,
                      "infer N=1");

  const ProcessResult seventeen =
      RunInfer(prefix_path, prefix_count, model_dir, {}, root);
  ExpectExit(seventeen, 0, "infer N=17");
  ParsePredictions(seventeen.out, prefix_count, "infer N=17");
  ExpectSuccessRecord(seventeen.err, prefix_count, 1,
                      course::InputMode::kNormalized, "infer N=17");

  const ProcessResult full = RunInfer(full_path, full_count, model_dir, {}, root);
  ExpectExit(full, 0, "infer N=1025");
  ParsePredictions(full.out, full_count, "infer N=1025");
  ExpectSuccessRecord(full.err, full_count, 2,
                      course::InputMode::kNormalized, "infer N=1025");
  EXPECT_TRUE(full.err.find("accuracy") == std::string::npos);

  // The first 17 images must remain the first 17 predictions regardless of
  // total request size.
  EXPECT_EQ(FirstLines(full.out, prefix_count, "infer N=1025"),
            seventeen.out);

  // A small configured capacity must preserve every prediction and its order.
  const ProcessResult batched =
      RunInfer(full_path, full_count, model_dir,
               {EnvVar{"COURSE_INFER_BATCH", "17"}}, root);
  ExpectExit(batched, 0, "infer N=1025 batch 17");
  ParsePredictions(batched.out, full_count, "infer N=1025 batch 17");
  ExpectSuccessRecord(batched.err, full_count, 61,
                      course::InputMode::kNormalized,
                      "infer N=1025 batch 17");
  EXPECT_EQ(full.out, batched.out);
}

TEST_CASE(infer_batch_partitions_keep_each_sample_input) {
  const std::string root = MakeScratchDirectory("runtime_infer_batch_partition");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kNormalized,
               BatchSensitiveCheckpoint());

  // The persisted bundle stays format-valid and every parameter finite.
  const course::CourseModel bundle = course::LoadCourseModel(model_dir);
  EXPECT_EQ(course::InputMode::kNormalized, bundle.input_mode);
  for (const ParameterTensor& tensor : bundle.checkpoint.parameters) {
    for (const float value : tensor.values) {
      EXPECT_TRUE(std::isfinite(value));
    }
  }

  // Two normalized raw images: all zero, then all one. Both are finite and
  // valid; the oracle is the GPU CLI output, never host NN arithmetic.
  std::vector<float> images(
      static_cast<std::size_t>(2) * kImagePixels, 0.0F);
  std::fill(images.begin() + kImagePixels, images.end(), 1.0F);
  const std::string images_path =
      WriteImages(root + "/data", "pair.f32", images);
  const std::vector<float> read_back = course::ReadImages(
      images_path, FixtureShape(2), course::InputMode::kNormalized);
  EXPECT_EQ(static_cast<std::size_t>(2) * kImagePixels, read_back.size());
  EXPECT_TRUE(SameBits(0.0F, read_back[0]));
  EXPECT_TRUE(SameBits(1.0F, read_back[kImagePixels]));

  // Correct per-sample slicing yields [1, 0] whether the two samples are
  // processed as two batches or as one batch.
  const ProcessResult split =
      RunInfer(images_path, 2, model_dir,
               {EnvVar{"COURSE_INFER_BATCH", "1"}}, root);
  ExpectExit(split, 0, "batch-sensitive infer B=1");
  const std::vector<int> split_predictions =
      ParsePredictions(split.out, 2, "batch-sensitive infer B=1");
  EXPECT_EQ(static_cast<std::size_t>(2), split_predictions.size());
  EXPECT_EQ(1, split_predictions[0]);
  EXPECT_EQ(0, split_predictions[1]);
  ExpectSuccessRecord(split.err, 2, 2, course::InputMode::kNormalized,
                      "batch-sensitive infer B=1");

  const ProcessResult joined =
      RunInfer(images_path, 2, model_dir,
               {EnvVar{"COURSE_INFER_BATCH", "2"}}, root);
  ExpectExit(joined, 0, "batch-sensitive infer B=2");
  const std::vector<int> joined_predictions =
      ParsePredictions(joined.out, 2, "batch-sensitive infer B=2");
  EXPECT_EQ(static_cast<std::size_t>(2), joined_predictions.size());
  EXPECT_EQ(1, joined_predictions[0]);
  EXPECT_EQ(0, joined_predictions[1]);
  ExpectSuccessRecord(joined.err, 2, 1, course::InputMode::kNormalized,
                      "batch-sensitive infer B=2");
  EXPECT_EQ(split.out, joined.out);
}

TEST_CASE(infer_needs_no_labels_and_rejects_an_extra_labels_argument) {
  const std::string root = MakeScratchDirectory("runtime_infer_no_labels");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kUnit, InitializedCheckpoint());
  const std::string data_dir = root + "/data";
  const std::string images_path =
      WriteImages(data_dir, "images.f32",
                  PatternImages(3, course::InputMode::kUnit));

  // The data directory contains images only.
  EXPECT_TRUE(PathExists(images_path));
  EXPECT_FALSE(PathExists(data_dir + "/labels.i32"));
  const ProcessResult inferred = RunInfer(images_path, 3, model_dir, {}, root);
  ExpectExit(inferred, 0, "infer without any labels file");
  ParsePredictions(inferred.out, 3, "infer without any labels file");
  ExpectSuccessRecord(inferred.err, 3, 1, course::InputMode::kUnit,
                      "infer without any labels file");

  const ProcessResult with_labels = RunProcess(
      OfficialExecutable("infer"),
      {images_path, "3", "1", "28", "28", "10", data_dir + "/labels.i32"},
      {EnvVar{"COURSE_MODEL_DIR", model_dir}}, root);
  ExpectExit(with_labels, 2, "infer with an unexpected labels argument");
  ExpectEmptyStdout(with_labels, "infer with an unexpected labels argument");
  ExpectNonEmptyDiagnostics(with_labels,
                            "infer with an unexpected labels argument");
}

TEST_CASE(infer_batch_option_accepts_only_one_through_16384) {
  const std::string root = MakeScratchDirectory("runtime_infer_batch_bounds");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kNormalized,
               InitializedCheckpoint());
  const std::string images_path =
      WriteImages(root + "/data", "images.f32",
                  PatternImages(17, course::InputMode::kNormalized));

  const ProcessResult maximum =
      RunInfer(images_path, 17, model_dir,
               {EnvVar{"COURSE_INFER_BATCH",
                       std::to_string(kMaximumInferBatch)}},
               root);
  ExpectExit(maximum, 0, "infer batch 16384");
  ParsePredictions(maximum.out, 17, "infer batch 16384");
  ExpectSuccessRecord(maximum.err, 17, 1, course::InputMode::kNormalized,
                      "infer batch 16384");

  const std::vector<std::string> rejected{"0", "16385", "-1", "abc"};
  for (const std::string& batch : rejected) {
    const ProcessResult result = RunInfer(
        images_path, 17, model_dir, {EnvVar{"COURSE_INFER_BATCH", batch}}, root);
    ExpectExit(result, 2, "infer rejected batch [" + batch + "]");
    ExpectEmptyStdout(result, "infer rejected batch [" + batch + "]");
    ExpectNonEmptyDiagnostics(result, "infer rejected batch [" + batch + "]");
  }

  // The default capacity is 1024, so N=17 remains one batch.
  const ProcessResult default_batch =
      RunInfer(images_path, 17, model_dir, {}, root);
  ExpectExit(default_batch, 0, "infer default batch");
  ExpectSuccessRecord(default_batch.err, 17, 1,
                      course::InputMode::kNormalized, "infer default batch");
  EXPECT_EQ(static_cast<std::uint32_t>(kDefaultInferBatch), 1024u);
}

// ---------------------------------------------------------------------------
// Profiling includes the first batch; the fast path has no profile records
// ---------------------------------------------------------------------------

TEST_CASE(infer_profile_reports_every_batch_including_the_first) {
  const std::string root = MakeScratchDirectory("runtime_infer_profile");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kNormalized,
               InitializedCheckpoint());
  const std::string images_path =
      WriteImages(root + "/data", "images.f32",
                  PatternImages(17, course::InputMode::kNormalized));

  const ProcessResult profiled =
      RunInfer(images_path, 17, model_dir,
               {EnvVar{"COURSE_PROFILE", "1"},
                EnvVar{"COURSE_INFER_BATCH", "7"}},
               root);
  ExpectExit(profiled, 0, "infer with COURSE_PROFILE=1");
  ParsePredictions(profiled.out, 17, "infer with COURSE_PROFILE=1");
  ExpectSuccessRecord(profiled.err, 17, 3, course::InputMode::kNormalized,
                      "infer with COURSE_PROFILE=1");

  const std::vector<std::string> batch_lines =
      LinesStarting(profiled.err, "event=course_profile_batch ");
  EXPECT_EQ(static_cast<std::size_t>(3), batch_lines.size());
  double h2d_sum = 0.0;
  double prepare_sum = 0.0;
  double forward_sum = 0.0;
  std::uint32_t counted_samples = 0;
  for (std::size_t index = 0; index < batch_lines.size(); ++index) {
    const std::string& line = batch_lines[index];
    const std::string expected_samples =
        index < 2 ? "7" : "3";
    ExpectToken(line, "index=", std::to_string(index + 1),
                "profile batch records");
    ExpectToken(line, "batch_samples=", expected_samples,
                "profile batch records");
    counted_samples +=
        static_cast<std::uint32_t>(std::strtoul(expected_samples.c_str(),
                                                nullptr, 10));
    h2d_sum += ParseDoubleToken(line, "h2d_ms=", "profile batch records");
    prepare_sum +=
        ParseDoubleToken(line, "prepare_ms=", "profile batch records");
    forward_sum +=
        ParseDoubleToken(line, "forward_argmax_ms=", "profile batch records");
  }
  EXPECT_EQ(17u, counted_samples);

  const std::string summary = SingleLineStarting(
      profiled.err, "event=course_profile ", "profile summary");
  ExpectToken(summary, "profile_batches=", "3", "profile summary");
  EXPECT_TRUE(std::fabs(ParseDoubleToken(summary, "gpu_h2d_ms=",
                                         "profile summary") -
                        h2d_sum) <= 0.01);
  EXPECT_TRUE(std::fabs(ParseDoubleToken(summary, "gpu_prepare_ms=",
                                         "profile summary") -
                        prepare_sum) <= 0.01);
  EXPECT_TRUE(std::fabs(ParseDoubleToken(summary, "gpu_forward_argmax_ms=",
                                         "profile summary") -
                        forward_sum) <= 0.01);

  // Every required host wall bucket is reported, and enqueue is explicitly a
  // host bucket rather than device execution.
  const std::vector<std::string> host_keys{
      "host_load_model_ms=",          "host_load_images_ms=",
      "host_cuda_setup_ms=",          "host_enqueue_batches_ms=",
      "host_readback_sync_ms=",       "host_write_predictions_ms="};
  for (const std::string& key : host_keys) {
    static_cast<void>(ParseDoubleToken(summary, key, "profile summary"));
  }
  EXPECT_TRUE(summary.find("gpu_enqueue") == std::string::npos);
  EXPECT_TRUE(profiled.err.find("event=course_profile") != std::string::npos);
}

TEST_CASE(infer_default_fast_path_creates_no_profile_records) {
  const std::string root = MakeScratchDirectory("runtime_infer_fast_path");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kNormalized,
               InitializedCheckpoint());
  const std::string images_path =
      WriteImages(root + "/data", "images.f32",
                  PatternImages(17, course::InputMode::kNormalized));

  const ProcessResult inferred = RunInfer(images_path, 17, model_dir, {}, root);
  ExpectExit(inferred, 0, "infer default fast path");
  ParsePredictions(inferred.out, 17, "infer default fast path");
  ExpectSuccessRecord(inferred.err, 17, 1, course::InputMode::kNormalized,
                      "infer default fast path");
  EXPECT_TRUE(inferred.err.find("event=course_profile") == std::string::npos);
  EXPECT_TRUE(inferred.err.find("course_profile_batch") == std::string::npos);
  EXPECT_TRUE(inferred.err.find("_ms=") == std::string::npos);
}

// ---------------------------------------------------------------------------
// Valid finite poison model: early non-finite batch, later finite batch
// ---------------------------------------------------------------------------

TEST_CASE(infer_poisoned_early_batch_fails_while_later_batch_stays_finite) {
  const std::string root = MakeScratchDirectory("runtime_infer_poison");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kNormalized,
               UniformCheckpoint(1.0F, 0.0F));

  // The persisted bundle and every parameter remain valid and finite.
  const course::CourseModel bundle = course::LoadCourseModel(model_dir);
  EXPECT_EQ(course::InputMode::kNormalized, bundle.input_mode);
  for (const ParameterTensor& tensor : bundle.checkpoint.parameters) {
    for (const float value : tensor.values) {
      EXPECT_TRUE(std::isfinite(value));
    }
  }

  const float extreme = std::numeric_limits<float>::max();
  EXPECT_TRUE(std::isfinite(extreme));
  const std::vector<float> early(kImagePixels, extreme);
  const std::vector<float> later(kImagePixels, 0.0F);
  std::vector<float> pair = early;
  pair.insert(pair.end(), later.begin(), later.end());

  const std::string early_path =
      WriteImages(root + "/data", "early.f32", early);
  const std::string later_path =
      WriteImages(root + "/data", "later.f32", later);
  const std::string pair_path = WriteImages(root + "/data", "pair.f32", pair);

  // The input itself is valid host data in normalized mode; failure comes from
  // the first GPU batch output, not from a corrupt model or rejected input.
  const std::vector<float> read_back =
      course::ReadImages(pair_path, FixtureShape(2),
                         course::InputMode::kNormalized);
  EXPECT_EQ(static_cast<std::size_t>(2) * kImagePixels, read_back.size());
  EXPECT_TRUE(SameBits(extreme, read_back[0]));
  EXPECT_TRUE(SameBits(0.0F, read_back[kImagePixels]));

  const std::vector<EnvVar> batch_one{EnvVar{"COURSE_INFER_BATCH", "1"}};

  // Early batch non-finite: the whole request fails with zero stdout.
  const ProcessResult poisoned =
      RunInfer(pair_path, 2, model_dir, batch_one, root);
  ExpectExit(poisoned, 1, "poisoned first batch");
  ExpectEmptyStdout(poisoned, "poisoned first batch");
  ExpectNonEmptyDiagnostics(poisoned, "poisoned first batch");

  const ProcessResult early_alone =
      RunInfer(early_path, 1, model_dir, batch_one, root);
  ExpectExit(early_alone, 1, "extreme image alone");
  ExpectEmptyStdout(early_alone, "extreme image alone");
  ExpectNonEmptyDiagnostics(early_alone, "extreme image alone");

  // Later zero image alone proves that the second batch is finite, so a cleared
  // sticky flag or truncated error handling would be observable.
  const ProcessResult later_alone =
      RunInfer(later_path, 1, model_dir, batch_one, root);
  ExpectExit(later_alone, 0, "later finite image alone");
  ParsePredictions(later_alone.out, 1, "later finite image alone");
  ExpectSuccessRecord(later_alone.err, 1, 1,
                      course::InputMode::kNormalized,
                      "later finite image alone");
}

// ---------------------------------------------------------------------------
// Direct RunCourseInfer contract: buffering, batching, and exception taxonomy
// ---------------------------------------------------------------------------

TEST_CASE(run_course_infer_direct_buffers_output_and_rejects_bad_options) {
  const std::string root = MakeScratchDirectory("runtime_direct_success");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kUnit, InitializedCheckpoint());
  const std::string images_path =
      WriteImages(root + "/data", "images.f32",
                  PatternImages(17, course::InputMode::kUnit));

  course::InferOptions options = DirectOptions(images_path, model_dir, 17, 7);
  std::ostringstream predictions;
  std::ostringstream diagnostics;
  EXPECT_EQ(0, course::RunCourseInfer(options, predictions, diagnostics));
  ParsePredictions(predictions.str(), 17, "direct infer N=17 batch 7");
  ExpectSuccessRecord(diagnostics.str(), 17, 3, course::InputMode::kUnit,
                      "direct infer N=17 batch 7");

  std::ostringstream geometry_predictions;
  std::ostringstream geometry_diagnostics;
  course::InferOptions bad_geometry = options;
  bad_geometry.shape.c = 3;
  EXPECT_THROWS_AS(course::RunCourseInfer(bad_geometry, geometry_predictions,
                                          geometry_diagnostics),
                   std::invalid_argument);
  EXPECT_TRUE(geometry_predictions.str().empty());

  std::ostringstream zero_batch_predictions;
  std::ostringstream zero_batch_diagnostics;
  course::InferOptions zero_batch = options;
  zero_batch.batch_size = 0;
  EXPECT_THROWS_AS(course::RunCourseInfer(zero_batch, zero_batch_predictions,
                                          zero_batch_diagnostics),
                   std::invalid_argument);
  EXPECT_TRUE(zero_batch_predictions.str().empty());

  std::ostringstream huge_batch_predictions;
  std::ostringstream huge_batch_diagnostics;
  course::InferOptions huge_batch = options;
  huge_batch.batch_size = kMaximumInferBatch + 1;
  EXPECT_THROWS_AS(course::RunCourseInfer(huge_batch, huge_batch_predictions,
                                          huge_batch_diagnostics),
                   std::invalid_argument);
  EXPECT_TRUE(huge_batch_predictions.str().empty());

  std::ostringstream empty_model_predictions;
  std::ostringstream empty_model_diagnostics;
  course::InferOptions empty_model = options;
  empty_model.model_dir.clear();
  EXPECT_THROWS_AS(course::RunCourseInfer(empty_model, empty_model_predictions,
                                          empty_model_diagnostics),
                   std::invalid_argument);
  EXPECT_TRUE(empty_model_predictions.str().empty());

  std::ostringstream zero_count_predictions;
  std::ostringstream zero_count_diagnostics;
  course::InferOptions zero_count = options;
  zero_count.shape.n = 0;
  EXPECT_THROWS_AS(course::RunCourseInfer(zero_count, zero_count_predictions,
                                          zero_count_diagnostics),
                   std::invalid_argument);
  EXPECT_TRUE(zero_count_predictions.str().empty());

  std::ostringstream missing_model_predictions;
  std::ostringstream missing_model_diagnostics;
  course::InferOptions missing_model = options;
  missing_model.model_dir = root + "/missing_model";
  EXPECT_THROWS_AS(course::RunCourseInfer(missing_model,
                                          missing_model_predictions,
                                          missing_model_diagnostics),
                   std::runtime_error);
  EXPECT_TRUE(missing_model_predictions.str().empty());

  std::ostringstream missing_images_predictions;
  std::ostringstream missing_images_diagnostics;
  course::InferOptions missing_images = options;
  missing_images.images_path = root + "/missing_images.f32";
  EXPECT_THROWS_AS(course::RunCourseInfer(missing_images,
                                          missing_images_predictions,
                                          missing_images_diagnostics),
                   std::runtime_error);
  EXPECT_TRUE(missing_images_predictions.str().empty());
}

TEST_CASE(run_course_infer_direct_keeps_poison_failure_out_of_predictions) {
  const std::string root = MakeScratchDirectory("runtime_direct_poison");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kNormalized,
               UniformCheckpoint(1.0F, 0.0F));
  const float extreme = std::numeric_limits<float>::max();
  const std::vector<float> early(kImagePixels, extreme);
  const std::vector<float> later(kImagePixels, 0.0F);
  std::vector<float> pair = early;
  pair.insert(pair.end(), later.begin(), later.end());
  const std::string pair_path =
      WriteImages(root + "/data", "pair.f32", pair);
  const std::string early_path =
      WriteImages(root + "/data", "early.f32", early);
  const std::string later_path =
      WriteImages(root + "/data", "later.f32", later);

  std::ostringstream poisoned_predictions;
  std::ostringstream poisoned_diagnostics;
  EXPECT_THROWS_AS(course::RunCourseInfer(
                       DirectOptions(pair_path, model_dir, 2, 1),
                       poisoned_predictions, poisoned_diagnostics),
                   std::runtime_error);
  EXPECT_TRUE(poisoned_predictions.str().empty());

  std::ostringstream early_predictions;
  std::ostringstream early_diagnostics;
  EXPECT_THROWS_AS(course::RunCourseInfer(
                       DirectOptions(early_path, model_dir, 1, 1),
                       early_predictions, early_diagnostics),
                   std::runtime_error);
  EXPECT_TRUE(early_predictions.str().empty());

  std::ostringstream later_predictions;
  std::ostringstream later_diagnostics;
  EXPECT_EQ(0, course::RunCourseInfer(
                   DirectOptions(later_path, model_dir, 1, 1),
                   later_predictions, later_diagnostics));
  ParsePredictions(later_predictions.str(), 1, "direct later finite image");
  ExpectSuccessRecord(later_diagnostics.str(), 1, 1,
                      course::InputMode::kNormalized,
                      "direct later finite image");
}

TEST_CASE(run_course_infer_reports_buffered_sink_failures) {
  // Keep this last: all fork-based subprocess cases must run before the parent
  // initializes CUDA through this direct-API case.
  const std::string root = MakeScratchDirectory("runtime_direct_sink_failure");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kUnit, InitializedCheckpoint());
  const std::string images_path =
      WriteImages(root + "/data", "images.f32",
                  PatternImages(4, course::InputMode::kUnit));
  const course::InferOptions options = DirectOptions(images_path, model_dir, 4, 2);

  // A real buffered prediction sink: the small payload is accepted into the
  // std::ofstream buffer and only /dev/full rejects it at the explicit flush.
  std::ofstream failing_predictions("/dev/full",
                                    std::ios::out | std::ios::binary);
  if (!failing_predictions.is_open()) {
    Fail(__FILE__, __LINE__,
         "could not open /dev/full for the prediction sink test");
  }
  std::ostringstream prediction_failure_diagnostics;
  EXPECT_THROWS_AS(course::RunCourseInfer(options, failing_predictions,
                                          prediction_failure_diagnostics),
                   std::runtime_error);
  EXPECT_TRUE(prediction_failure_diagnostics.str().find(
                  "event=course_infer status=pass") == std::string::npos);

  // A real buffered diagnostics sink. Predictions may already be validly
  // published before this failure; rollback is deliberately not asserted.
  std::ostringstream published_predictions;
  std::ofstream failing_diagnostics("/dev/full",
                                    std::ios::out | std::ios::binary);
  if (!failing_diagnostics.is_open()) {
    Fail(__FILE__, __LINE__,
         "could not open /dev/full for the diagnostics sink test");
  }
  EXPECT_THROWS_AS(course::RunCourseInfer(options, published_predictions,
                                          failing_diagnostics),
                   std::runtime_error);
  ParsePredictions(published_predictions.str(), 4,
                   "diagnostics sink failure");
}

}  // namespace
