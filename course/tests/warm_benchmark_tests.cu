// Tests-first contract for the optional warm full-request benchmark tool.
//
// These cases are written before course/tools/warm_benchmark.cu and its CMake
// target exist. Until the tool is built, every case fails by name at the
// build-adjacent `course_warm_benchmark` entry point. A missing tool is never
// reported as a hardware failure, and no case falls back to CPU neural-network
// arithmetic.
//
// Each case runs the real executable in a fresh subprocess with an explicitly
// built environment (all inherited COURSE_* variables removed), a build-owned
// scratch working directory, and absolute fixture paths. Fixtures are tiny raw
// FP32 arrays and real course model bundles published with SaveCourseModel; the
// official `infer` executable is the only oracle for non-literal models. This
// translation unit makes no CUDA call, so the parent test process never
// initializes CUDA before fork/exec.
//
// The deployment target is Linux; Windows is rejected explicitly because these
// cases require /proc/self/exe and POSIX fork/execve subprocesses.

#if defined(_WIN32)
#error "course/tests/warm_benchmark_tests.cu requires the POSIX subprocess runner"
#endif

#include "checkpoint.h"
#include "course/interface.h"
#include "course/model_io.h"
#include "course/tests/test_support.h"
#include "parameters.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
using course_test_support::WriteText;

constexpr std::uint32_t kChannelCount = 1;
constexpr std::uint32_t kImageSide = 28;
constexpr std::uint32_t kClassCount = 10;
constexpr std::uint32_t kImagePixels = kImageSide * kImageSide;
constexpr std::uint32_t kDefaultInferBatch = 1024;
constexpr std::uint32_t kMaximumInferBatch = 16384;
constexpr std::uint64_t kFixtureSeed = UINT64_C(1337);

// Every child is bounded; CTest should additionally apply a larger TIMEOUT.
constexpr int kChildTimeoutMilliseconds = 240000;

using Clock = std::chrono::steady_clock;

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
        "could not resolve /proc/self/exe for the warm benchmark tests");
  }
  buffer[length] = '\0';
  return std::string(buffer);
}

// Directory containing this test binary. CMake places the tool next to it.
std::string SelfExecutableDirectory() {
  const std::string path = SelfExecutablePath();
  const std::string::size_type separator = path.find_last_of('/');
  if (separator == std::string::npos) {
    return ".";
  }
  return path.substr(0, separator);
}

// Requires the named build-adjacent executable. Its absence is reported by
// name before any hardware or fixture work.
std::string BuildAdjacentExecutable(const char* name, const char* purpose) {
  const std::string path = SelfExecutableDirectory() + "/" + name;
  if (!PathExists(path)) {
    throw std::runtime_error(std::string("missing ") + purpose + " '" + name +
                             "' at build-adjacent path " + path +
                             "; build that target next to the warm benchmark "
                             "tests");
  }
  return path;
}

std::string BenchmarkExecutable() {
  return BuildAdjacentExecutable("course_warm_benchmark",
                                 "warm benchmark entry point");
}

// Called first in every case so a missing tool is reported by name before any
// fixture, official oracle, or hardware access.
void RequireBenchmarkExecutable() {
  static_cast<void>(BenchmarkExecutable());
}

std::string OfficialInferExecutable() {
  return BuildAdjacentExecutable("infer", "official entry point");
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
// checks each write result and retries short writes; it allocates nothing
// between fork and exec.
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

void CloseDescriptor(int* descriptor) {
  if (*descriptor >= 0) {
    ::close(*descriptor);
    *descriptor = -1;
  }
}

// Runs one executable with an exact argv, an exact COURSE_* environment, and an
// optional working directory. Both output streams are captured through pipes
// with poll, and both the stream loop and the final wait are bounded.
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

  std::vector<std::string> environment_storage = BuildEnvironment(environment);
  std::vector<char*> environment_pointers;
  for (std::string& variable : environment_storage) {
    environment_pointers.push_back(&variable[0]);
  }
  environment_pointers.push_back(nullptr);

  int output_pipe[2] = {-1, -1};
  int error_pipe[2] = {-1, -1};
  if (::pipe(output_pipe) != 0) {
    throw std::runtime_error("could not create subprocess capture pipe");
  }
  if (::pipe(error_pipe) != 0) {
    CloseDescriptor(&output_pipe[0]);
    CloseDescriptor(&output_pipe[1]);
    throw std::runtime_error("could not create subprocess error pipe");
  }

  const pid_t child = ::fork();
  if (child < 0) {
    CloseDescriptor(&output_pipe[0]);
    CloseDescriptor(&output_pipe[1]);
    CloseDescriptor(&error_pipe[0]);
    CloseDescriptor(&error_pipe[1]);
    throw std::runtime_error("could not fork a benchmark subprocess");
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
    CloseDescriptor(&output_pipe[0]);
    CloseDescriptor(&output_pipe[1]);
    CloseDescriptor(&error_pipe[0]);
    CloseDescriptor(&error_pipe[1]);

    ::execve(executable.c_str(), argument_pointers.data(),
             environment_pointers.data());
    WriteChildDiagnostic("could not execute the benchmark executable\n",
                         sizeof("could not execute the benchmark executable\n") -
                             1);
    ::_exit(127);
  }

  CloseDescriptor(&output_pipe[1]);
  CloseDescriptor(&error_pipe[1]);

  ProcessResult result;
  result.status = 255;
  int descriptors[2] = {output_pipe[0], error_pipe[0]};
  std::string* destinations[2] = {&result.out, &result.err};
  const Clock::time_point deadline =
      Clock::now() + std::chrono::milliseconds(kChildTimeoutMilliseconds);
  bool timed_out = false;

  while (descriptors[0] >= 0 || descriptors[1] >= 0) {
    if (Clock::now() >= deadline) {
      timed_out = true;
      break;
    }
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

    const long long remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline -
                                                              Clock::now())
            .count();
    const int poll_timeout =
        remaining > static_cast<long long>(INT_MAX)
            ? INT_MAX
            : static_cast<int>(remaining > 0 ? remaining : 1);
    const int ready = ::poll(watched, watched_count, poll_timeout);
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
        CloseDescriptor(&descriptors[index]);
      }
    }
  }
  CloseDescriptor(&descriptors[0]);
  CloseDescriptor(&descriptors[1]);

  int wait_status = 0;
  pid_t waited = -1;
  if (!timed_out) {
    while (true) {
      waited = ::waitpid(child, &wait_status, WNOHANG);
      if (waited == child) {
        break;
      }
      if (waited < 0 && errno != EINTR) {
        throw std::runtime_error(
            "could not wait for a benchmark subprocess");
      }
      if (Clock::now() >= deadline) {
        timed_out = true;
        break;
      }
      ::poll(nullptr, 0, 10);
    }
  }

  if (timed_out) {
    ::kill(child, SIGKILL);
    int reaped_status = 0;
    pid_t reaped = -1;
    do {
      reaped = ::waitpid(child, &reaped_status, 0);
    } while (reaped < 0 && errno == EINTR);
    std::ostringstream message;
    message << "benchmark subprocess timed out after "
            << kChildTimeoutMilliseconds << " ms; stdout=[" << result.out
            << "] stderr=[" << result.err << ']';
    throw std::runtime_error(message.str());
  }

  if (waited != child) {
    throw std::runtime_error("benchmark subprocess exited without a wait status");
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

// One complete failed invocation must publish no predictions and no success
// completion record.
void ExpectFailedRun(const ProcessResult& result, int expected_status,
                     const std::string& context) {
  ExpectExit(result, expected_status, context);
  ExpectEmptyStdout(result, context);
  ExpectNonEmptyDiagnostics(result, context);
  if (result.err.find("event=course_warm_complete status=pass") !=
      std::string::npos) {
    Fail(__FILE__, __LINE__,
         context + ": failed run published a passing completion record: [" +
             result.err + ']');
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

// Deterministic normalized images spanning both signs with fractional values.
std::vector<float> NormalizedPatternImages(std::uint32_t sample_count) {
  std::vector<float> values(static_cast<std::size_t>(sample_count) *
                            kImagePixels);
  for (std::size_t index = 0; index < values.size(); ++index) {
    const std::int64_t centered =
        static_cast<std::int64_t>((index * 53u + 17u) % 1001u) - 500;
    values[index] = static_cast<float>(centered) / 1000.0F;
  }
  return values;
}

// Deterministic unit-mode images inside the inclusive [0, 1] contract.
std::vector<float> UnitPatternImages(std::uint32_t sample_count) {
  std::vector<float> values(static_cast<std::size_t>(sample_count) *
                            kImagePixels);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] =
        static_cast<float>((index * 29u + 7u) % 1001u) / 1000.0F;
  }
  return values;
}

// Two normalized images: all zero, then all one.
std::vector<float> ZeroOneImages() {
  std::vector<float> images(static_cast<std::size_t>(2) * kImagePixels, 0.0F);
  std::fill(images.begin() + kImagePixels, images.end(), 1.0F);
  return images;
}

// Finite normalized inputs whose first image overflows the forward chain.
std::vector<float> PoisonImages() {
  std::vector<float> images(static_cast<std::size_t>(2) * kImagePixels, 0.0F);
  std::fill(images.begin(), images.begin() + kImagePixels,
            std::numeric_limits<float>::max());
  return images;
}

Checkpoint BaseCheckpoint() {
  Checkpoint checkpoint;
  checkpoint.metadata = CheckpointMetadata{1, 0.0F, 0.1307F, 0.3081F};
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
// starts one float late changes the second prediction. The expected classes are
// an oracle fixed by the fixture, never computed with host NN arithmetic.
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

// Rewrites the persisted CRC field to a different eight-digit value without
// touching a valid checkpoint payload.
void TamperMetadataCrc(const std::string& directory) {
  const std::string path = directory + "/course.meta";
  std::string text = ReadText(path);
  const std::string key = "checkpoint_crc32=";
  const std::string::size_type position = text.find(key);
  if (position == std::string::npos ||
      position + key.size() + 8 > text.size()) {
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

std::vector<EnvVar> ModelEnvironment(const std::string& model_dir,
                                     const std::string& batch) {
  return {EnvVar{"COURSE_MODEL_DIR", model_dir},
          EnvVar{"COURSE_INFER_BATCH", batch}};
}

// Exact eight-argument tool request; argv[0] is supplied by RunProcess.
std::vector<std::string> WarmArguments(const std::string& images_path,
                                       const std::string& n,
                                       const std::string& warmups,
                                       const std::string& runs) {
  return {images_path, n, "1", "28", "28", "10", warmups, runs};
}

ProcessResult RunWarm(const std::string& images_path, const std::string& n,
                      const std::string& warmups, const std::string& runs,
                      const std::vector<EnvVar>& environment,
                      const std::string& working_directory) {
  return RunProcess(BenchmarkExecutable(),
                    WarmArguments(images_path, n, warmups, runs), environment,
                    working_directory);
}

ProcessResult RunOfficialInfer(const std::string& images_path,
                               std::uint32_t sample_count,
                               const std::string& model_dir,
                               const std::string& batch,
                               const std::string& working_directory) {
  return RunProcess(OfficialInferExecutable(),
                    {images_path, std::to_string(sample_count), "1", "28", "28",
                     "10"},
                    ModelEnvironment(model_dir, batch), working_directory);
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

// Splits one protocol line on single spaces, ignoring empty runs.
std::vector<std::string> SplitTokens(const std::string& line) {
  std::vector<std::string> tokens;
  std::string::size_type start = 0;
  while (start < line.size()) {
    const std::string::size_type end = line.find(' ', start);
    if (end == std::string::npos) {
      if (start < line.size()) {
        tokens.push_back(line.substr(start));
      }
      break;
    }
    if (end > start) {
      tokens.push_back(line.substr(start, end - start));
    }
    start = end + 1;
  }
  return tokens;
}

// Extracts the value of an exact whitespace-delimited key=value token from one
// line. Using complete tokens avoids matching "samples=" inside another field.
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

// Requires an exact ordered field sequence, rejecting extra or missing fields.
void ExpectProtocolFields(const std::string& line,
                          const std::vector<std::string>& keys,
                          const std::string& context) {
  const std::vector<std::string> tokens = SplitTokens(line);
  if (tokens.size() != keys.size()) {
    Fail(__FILE__, __LINE__,
         context + ": expected " + std::to_string(keys.size()) +
             " protocol fields but observed " + std::to_string(tokens.size()) +
             " in line [" + line + ']');
  }
  for (std::size_t index = 0; index < keys.size(); ++index) {
    if (tokens[index].compare(0, keys[index].size(), keys[index]) != 0) {
      Fail(__FILE__, __LINE__,
           context + ": expected field [" + keys[index] + "] at position " +
               std::to_string(index + 1) + " but observed [" + tokens[index] +
               "] in line [" + line + ']');
    }
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

// Returns the unique line beginning with prefix, or fails with the complete
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

// Parses exactly N canonical decimal class lines, each one digit in [0, 10)
// followed by '\n', with nothing else on stdout.
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

// Requires one complete round record with exact field order, a one-based index
// for its phase, full-request sample count, positive wall time, and the
// prediction-invariance flag.
void ExpectIterationRecord(const std::string& line, const char* phase,
                           std::uint32_t iteration, std::uint32_t samples,
                           const std::string& context) {
  ExpectProtocolFields(line,
                       {"event=", "phase=", "iteration=", "samples=",
                        "wall_ms=", "predictions_identical="},
                       context);
  ExpectToken(line, "event=", "course_warm_iteration", context);
  ExpectToken(line, "phase=", phase, context);
  ExpectToken(line, "iteration=", std::to_string(iteration), context);
  ExpectToken(line, "samples=", std::to_string(samples), context);
  ExpectToken(line, "predictions_identical=", "1", context);
  const double wall_ms = ParseDoubleToken(line, "wall_ms=", context);
  if (!(wall_ms > 0.0)) {
    Fail(__FILE__, __LINE__,
         context + ": wall_ms must be strictly positive in line [" + line + ']');
  }
}

void ExpectCompletionRecord(const std::string& line, std::uint32_t samples,
                            std::uint32_t warmups, std::uint32_t runs,
                            std::uint32_t batch, course::InputMode input_mode,
                            const std::string& context) {
  ExpectProtocolFields(line,
                       {"event=", "status=", "warmups=", "runs=", "samples=",
                        "batch=", "input_mode="},
                       context);
  ExpectToken(line, "event=", "course_warm_complete", context);
  ExpectToken(line, "status=", "pass", context);
  ExpectToken(line, "warmups=", std::to_string(warmups), context);
  ExpectToken(line, "runs=", std::to_string(runs), context);
  ExpectToken(line, "samples=", std::to_string(samples), context);
  ExpectToken(line, "batch=", std::to_string(batch), context);
  ExpectToken(line, "input_mode=", InputModeName(input_mode), context);
}

// Requires a successful run: exit 0, exactly N canonical predictions, exactly
// W warmup records followed by R measure records with separate one-based
// indices, and one passing completion record.
void ExpectWarmSuccess(const ProcessResult& result, std::uint32_t samples,
                       std::uint32_t warmups, std::uint32_t runs,
                       std::uint32_t batch, course::InputMode input_mode,
                       const std::string& context) {
  ExpectExit(result, 0, context);
  ParsePredictions(result.out, samples, context);
  ExpectNonEmptyDiagnostics(result, context);

  const std::vector<std::string> iterations =
      LinesStarting(result.err, "event=course_warm_iteration ");
  const std::size_t expected_iterations =
      static_cast<std::size_t>(warmups) + runs;
  if (iterations.size() != expected_iterations) {
    std::ostringstream message;
    message << context << ": expected " << expected_iterations
            << " iteration records but observed " << iterations.size()
            << "; diagnostics=[" << result.err << ']';
    Fail(__FILE__, __LINE__, message.str());
  }
  for (std::uint32_t index = 0; index < warmups; ++index) {
    ExpectIterationRecord(iterations[index], "warmup", index + 1, samples,
                          context);
  }
  for (std::uint32_t index = 0; index < runs; ++index) {
    ExpectIterationRecord(iterations[warmups + index], "measure", index + 1,
                          samples, context);
  }

  const std::string completion = SingleLineStarting(
      result.err, "event=course_warm_complete ", context);
  ExpectCompletionRecord(completion, samples, warmups, runs, batch, input_mode,
                         context);
}

// ---------------------------------------------------------------------------
// Protocol, literal fixture, and batch partitions
// ---------------------------------------------------------------------------

TEST_CASE(warm_tool_literal_fixture_is_batch_invariant_across_rounds) {
  RequireBenchmarkExecutable();
  const std::string root = MakeScratchDirectory("warm_literal");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kNormalized,
               BatchSensitiveCheckpoint());
  const std::string images_path =
      WriteImages(root + "/data", "pair.f32", ZeroOneImages());

  // The fixture is format-valid; both pixels are retained bit-for-bit.
  const std::vector<float> read_back = course::ReadImages(
      images_path, FixtureShape(2), course::InputMode::kNormalized);
  EXPECT_EQ(static_cast<std::size_t>(2) * kImagePixels, read_back.size());
  EXPECT_TRUE(SameBits(0.0F, read_back[0]));
  EXPECT_TRUE(SameBits(1.0F, read_back[kImagePixels]));

  // Both batches must yield the fixture's [1, 0] in warmup and measured rounds.
  const char* batches[] = {"1", "2"};
  for (const char* batch : batches) {
    const std::uint32_t expected_batch =
        std::strcmp(batch, "1") == 0 ? 1u : 2u;
    const std::string context =
        std::string("literal warm benchmark batch=") + batch;
    const ProcessResult result =
        RunWarm(images_path, "2", "2", "3", ModelEnvironment(model_dir, batch),
                root);
    ExpectWarmSuccess(result, 2, 2, 3, expected_batch,
                      course::InputMode::kNormalized, context);
    const std::vector<int> predictions =
        ParsePredictions(result.out, 2, context);
    EXPECT_EQ(1, predictions[0]);
    EXPECT_EQ(0, predictions[1]);
  }
}

TEST_CASE(warm_tool_matches_official_infer_for_17_and_1025_images) {
  RequireBenchmarkExecutable();
  const std::string root = MakeScratchDirectory("warm_official_match");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kNormalized,
               InitializedCheckpoint());

  const std::uint32_t full_count = 1025;
  const std::vector<float> full_images = NormalizedPatternImages(full_count);
  const std::string full_path =
      WriteImages(root + "/data", "full.f32", full_images);

  struct CountCase {
    std::uint32_t samples;
    const char* batch;
  };
  const CountCase cases[] = {{17, "4"}, {1025, "1024"}};
  for (const CountCase& count_case : cases) {
    const std::uint32_t batch_value = static_cast<std::uint32_t>(
        std::strtoul(count_case.batch, nullptr, 10));
    const std::vector<float> prefix_images(
        full_images.begin(),
        full_images.begin() +
            static_cast<std::size_t>(count_case.samples) * kImagePixels);
    const std::string images_path =
        count_case.samples == full_count
            ? full_path
            : WriteImages(root + "/data", "prefix17.f32", prefix_images);
    const std::string context =
        std::string("warm tool N=") + std::to_string(count_case.samples) +
        " batch=" + count_case.batch;

    // The official executable is the oracle; no class is hardcoded here.
    const ProcessResult oracle = RunOfficialInfer(
        images_path, count_case.samples, model_dir, count_case.batch, root);
    ExpectExit(oracle, 0, context + " official infer oracle");
    const std::vector<int> oracle_predictions =
        ParsePredictions(oracle.out, count_case.samples,
                         context + " official infer oracle");
    const std::string oracle_completion = SingleLineStarting(
        oracle.err, "event=course_infer status=pass ",
        context + " official infer oracle");
    ExpectToken(oracle_completion, "samples=",
                std::to_string(count_case.samples),
                context + " official infer oracle");
    const std::uint32_t oracle_batches =
        1 + (count_case.samples - 1) / batch_value;
    ExpectToken(oracle_completion, "batches=", std::to_string(oracle_batches),
                context + " official infer oracle");
    ExpectToken(oracle_completion, "input_mode=", "normalized",
                context + " official infer oracle");

    const ProcessResult warm =
        RunWarm(images_path, std::to_string(count_case.samples), "2", "3",
                ModelEnvironment(model_dir, count_case.batch), root);
    ExpectWarmSuccess(warm, count_case.samples, 2, 3, batch_value,
                      course::InputMode::kNormalized, context);
    const std::vector<int> warm_predictions =
        ParsePredictions(warm.out, count_case.samples, context);
    EXPECT_EQ(oracle_predictions, warm_predictions);
    EXPECT_EQ(oracle.out, warm.out);
  }
}

TEST_CASE(warm_tool_rejects_early_nonfinite_without_predictions) {
  RequireBenchmarkExecutable();
  const std::string root = MakeScratchDirectory("warm_poison");
  const std::string model_dir = root + "/model";
  // Every weight is finite one and every bias is zero; the input alone makes
  // the forward chain overflow, so the device sticky flag must abort the run.
  PublishModel(model_dir, course::InputMode::kNormalized,
               UniformCheckpoint(1.0F, 0.0F));

  // The same tool, model, and round counts succeed on finite in-range images,
  // so the poison failure below is not a missing-device or setup failure.
  const std::string safe_path =
      WriteImages(root + "/data", "safe.f32", NormalizedPatternImages(2));
  const ProcessResult safe = RunWarm(
      safe_path, "2", "2", "3", ModelEnvironment(model_dir, "1"), root);
  ExpectWarmSuccess(safe, 2, 2, 3, 1, course::InputMode::kNormalized,
                    "finite warm benchmark precondition");

  const std::string images_path =
      WriteImages(root + "/data", "poison.f32", PoisonImages());
  const std::string context = "poison warm benchmark";
  const ProcessResult result =
      RunWarm(images_path, "2", "2", "3", ModelEnvironment(model_dir, "1"), root);
  ExpectFailedRun(result, 1, context);
}

// ---------------------------------------------------------------------------
// Exact argv, strict decimal counts, batch, model, and input validation
// ---------------------------------------------------------------------------

TEST_CASE(warm_tool_requires_exact_argv_and_bounded_counts) {
  RequireBenchmarkExecutable();
  const std::string root = MakeScratchDirectory("warm_usage");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kNormalized,
               BatchSensitiveCheckpoint());
  const std::string images_path =
      WriteImages(root + "/data", "pair.f32", ZeroOneImages());
  const std::vector<EnvVar> environment = ModelEnvironment(model_dir, "1");
  const std::string tool = BenchmarkExecutable();

  ExpectFailedRun(RunProcess(tool, {}, {}, root), 2,
                  "warm tool with no arguments");
  ExpectFailedRun(RunProcess(tool,
                             {images_path, "2", "1", "28", "28", "10", "2"},
                             environment, root),
                  2, "warm tool with 7 arguments (argc 8)");
  ExpectFailedRun(
      RunProcess(tool,
                 {images_path, "2", "1", "28", "28", "10", "2", "3", "extra"},
                 environment, root),
      2, "warm tool with 9 arguments (argc 10)");

  auto exact = [&](const std::string& n, const std::string& warmups,
                   const std::string& runs) {
    return WarmArguments(images_path, n, warmups, runs);
  };

  const char* bad_n[] = {"0", "-1", "+2", "abc", "2.0", "4294967296"};
  for (const char* value : bad_n) {
    ExpectFailedRun(RunProcess(tool, exact(value, "2", "3"), environment, root),
                    2, std::string("warm tool rejected N [") + value + ']');
  }

  const char* bad_warmups[] = {"0", "-1", "abc", "100001", "1.5"};
  for (const char* value : bad_warmups) {
    ExpectFailedRun(RunProcess(tool, exact("2", value, "3"), environment, root),
                    2, std::string("warm tool rejected warmups [") + value +
                           ']');
  }

  const char* bad_runs[] = {"0", "-3", "abc", "100001", "3x"};
  for (const char* value : bad_runs) {
    ExpectFailedRun(RunProcess(tool, exact("2", "2", value), environment, root),
                    2, std::string("warm tool rejected runs [") + value + ']');
  }

  const std::vector<std::vector<std::string>> unsupported_geometry{
      {images_path, "2", "3", "28", "28", "10", "2", "3"},
      {images_path, "2", "1", "27", "28", "10", "2", "3"},
      {images_path, "2", "1", "28", "27", "10", "2", "3"},
      {images_path, "2", "1", "28", "28", "9", "2", "3"},
  };
  for (const std::vector<std::string>& arguments : unsupported_geometry) {
    ExpectFailedRun(RunProcess(tool, arguments, environment, root), 2,
                    "warm tool with unsupported geometry");
  }
}

TEST_CASE(warm_tool_rejects_invalid_batch_and_model_bundles) {
  RequireBenchmarkExecutable();
  const std::string root = MakeScratchDirectory("warm_bad_options");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kNormalized,
               InitializedCheckpoint());
  const std::string images_path =
      WriteImages(root + "/data", "pair.f32", ZeroOneImages());
  const std::vector<std::string> arguments =
      WarmArguments(images_path, "2", "2", "3");
  const std::string tool = BenchmarkExecutable();

  // COURSE_INFER_BATCH has the official [1, 16384] usage contract.
  const char* bad_batches[] = {"0", "16385", "-1", "abc", ""};
  for (const char* value : bad_batches) {
    ExpectFailedRun(RunProcess(tool, arguments,
                               ModelEnvironment(model_dir, value), root),
                    2, std::string("warm tool rejected COURSE_INFER_BATCH [") +
                           value + ']');
  }
  EXPECT_EQ(static_cast<std::uint32_t>(kDefaultInferBatch), 1024u);
  EXPECT_EQ(static_cast<std::uint32_t>(kMaximumInferBatch), 16384u);

  // A missing or empty model directory is a usage error.
  ExpectFailedRun(RunProcess(tool, arguments, {}, root), 2,
                  "warm tool without COURSE_MODEL_DIR");
  ExpectFailedRun(RunProcess(tool, arguments,
                             {EnvVar{"COURSE_MODEL_DIR", ""}}, root),
                  2, "warm tool with empty COURSE_MODEL_DIR");

  // Bundle failures are runtime errors, not usage errors.
  const std::string missing_model = root + "/missing_model";
  ExpectFailedRun(
      RunProcess(tool, arguments, ModelEnvironment(missing_model, "1"), root),
      1, "warm tool with missing model directory");

  const std::string malformed_model = root + "/malformed_model";
  PublishModel(malformed_model, course::InputMode::kUnit,
               InitializedCheckpoint());
  WriteText(malformed_model + "/course.meta", "not_a_course_model\n");
  ExpectFailedRun(
      RunProcess(tool, arguments, ModelEnvironment(malformed_model, "1"), root),
      1, "warm tool with malformed model metadata");

  const std::string bad_crc_model = root + "/bad_crc_model";
  PublishModel(bad_crc_model, course::InputMode::kUnit,
               InitializedCheckpoint());
  TamperMetadataCrc(bad_crc_model);
  ExpectFailedRun(
      RunProcess(tool, arguments, ModelEnvironment(bad_crc_model, "1"), root),
      1, "warm tool with bad model CRC");

  const std::string corrupt_model = root + "/corrupt_model";
  PublishModel(corrupt_model, course::InputMode::kUnit,
               InitializedCheckpoint());
  CorruptCheckpointPayload(corrupt_model);
  ExpectFailedRun(
      RunProcess(tool, arguments, ModelEnvironment(corrupt_model, "1"), root),
      1, "warm tool with corrupt model payload");
}

TEST_CASE(warm_tool_rejects_invalid_images) {
  RequireBenchmarkExecutable();
  const std::string root = MakeScratchDirectory("warm_bad_images");
  const std::string model_dir = root + "/model";
  PublishModel(model_dir, course::InputMode::kUnit, InitializedCheckpoint());
  const std::vector<EnvVar> environment = ModelEnvironment(model_dir, "1");
  const std::vector<float> valid = UnitPatternImages(2);
  const std::string data_dir = root + "/data";

  // Declared N=2 with a file that holds exactly one image.
  const std::string short_path = WriteImages(
      data_dir, "short.f32",
      std::vector<float>(valid.begin(), valid.begin() + kImagePixels));
  EXPECT_EQ(static_cast<std::size_t>(kImagePixels) * sizeof(float),
            ReadBytes(short_path).size());

  // Declared N=2 with three images present.
  std::vector<float> long_values(static_cast<std::size_t>(3) * kImagePixels,
                                 0.25F);
  const std::string long_path =
      WriteImages(data_dir, "long.f32", long_values);
  EXPECT_EQ(static_cast<std::size_t>(3) * kImagePixels * sizeof(float),
            ReadBytes(long_path).size());

  // Non-finite and out-of-mode values are rejected on the host.
  std::vector<float> non_finite = valid;
  non_finite[0] = std::numeric_limits<float>::quiet_NaN();
  const std::string non_finite_path =
      WriteImages(data_dir, "non_finite.f32", non_finite);

  std::vector<float> out_of_range = valid;
  out_of_range[0] = 2.0F;
  const std::string out_of_range_path =
      WriteImages(data_dir, "out_of_range.f32", out_of_range);

  const std::string missing_path = data_dir + "/missing.f32";
  EXPECT_FALSE(PathExists(missing_path));

  struct ImageCase {
    std::string path;
    std::string label;
  };
  const ImageCase cases[] = {
      {short_path, "short image file"},
      {long_path, "long image file"},
      {non_finite_path, "non-finite image value"},
      {out_of_range_path, "unit image outside [0, 1]"},
      {missing_path, "missing image file"},
  };
  for (const ImageCase& image_case : cases) {
    const std::string context = "warm tool with " + image_case.label;
    const ProcessResult result =
        RunWarm(image_case.path, "2", "2", "3", environment, root);
    ExpectFailedRun(result, 1, context);
  }
}

// ---------------------------------------------------------------------------
// Persisted mode and ignored training/profile environment
// ---------------------------------------------------------------------------

TEST_CASE(warm_tool_uses_persisted_mode_and_ignores_training_profile_variables) {
  RequireBenchmarkExecutable();
  const std::string root = MakeScratchDirectory("warm_persisted_mode");

  struct ModeCase {
    course::InputMode input_mode;
    std::string spelling;
    std::vector<float> images;
    std::string conflicting_mode;
    std::string conflicting_epochs;
  };
  std::vector<ModeCase> cases;
  // Negative normalized pixels would be rejected by a wrongly selected unit
  // mode. Unit pixels stay valid under normalized mode, so the oracle output is
  // what proves the persisted mode was actually used.
  cases.push_back({course::InputMode::kNormalized, "normalized",
                   NormalizedPatternImages(5), "unit", "0"});
  cases.push_back({course::InputMode::kUnit, "unit", UnitPatternImages(5),
                   "normalized", "abc"});

  const std::uint32_t sample_count = 5;
  const std::string batch = "2";
  for (const ModeCase& mode_case : cases) {
    const std::string case_root = root + "/" + mode_case.spelling;
    MakeDirectory(case_root);
    const std::string model_dir = case_root + "/model";
    PublishModel(model_dir, mode_case.input_mode, InitializedCheckpoint());
    const std::string images_path =
        WriteImages(case_root + "/data", "images.f32", mode_case.images);
    const std::string context = "persisted mode " + mode_case.spelling;

    const ProcessResult oracle =
        RunOfficialInfer(images_path, sample_count, model_dir, batch, root);
    ExpectExit(oracle, 0, context + " official infer oracle");
    ParsePredictions(oracle.out, sample_count,
                     context + " official infer oracle");

    std::vector<EnvVar> environment = ModelEnvironment(model_dir, batch);
    // Training mode, epochs, seed, and profile are not warm-tool inputs. The
    // invalid spellings must be ignored rather than parsed as train options.
    environment.push_back(
        EnvVar{"COURSE_INPUT_MODE", mode_case.conflicting_mode});
    environment.push_back(
        EnvVar{"COURSE_EPOCHS", mode_case.conflicting_epochs});
    environment.push_back(EnvVar{"COURSE_SEED", "999"});
    environment.push_back(EnvVar{"COURSE_PROFILE", "1"});

    const ProcessResult warm =
        RunWarm(images_path, std::to_string(sample_count), "2", "3",
                environment, root);
    ExpectWarmSuccess(warm, sample_count, 2, 3, 2, mode_case.input_mode,
                      context);
    EXPECT_EQ(oracle.out, warm.out);

    // The tool is not the profiling runtime and never emits infer/profile
    // records or a second stdout publication.
    EXPECT_TRUE(
        LinesStarting(warm.err, "event=course_profile_batch ").empty());
    EXPECT_TRUE(LinesStarting(warm.err, "event=course_profile ").empty());
    EXPECT_TRUE(LinesStarting(warm.err, "event=course_infer ").empty());
  }
}

}  // namespace
