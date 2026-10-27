// Implements the tests-owned snapshot comparison utility. It runs the existing
// CUDA forward, softmax, and argmax launchers over the untouched official
// inputs and writes one deterministic binary snapshot for old/new comparison.
// This helper targets the Linux/H20 test environment; production builds never
// link it and normal test runners never execute it.
#if defined(_WIN32)
#error "tests/evaluate_snapshot.cu is a Linux/H20-only test helper"
#endif
// Request the POSIX.1-2008 realpath declaration before libc headers. The
// guarded declaration after the includes covers strict-ANSI toolchains where an
// earlier CUDA header may already have fixed the libc feature set.
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif
#include "checkpoint.h"
#include "cuda_check.h"
#include "dataset.h"
#include "layers.h"
#include "lenet.h"
#include "tensor.h"
#include "train.h"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(__USE_XOPEN2K8) && !defined(__USE_XOPEN_EXTENDED)
// glibc hides the POSIX realpath declaration when strict ANSI mode was fixed
// before this source. Supply the documented signature for this Linux-only tool.
extern "C" char* realpath(const char* path, char* resolved_path);
#endif

namespace {

// Snapshot shape: 10000 sample-ordered logits, probabilities, and predictions.
const std::uint32_t kExpectedSampleCount = 10000;
const int kClassCount = 10;
const std::size_t kImagePixels = 28 * 28;
const std::uint32_t kMaximumBatchSize = 128;
const std::size_t kClassElementCount =
    static_cast<std::size_t>(kExpectedSampleCount) * kClassCount;
const std::size_t kLogitsBytes = kClassElementCount * sizeof(float);
const std::size_t kProbabilitiesBytes = kClassElementCount * sizeof(float);
const std::size_t kExpectedSnapshotBytes =
    kLogitsBytes + kProbabilitiesBytes + kExpectedSampleCount;
static_assert(kExpectedSnapshotBytes == 810000,
              "snapshot output size must remain exactly 810000 bytes");
// The H20 target is little-endian, so native float and uint8 bytes are the
// documented no-header format; the fixed width also keeps offsets exact.
static_assert(sizeof(float) == 4,
              "snapshot format requires 32-bit IEEE floats");

// Mounted read-only roots that must never receive snapshot output.
const char* const kDefaultMountRoots[] = {
    "/gemini/data-1", "/gemini/data-2", "/gemini/data-3", "/gemini/pretrain"};
const char* const kMountRootVariables[] = {
    "GEMINI_DATA_IN1", "GEMINI_DATA_IN2", "GEMINI_DATA_IN3",
    "GEMINI_PRETRAIN"};

// Returns the stable usage text printed for syntax failures.
const char* UsageText() {
  return "usage: evaluate_snapshot DATA WEIGHTS OUTPUT BATCH\n";
}

// Parses a strictly decimal batch size in the inclusive [1,128] range.
std::uint32_t ParseBatchSize(const std::string& text) {
  if (text.empty()) {
    throw std::invalid_argument("batch size must be decimal digits in [1,128]");
  }
  std::uint32_t value = 0;
  for (std::string::const_iterator character = text.begin();
       character != text.end(); ++character) {
    if (*character < '0' || *character > '9') {
      throw std::invalid_argument(
          "batch size must be decimal digits in [1,128]");
    }
    const std::uint32_t digit =
        static_cast<std::uint32_t>(*character - '0');
    if (value >
        (kMaximumBatchSize - digit) / static_cast<std::uint32_t>(10)) {
      throw std::invalid_argument("batch size is outside [1,128]");
    }
    value = value * 10 + digit;
  }
  if (value < 1 || value > kMaximumBatchSize) {
    throw std::invalid_argument("batch size is outside [1,128]");
  }
  return value;
}

// Returns the parent directory of a POSIX path.
std::string ParentPath(const std::string& path) {
  const std::string::size_type separator = path.find_last_of('/');
  if (separator == std::string::npos) {
    return ".";
  }
  if (separator == 0) {
    return "/";
  }
  return path.substr(0, separator);
}

// Resolves an existing path to its canonical absolute form. realpath allocates
// the result; the owner releases it on every exit path. Resolution failures are
// fatal so a mounted or aliased path can never be treated as safe by accident.
std::string CanonicalExistingPath(const std::string& path,
                                  const char* description) {
  char* resolved = realpath(path.c_str(), nullptr);
  if (resolved == nullptr) {
    throw std::runtime_error(std::string("cannot resolve ") + description +
                             ": " + path + ": " + std::strerror(errno));
  }
  std::unique_ptr<char, void (*)(void*)> owned(resolved, std::free);
  return std::string(owned.get());
}

// Returns true when candidate equals root or is a strict descendant. Both
// inputs are canonical absolute paths; the separator boundary keeps /data-10
// outside /data-1. Root "/" contains every absolute path.
bool IsSameOrDescendant(const std::string& candidate,
                        const std::string& root) {
  if (candidate == root) {
    return true;
  }
  if (root == "/") {
    return !candidate.empty() && candidate[0] == '/';
  }
  return candidate.size() > root.size() &&
         candidate.compare(0, root.size(), root) == 0 &&
         candidate[root.size()] == '/';
}

// Collects every mounted read-only root selected by the platform defaults and
// environment. Absent or non-directory roots are ignored; every present root is
// canonicalized so aliases and nested paths compare against its real location.
std::vector<std::string> MountedRoots() {
  std::vector<std::string> roots;
  for (std::size_t index = 0;
       index < sizeof(kDefaultMountRoots) / sizeof(kDefaultMountRoots[0]);
       ++index) {
    roots.push_back(kDefaultMountRoots[index]);
  }
  for (std::size_t index = 0;
       index < sizeof(kMountRootVariables) / sizeof(kMountRootVariables[0]);
       ++index) {
    const char* value = std::getenv(kMountRootVariables[index]);
    if (value != nullptr && value[0] != '\0') {
      roots.push_back(value);
    }
  }
  std::vector<std::string> present;
  for (std::size_t index = 0; index < roots.size(); ++index) {
    struct stat information;
    if (stat(roots[index].c_str(), &information) == 0 &&
        (information.st_mode & S_IFMT) == S_IFDIR) {
      present.push_back(CanonicalExistingPath(roots[index], "mount root"));
    }
  }
  return present;
}

// Rejects missing parents, existing destinations, and mounted destinations
// before any dataset, checkpoint, or device state is created.
void ValidateOutputPath(const std::string& path) {
  if (path.empty()) {
    throw std::invalid_argument("output path must not be empty");
  }
  struct stat information;
  if (stat(path.c_str(), &information) == 0) {
    throw std::runtime_error(
        "output path already exists and will not be overwritten: " + path);
  }
  if (errno != ENOENT) {
    throw std::runtime_error("output path cannot be inspected: " + path +
                             ": " + std::strerror(errno));
  }
  const std::string parent = ParentPath(path);
  if (stat(parent.c_str(), &information) != 0 ||
      (information.st_mode & S_IFMT) != S_IFDIR) {
    throw std::runtime_error("output parent directory does not exist: " +
                             parent);
  }
  const std::string canonical_parent =
      CanonicalExistingPath(parent, "output parent");
  const std::vector<std::string> roots = MountedRoots();
  for (std::size_t index = 0; index < roots.size(); ++index) {
    if (IsSameOrDescendant(canonical_parent, roots[index])) {
      throw std::runtime_error(
          "output path must be outside mounted data/pretrain roots: " + path +
          " (root " + roots[index] + ")");
    }
  }
}

// Writes every requested byte, reporting short or failed writes with a path.
void WriteAll(int descriptor, const void* data, std::size_t bytes,
              const std::string& path) {
  const char* cursor = static_cast<const char*>(data);
  std::size_t remaining = bytes;
  while (remaining > 0) {
    const ssize_t written = write(descriptor, cursor, remaining);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error("cannot write snapshot: " + path + ": " +
                               std::strerror(errno));
    }
    if (written == 0) {
      throw std::runtime_error("snapshot write returned zero bytes: " + path);
    }
    cursor += written;
    remaining -= static_cast<std::size_t>(written);
  }
}

// Creates the destination exclusively and writes logits, probabilities, then
// predictions with no header. A partial file is removed on any failure.
void WriteSnapshot(const std::string& path, const std::vector<float>& logits,
                   const std::vector<float>& probabilities,
                   const std::vector<std::uint8_t>& predictions) {
  const std::size_t total_bytes = logits.size() * sizeof(float) +
                                  probabilities.size() * sizeof(float) +
                                  predictions.size();
  if (total_bytes != kExpectedSnapshotBytes) {
    throw std::logic_error("snapshot host buffers have an unexpected size");
  }
  const int descriptor =
      open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (descriptor < 0) {
    throw std::runtime_error("cannot create snapshot exclusively: " + path +
                             ": " + std::strerror(errno));
  }
  bool descriptor_open = true;
  try {
    WriteAll(descriptor, logits.data(), logits.size() * sizeof(float), path);
    WriteAll(descriptor, probabilities.data(),
             probabilities.size() * sizeof(float), path);
    WriteAll(descriptor, predictions.data(), predictions.size(), path);
    if (fsync(descriptor) != 0) {
      throw std::runtime_error("cannot synchronize snapshot: " + path + ": " +
                               std::strerror(errno));
    }
    const int close_result = close(descriptor);
    descriptor_open = false;
    if (close_result != 0) {
      throw std::runtime_error("cannot close snapshot: " + path + ": " +
                               std::strerror(errno));
    }
    struct stat information;
    if (stat(path.c_str(), &information) != 0 ||
        static_cast<std::size_t>(information.st_size) !=
            kExpectedSnapshotBytes) {
      throw std::runtime_error("snapshot size must equal 810000 bytes: " +
                               path);
    }
  } catch (...) {
    if (descriptor_open) {
      static_cast<void>(close(descriptor));
    }
    static_cast<void>(unlink(path.c_str()));
    throw;
  }
}

// Owns the single non-blocking stream borrowed by the model and every launcher.
class SnapshotStream {
 public:
  SnapshotStream() : stream_(nullptr) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  ~SnapshotStream() {
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  SnapshotStream(const SnapshotStream&) = delete;
  SnapshotStream& operator=(const SnapshotStream&) = delete;

  cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_;
};

// Owns the caller-side packed host batch. Only packing and format validation
// stay on the host; all neural-network work remains on the device.
struct HostBatch {
  std::vector<std::uint8_t> images;
  std::vector<std::uint8_t> labels;
  std::vector<std::uint32_t> original_indices;
};

// Packs up to capacity entries beginning at offset in original sample order.
std::uint32_t PackSnapshotBatch(const MnistDataset& dataset,
                                const std::vector<std::uint32_t>& order,
                                std::size_t offset, std::uint32_t capacity,
                                HostBatch* batch) {
  if (offset >= order.size() || capacity == 0) {
    throw std::runtime_error("snapshot batch packing received invalid input");
  }
  const std::size_t remaining = order.size() - offset;
  const std::uint32_t actual =
      static_cast<std::uint32_t>(remaining < capacity ? remaining : capacity);
  for (std::uint32_t sample = 0; sample < actual; ++sample) {
    const std::uint32_t index = order[offset + sample];
    if (index >= dataset.sample_count) {
      throw std::out_of_range("snapshot sample index is outside the dataset");
    }
    std::memcpy(batch->images.data() + static_cast<std::size_t>(sample) *
                                         kImagePixels,
                dataset.images.data() + static_cast<std::size_t>(index) *
                                            kImagePixels,
                kImagePixels);
    batch->labels[sample] = dataset.labels[index];
    batch->original_indices[sample] = index;
  }
  return actual;
}

// Owns the caller-side device and host buffers reused by every batch.
struct SnapshotStorage {
  explicit SnapshotStorage(std::uint32_t maximum_batch_size)
      : host_batch(),
        device_images(static_cast<std::size_t>(maximum_batch_size) *
                      kImagePixels),
        device_labels(maximum_batch_size),
        device_original_indices(maximum_batch_size),
        normalized_images(static_cast<std::size_t>(maximum_batch_size) *
                          kImagePixels),
        probabilities(static_cast<std::size_t>(maximum_batch_size) *
                      kClassCount),
        predictions(maximum_batch_size),
        correct_flags(maximum_batch_size),
        correct_count(1) {
    host_batch.images.resize(static_cast<std::size_t>(maximum_batch_size) *
                             kImagePixels);
    host_batch.labels.resize(maximum_batch_size);
    host_batch.original_indices.resize(maximum_batch_size);
  }

  HostBatch host_batch;
  DeviceBuffer<std::uint8_t> device_images;
  DeviceBuffer<std::uint8_t> device_labels;
  DeviceBuffer<std::uint32_t> device_original_indices;
  DeviceBuffer<float> normalized_images;
  DeviceBuffer<float> probabilities;
  DeviceBuffer<std::uint8_t> predictions;
  DeviceBuffer<int> correct_flags;
  DeviceBuffer<int> correct_count;
};

// Validates one downloaded batch without scanning the model again on device.
// Host finite classification is only format checking for this comparison tool.
void ValidateBatch(std::size_t offset, std::uint32_t actual,
                   const std::vector<float>& logits,
                   const std::vector<float>& probabilities,
                   const std::vector<std::uint8_t>& predictions) {
  for (std::uint32_t sample = 0; sample < actual; ++sample) {
    const std::size_t sample_index = offset + sample;
    const std::size_t base = sample_index * kClassCount;
    int best = 0;
    double row_sum = 0.0;
    for (int class_index = 0; class_index < kClassCount; ++class_index) {
      const float logit = logits[base + class_index];
      if (!std::isfinite(logit)) {
        throw std::runtime_error(
            "snapshot logits contain a non-finite value at sample " +
            std::to_string(sample_index) + " class " +
            std::to_string(class_index));
      }
      if (logit > logits[base + best]) {
        best = class_index;
      }
      const float probability = probabilities[base + class_index];
      if (!std::isfinite(probability) || probability < 0.0F ||
          probability > 1.0F) {
        throw std::runtime_error(
            "snapshot probabilities are invalid at sample " +
            std::to_string(sample_index) + " class " +
            std::to_string(class_index));
      }
      row_sum += static_cast<double>(probability);
    }
    if (std::fabs(row_sum - 1.0) > 1.0e-3) {
      throw std::runtime_error(
          "snapshot probability row does not sum to one at sample " +
          std::to_string(sample_index));
    }
    if (predictions[sample_index] >= kClassCount) {
      throw std::runtime_error(
          "snapshot prediction is outside the class range at sample " +
          std::to_string(sample_index));
    }
    if (static_cast<int>(predictions[sample_index]) != best) {
      throw std::runtime_error(
          "snapshot prediction disagrees with the logits argmax at sample " +
          std::to_string(sample_index));
    }
  }
}

// Runs the official forward/softmax/argmax path in original sample order and
// accepts only complete finite host output before the destination is created.
void RunSnapshot(const std::string& data_path, const std::string& weights_path,
                 const std::string& output_path, std::uint32_t batch_size) {
  ValidateOutputPath(output_path);
  const MnistDataset dataset = LoadMnistDataset(data_path);
  RequireDatasetCount(dataset, kExpectedSampleCount, false,
                      "snapshot dataset");
  const Checkpoint checkpoint = LoadCheckpoint(weights_path);

  std::vector<std::uint32_t> order(kExpectedSampleCount);
  for (std::uint32_t index = 0; index < kExpectedSampleCount; ++index) {
    order[index] = index;
  }

  CUDA_CHECK(cudaSetDevice(0));
  SnapshotStream stream;
  SnapshotStorage storage(batch_size);
  LeNet model(static_cast<int>(batch_size), 0, stream.get());
  model.ImportParameters(checkpoint.parameters);
  model.RequireFinite("snapshot checkpoint import");

  std::vector<float> logits(kClassElementCount);
  std::vector<float> probabilities(kClassElementCount);
  std::vector<std::uint8_t> predictions(kExpectedSampleCount);

  std::size_t offset = 0;
  while (offset < order.size()) {
    if (offset > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("snapshot batch offset does not fit uint32");
    }
    const std::uint32_t actual = PackSnapshotBatch(
        dataset, order, offset, batch_size, &storage.host_batch);
    if (actual == 0 || actual > batch_size ||
        offset + actual > order.size()) {
      throw std::runtime_error("snapshot batch packing produced a bad size");
    }
    CUDA_CHECK(cudaMemcpyAsync(
        storage.device_images.get(), storage.host_batch.images.data(),
        static_cast<std::size_t>(actual) * kImagePixels *
            sizeof(std::uint8_t),
        cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(
        storage.device_labels.get(), storage.host_batch.labels.data(),
        static_cast<std::size_t>(actual) * sizeof(std::uint8_t),
        cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(
        storage.device_original_indices.get(),
        storage.host_batch.original_indices.data(),
        static_cast<std::size_t>(actual) * sizeof(std::uint32_t),
        cudaMemcpyHostToDevice, stream.get()));
    LaunchNormalizeTranslate(
        storage.device_images.get(), storage.device_original_indices.get(),
        storage.normalized_images.get(), static_cast<int>(actual), 0, 1, false,
        stream.get());
    const float* device_logits =
        model.Forward(storage.normalized_images.get(), static_cast<int>(actual));
    LaunchSoftmax(device_logits, storage.probabilities.get(),
                  static_cast<int>(actual), kClassCount, stream.get());
    LaunchArgmaxAndCountCorrect(
        device_logits, storage.device_labels.get(),
        storage.predictions.get(), storage.correct_flags.get(),
        storage.correct_count.get(), static_cast<int>(actual), kClassCount,
        stream.get());
    const std::size_t batch_elements =
        static_cast<std::size_t>(actual) * kClassCount;
    CUDA_CHECK(cudaMemcpyAsync(
        logits.data() + offset * kClassCount, device_logits,
        batch_elements * sizeof(float), cudaMemcpyDeviceToHost, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(
        probabilities.data() + offset * kClassCount,
        storage.probabilities.get(), batch_elements * sizeof(float),
        cudaMemcpyDeviceToHost, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(
        predictions.data() + offset, storage.predictions.get(),
        static_cast<std::size_t>(actual) * sizeof(std::uint8_t),
        cudaMemcpyDeviceToHost, stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    ValidateBatch(offset, actual, logits, probabilities, predictions);
    offset += actual;
  }
  if (offset != kExpectedSampleCount) {
    throw std::runtime_error("snapshot did not cover exactly 10000 samples");
  }

  WriteSnapshot(output_path, logits, probabilities, predictions);
  std::printf(
      "event=snapshot status=pass samples=%u batch=%u bytes=%zu\n",
      static_cast<unsigned int>(kExpectedSampleCount),
      static_cast<unsigned int>(batch_size), kExpectedSnapshotBytes);
}

}  // namespace

// Validates arguments and paths, runs the snapshot, and maps failures to the
// project exit codes. No CPU neural-network path exists in this tool.
int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      std::fprintf(stderr,
                   "error: expected exactly four arguments: "
                   "DATA WEIGHTS OUTPUT BATCH\n%s",
                   UsageText());
      return kUsageError;
    }
    const std::string data_path(argv[1]);
    const std::string weights_path(argv[2]);
    const std::string output_path(argv[3]);
    const std::uint32_t batch_size = ParseBatchSize(argv[4]);
    RunSnapshot(data_path, weights_path, output_path, batch_size);
    return kSuccess;
  } catch (const std::invalid_argument& error) {
    std::fprintf(stderr, "error: %s\n%s", error.what(), UsageText());
    return kUsageError;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return kRuntimeError;
  }
}
