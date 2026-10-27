// Implements the course-only training workflow over supplied raw FP32 images
// and int32 labels. It reuses the original LeNet forward/backward/AdamW path,
// original split/shuffle helpers, and the reviewed course model bundle, but it
// never loads an external test dataset or calls the old RunTrain workflow.
#include "course/training.h"

#include "checkpoint.h"
#include "cuda_check.h"
#include "course/gpu_ops.h"
#include "course/model_io.h"
#include "layers.h"
#include "parameters.h"
#include "random.h"
#include "tensor.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#endif

namespace {

constexpr std::size_t kImagePixels = 28 * 28;
constexpr int kClassCount = 10;
constexpr std::uint32_t kChannels = 1;
constexpr std::uint32_t kHeight = 28;
constexpr std::uint32_t kWidth = 28;
constexpr std::uint32_t kClasses = 10;
constexpr std::uint32_t kMinimumEpochs = 1;
constexpr std::uint32_t kMaximumEpochs = 1000;
constexpr float kNormalizationMean = 0.1307F;
constexpr float kNormalizationStddev = 0.3081F;
const char kMetadataName[] = "course.meta";
const char kCrcPrefix[] = "checkpoint_crc32=";

using TrainingClock = std::chrono::steady_clock;

// Returns the wall time between two steady-clock samples in milliseconds.
double ElapsedMilliseconds(TrainingClock::time_point start,
                           TrainingClock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// Compares two FP32 values by representation.
bool SameBits(float left, float right) {
  std::uint32_t left_bits = 0;
  std::uint32_t right_bits = 0;
  std::memcpy(&left_bits, &left, sizeof(left_bits));
  std::memcpy(&right_bits, &right, sizeof(right_bits));
  return left_bits == right_bits;
}

// Returns true only for a known InputMode enumerator.
bool KnownInputMode(course::InputMode input_mode) {
  switch (input_mode) {
    case course::InputMode::kUnit:
    case course::InputMode::kByte:
    case course::InputMode::kNormalized:
      return true;
  }
  return false;
}

// Returns one documented mode spelling for diagnostics.
const char* InputModeName(course::InputMode input_mode) {
  switch (input_mode) {
    case course::InputMode::kUnit:
      return "unit";
    case course::InputMode::kByte:
      return "byte";
    case course::InputMode::kNormalized:
      return "normalized";
  }
  throw std::runtime_error("unsupported course input mode");
}

// Validates every direct-API option before work or file access. Training has no
// fixed upper batch limit in the contract; the batch must merely fit the
// existing LeNet int capacities. Large valid batches may therefore fail
// normally on CUDA memory.
void ValidateOptions(const course::TrainOptions& options) {
  if (options.images_path.empty()) {
    throw std::invalid_argument("training images path must not be empty");
  }
  if (options.labels_path.empty()) {
    throw std::invalid_argument("training labels path must not be empty");
  }
  if (options.work_dir.empty()) {
    throw std::invalid_argument("training work directory must not be empty");
  }

  const course::Shape& shape = options.shape;
  if (shape.n < 2) {
    throw std::invalid_argument("training requires N >= 2");
  }
  if (shape.c != kChannels || shape.h != kHeight || shape.w != kWidth ||
      shape.k != kClasses) {
    throw std::invalid_argument("only C=1, H=28, W=28, K=10 are supported");
  }
  if (options.epochs < kMinimumEpochs ||
      options.epochs > kMaximumEpochs) {
    throw std::invalid_argument("training epochs must be in [1, 1000]");
  }
  if (options.batch_size == 0 ||
      options.batch_size >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(
        "training batch size must be positive and fit an int");
  }
  if (!KnownInputMode(options.input_mode)) {
    throw std::invalid_argument("training input mode is unsupported");
  }
}

// Creates the caller-selected work directory when missing. An existing
// directory is accepted untouched; unrelated entries are never removed.
void EnsureDirectory(const std::string& path) {
  struct stat information;
  if (::stat(path.c_str(), &information) == 0) {
    if (!S_ISDIR(information.st_mode)) {
      throw std::runtime_error(path + ": work path is not a directory");
    }
    return;
  }
  if (errno != ENOENT) {
    throw std::runtime_error(path + ": could not inspect work directory");
  }

#ifdef _WIN32
  const int status = ::_mkdir(path.c_str());
#else
  const int status = ::mkdir(path.c_str(), 0700);
#endif
  if (status != 0) {
    if (errno == EEXIST && ::stat(path.c_str(), &information) == 0 &&
        S_ISDIR(information.st_mode)) {
      return;
    }
    throw std::runtime_error(path + ": could not create work directory");
  }
}

// Owns the single non-blocking stream used by one training workflow.
class TrainingStream {
 public:
  TrainingStream() : stream_(nullptr) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  ~TrainingStream() {
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  TrainingStream(const TrainingStream&) = delete;
  TrainingStream& operator=(const TrainingStream&) = delete;

  cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_;
};

// Owns the host packing buffers reused by every batch.
struct HostBatch {
  explicit HostBatch(std::size_t maximum_batch_size) {
    images.reserve(maximum_batch_size * kImagePixels);
    labels.reserve(maximum_batch_size);
    original_indices.reserve(maximum_batch_size);
    size = 0;
  }

  std::vector<float> images;
  std::vector<std::uint8_t> labels;
  std::vector<std::uint32_t> original_indices;
  std::size_t size;
};

// Owns every reusable device buffer for one training workflow. LeNet owns the
// parameters, gradients, moments, activations, and pooling winners separately.
struct TrainingStorage {
  explicit TrainingStorage(std::size_t maximum_batch_size)
      : host_batch(maximum_batch_size),
        device_images(maximum_batch_size * kImagePixels),
        device_labels(maximum_batch_size),
        device_original_indices(maximum_batch_size),
        normalized_images(maximum_batch_size * kImagePixels),
        probabilities(maximum_batch_size * kClassCount),
        per_sample_losses(maximum_batch_size),
        mean_loss(1),
        logits_gradient(maximum_batch_size * kClassCount),
        predictions(maximum_batch_size),
        correct_flags(maximum_batch_size),
        correct_count(1),
        first_non_finite(1) {}

  HostBatch host_batch;
  DeviceBuffer<float> device_images;
  DeviceBuffer<std::uint8_t> device_labels;
  DeviceBuffer<std::uint32_t> device_original_indices;
  DeviceBuffer<float> normalized_images;
  DeviceBuffer<float> probabilities;
  DeviceBuffer<float> per_sample_losses;
  DeviceBuffer<float> mean_loss;
  DeviceBuffer<float> logits_gradient;
  DeviceBuffer<std::uint8_t> predictions;
  DeviceBuffer<int> correct_flags;
  DeviceBuffer<int> correct_count;
  DeviceBuffer<int> first_non_finite;
};

// Packs up to capacity selected samples beginning at offset, preserving the
// exact FP32 pixel payload and each sample's original dataset index.
void PackBatch(const std::vector<float>& raw_images,
               const std::vector<std::uint8_t>& raw_labels,
               const std::vector<std::uint32_t>& order, std::size_t offset,
               std::size_t capacity, HostBatch* batch) {
  if (batch == nullptr || capacity == 0 || offset >= order.size()) {
    throw std::runtime_error("training batch packing arguments are invalid");
  }
  const std::size_t actual = std::min(capacity, order.size() - offset);
  batch->images.resize(actual * kImagePixels);
  batch->labels.resize(actual);
  batch->original_indices.resize(actual);
  for (std::size_t sample = 0; sample < actual; ++sample) {
    const std::uint32_t original = order[offset + sample];
    if (original >= raw_labels.size() ||
        static_cast<std::size_t>(original) * kImagePixels + kImagePixels >
            raw_images.size()) {
      throw std::runtime_error(
          "training batch index is outside the supplied dataset");
    }
    std::memcpy(batch->images.data() + sample * kImagePixels,
                raw_images.data() +
                    static_cast<std::size_t>(original) * kImagePixels,
                kImagePixels * sizeof(float));
    batch->labels[sample] = raw_labels[original];
    batch->original_indices[sample] = original;
  }
  batch->size = actual;
}

// Uploads one packed host batch to the reusable device buffers.
void CopyBatchToDevice(const HostBatch& batch, TrainingStorage* storage,
                       cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(storage->device_images.get(), batch.images.data(),
                             batch.size * kImagePixels * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(storage->device_labels.get(), batch.labels.data(),
                             batch.size * sizeof(std::uint8_t),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(storage->device_original_indices.get(),
                             batch.original_indices.data(),
                             batch.size * sizeof(std::uint32_t),
                             cudaMemcpyHostToDevice, stream));
}

// Scans a device buffer for non-finite values in INT_MAX-sized chunks so an
// arbitrary positive batch capacity remains representable by the original
// finite-scan launcher. The stream is synchronized before returning.
void RequireFiniteDevice(const float* values, std::size_t count,
                         const std::string& phase, int* first_non_finite,
                         cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  const std::size_t chunk_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  for (std::size_t offset = 0; offset < count; offset += chunk_limit) {
    const std::size_t chunk = std::min(chunk_limit, count - offset);
    LaunchFindFirstNonFinite(values + offset, chunk, first_non_finite, stream);
    int first_bad = 0;
    CUDA_CHECK(cudaMemcpyAsync(&first_bad, first_non_finite,
                               sizeof(first_bad), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (first_bad != static_cast<int>(chunk)) {
      std::ostringstream message;
      message << phase << " contains a non-finite value at index "
              << offset + static_cast<std::size_t>(first_bad);
      throw std::runtime_error(message.str());
    }
  }
}

// Uploads one host batch, prepares it on the GPU with the persisted mode, and
// checks the normalized output. Augmentation uses the original dataset indices.
void PrepareBatch(HostBatch* batch, std::uint64_t seed, std::uint32_t epoch,
                  bool augment, course::InputMode input_mode,
                  TrainingStorage* storage, cudaStream_t stream) {
  CopyBatchToDevice(*batch, storage, stream);
  course::LaunchPrepareInput(
      storage->device_images.get(),
      augment ? storage->device_original_indices.get() : nullptr,
      storage->normalized_images.get(), static_cast<int>(batch->size),
      input_mode, seed, epoch, augment, stream);
  RequireFiniteDevice(
      storage->normalized_images.get(), batch->size * kImagePixels,
      augment ? "course training normalized input"
              : "course validation normalized input",
      storage->first_non_finite.get(), stream);
}

// Evaluates exactly the supplied validation order without augmentation and
// returns the number of correct predictions. The labels belong to the supplied
// training split; no external test labels exist in this workflow.
std::uint64_t EvaluateValidation(
    const std::vector<float>& raw_images,
    const std::vector<std::uint8_t>& raw_labels,
    const std::vector<std::uint32_t>& order, std::uint32_t batch_size,
    course::InputMode input_mode, LeNet* model, TrainingStorage* storage,
    cudaStream_t stream) {
  if (order.empty()) {
    throw std::runtime_error("course validation partition must not be empty");
  }
  std::uint64_t correct = 0;
  std::size_t offset = 0;
  while (offset < order.size()) {
    PackBatch(raw_images, raw_labels, order, offset, batch_size,
              &storage->host_batch);
    const std::size_t actual = storage->host_batch.size;
    PrepareBatch(&storage->host_batch, 0, 1, false, input_mode, storage,
                 stream);
    const float* const logits =
        model->Forward(storage->normalized_images.get(),
                       static_cast<int>(actual));
    model->RequireFinite("course validation forward");
    LaunchArgmaxAndCountCorrect(
        logits, storage->device_labels.get(), storage->predictions.get(),
        storage->correct_flags.get(), storage->correct_count.get(),
        static_cast<int>(actual), kClassCount, stream);
    int batch_correct = 0;
    CUDA_CHECK(cudaMemcpyAsync(&batch_correct, storage->correct_count.get(),
                               sizeof(batch_correct), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (batch_correct < 0 || batch_correct > static_cast<int>(actual)) {
      throw std::runtime_error(
          "GPU validation correct count is outside the batch bounds");
    }
    correct += static_cast<std::uint64_t>(batch_correct);
    offset += actual;
  }
  if (offset != order.size() || correct > order.size()) {
    throw std::runtime_error(
        "validation batches did not cover the full partition cleanly");
  }
  return correct;
}

// Returns true only when every canonical parameter value is bit-identical.
bool ParametersMatch(const ParameterSet& left, const ParameterSet& right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t tensor = 0; tensor < left.size(); ++tensor) {
    if (left[tensor].values.size() != right[tensor].values.size()) {
      return false;
    }
    for (std::size_t index = 0; index < left[tensor].values.size(); ++index) {
      if (!SameBits(left[tensor].values[index],
                    right[tensor].values[index])) {
        return false;
      }
    }
  }
  return true;
}

// Reads the CRC field from the already validated course.meta after
// LoadCourseModel has verified that it matches the checkpoint bytes.
std::string ReadValidatedCrc(const std::string& directory) {
  const std::string path = directory + "/" + kMetadataName;
  std::ifstream input(path.c_str(), std::ios::binary);
  if (!input) {
    throw std::runtime_error(path + ": could not read validated metadata");
  }
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t prefix_length = sizeof(kCrcPrefix) - 1;
    if (line.size() >= prefix_length &&
        line.compare(0, prefix_length, kCrcPrefix) == 0) {
      const std::string value = line.substr(prefix_length);
      if (value.size() != 8) {
        break;
      }
      return value;
    }
  }
  throw std::runtime_error(path +
                           ": validated checkpoint_crc32 is unavailable");
}

// Emits one classic-locale epoch diagnostic. It deliberately avoids the final
// completion-record keys so the completion line is unambiguous.
void WriteEpochRecord(std::ostream& diagnostics, std::uint32_t epoch,
                      float weighted_loss, float validation_accuracy,
                      double elapsed_ms) {
  std::ostringstream record;
  record.imbue(std::locale::classic());
  record << std::setprecision(std::numeric_limits<float>::max_digits10)
         << "event=course_train_epoch epoch=" << epoch
         << " weighted_loss=" << weighted_loss
         << " val_accuracy=" << validation_accuracy
         << std::fixed << std::setprecision(3)
         << " elapsed_ms=" << elapsed_ms << '\n';
  const std::string text = record.str();
  diagnostics.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!diagnostics) {
    throw std::runtime_error("could not write course training diagnostics");
  }
}

// Emits the required final completion record with classic-locale numbers.
void WriteCompletionRecord(std::ostream& diagnostics, std::uint32_t epochs,
                           std::uint32_t best_epoch,
                           float validation_accuracy,
                           course::InputMode input_mode,
                           const std::string& checkpoint_crc32) {
  std::ostringstream record;
  record.imbue(std::locale::classic());
  record << "event=course_train status=pass epochs=" << epochs
         << " best_epoch=" << best_epoch
         << std::setprecision(std::numeric_limits<float>::max_digits10)
         << " validation_accuracy=" << validation_accuracy
         << " input_mode=" << InputModeName(input_mode)
         << " checkpoint_crc32=" << checkpoint_crc32 << '\n';
  const std::string text = record.str();
  diagnostics.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!diagnostics) {
    throw std::runtime_error("could not write course training diagnostics");
  }
}

}  // namespace

namespace course {

std::uint32_t CourseValidationCount(std::uint32_t sample_count) {
  if (sample_count < 2) {
    throw std::invalid_argument("course training requires N >= 2");
  }
  if (sample_count == 60000) {
    return 5000;
  }
  const std::uint32_t count = sample_count / 5;
  return count == 0 ? 1 : count;
}

AdamWConfig CourseAdamWConfig() {
  return AdamWConfig{0.9F, 0.999F, 1.0e-8F, 1.0e-4F};
}

float CourseLearningRate(std::uint32_t one_based_epoch) {
  if (one_based_epoch == 0) {
    throw std::invalid_argument("learning-rate epoch must be one-based");
  }
  if (one_based_epoch <= 12) {
    return 1.0e-3F;
  }
  if (one_based_epoch <= 17) {
    return 1.0e-4F;
  }
  return 1.0e-5F;
}

bool BestValidationSelection::Consider(std::uint32_t one_based_epoch,
                                       float validation_accuracy) {
  if (one_based_epoch == 0) {
    throw std::invalid_argument("best validation epoch must be one-based");
  }
  if (!std::isfinite(validation_accuracy) || validation_accuracy < 0.0F ||
      validation_accuracy > 1.0F) {
    throw std::invalid_argument(
        "validation accuracy must be finite and in [0, 1]");
  }
  if (!(validation_accuracy > best_accuracy_)) {
    return false;
  }
  best_epoch_ = one_based_epoch;
  best_accuracy_ = validation_accuracy;
  return true;
}

int RunCourseTrain(const TrainOptions& options, std::ostream& diagnostics) {
  ValidateOptions(options);

  const std::vector<float> raw_images =
      ReadImages(options.images_path, options.shape, options.input_mode);
  const std::vector<std::uint8_t> raw_labels =
      ReadLabels(options.labels_path, options.shape);
  EnsureDirectory(options.work_dir);

  const DatasetSplit split = MakeTrainValidationSplit(
      options.shape.n, CourseValidationCount(options.shape.n), options.seed);
  if (split.training_indices.empty() || split.validation_indices.empty()) {
    throw std::runtime_error("course training split is empty");
  }

  TrainingStream stream;
  TrainingStorage storage(options.batch_size);
  LeNet model(static_cast<int>(options.batch_size), options.seed, stream.get());
  const AdamWConfig optimizer = CourseAdamWConfig();

  std::vector<std::uint32_t> training_order;
  training_order.reserve(split.training_indices.size());

  Checkpoint best_checkpoint;
  best_checkpoint.metadata.best_epoch = 0;
  best_checkpoint.metadata.validation_accuracy = 0.0F;
  best_checkpoint.metadata.normalization_mean = kNormalizationMean;
  best_checkpoint.metadata.normalization_stddev = kNormalizationStddev;
  best_checkpoint.parameters = CreateLenetParameters();

  BestValidationSelection best;
  std::uint64_t global_step = 0;

  for (std::uint32_t epoch = 1; epoch <= options.epochs; ++epoch) {
    const TrainingClock::time_point epoch_start = TrainingClock::now();
    ShuffledTrainingIndices(split.training_indices, options.seed, epoch,
                            &training_order);

    double weighted_loss = 0.0;
    std::uint64_t trained_samples = 0;
    std::size_t offset = 0;
    while (offset < training_order.size()) {
      PackBatch(raw_images, raw_labels, training_order, offset,
                options.batch_size, &storage.host_batch);
      const std::size_t actual = storage.host_batch.size;
      PrepareBatch(&storage.host_batch, options.seed, epoch, true,
                   options.input_mode, &storage, stream.get());

      const float* const logits =
          model.Forward(storage.normalized_images.get(),
                        static_cast<int>(actual));
      model.RequireFinite("course training forward");

      LaunchSoftmaxCrossEntropy(
          logits, storage.device_labels.get(), storage.probabilities.get(),
          storage.per_sample_losses.get(), storage.mean_loss.get(),
          storage.logits_gradient.get(), static_cast<int>(actual),
          kClassCount, stream.get());
      RequireFiniteDevice(storage.probabilities.get(),
                          actual * kClassCount,
                          "course training probabilities",
                          storage.first_non_finite.get(), stream.get());
      RequireFiniteDevice(storage.per_sample_losses.get(), actual,
                          "course training per-sample loss",
                          storage.first_non_finite.get(), stream.get());
      RequireFiniteDevice(storage.mean_loss.get(), 1,
                          "course training mean loss",
                          storage.first_non_finite.get(), stream.get());
      RequireFiniteDevice(storage.logits_gradient.get(),
                          actual * kClassCount,
                          "course training logits gradient",
                          storage.first_non_finite.get(), stream.get());

      float batch_loss = 0.0F;
      CUDA_CHECK(cudaMemcpyAsync(&batch_loss, storage.mean_loss.get(),
                                 sizeof(batch_loss), cudaMemcpyDeviceToHost,
                                 stream.get()));
      CUDA_CHECK(cudaStreamSynchronize(stream.get()));

      model.Backward(storage.logits_gradient.get(),
                     static_cast<int>(actual));
      model.RequireFinite("course training backward");

      ++global_step;
      model.AdamWStep(global_step, CourseLearningRate(epoch), optimizer);
      model.RequireFinite("course training optimizer step");

      weighted_loss += static_cast<double>(batch_loss) *
                       static_cast<double>(actual);
      trained_samples += static_cast<std::uint64_t>(actual);
      offset += actual;
    }
    if (trained_samples != split.training_indices.size()) {
      throw std::runtime_error(
          "training batches did not cover every training sample");
    }
    const float epoch_loss =
        static_cast<float>(weighted_loss /
                           static_cast<double>(trained_samples));
    if (!std::isfinite(epoch_loss)) {
      throw std::runtime_error("sample-weighted epoch loss is not finite");
    }

    const std::uint64_t correct =
        EvaluateValidation(raw_images, raw_labels, split.validation_indices,
                           options.batch_size, options.input_mode, &model,
                           &storage, stream.get());
    const float validation_accuracy =
        static_cast<float>(correct) /
        static_cast<float>(split.validation_indices.size());
    if (!std::isfinite(validation_accuracy)) {
      throw std::runtime_error("validation accuracy is not finite");
    }

    // Replace the published bundle only on strict validation improvement.
    if (best.Consider(epoch, validation_accuracy)) {
      best_checkpoint.metadata.best_epoch = epoch;
      best_checkpoint.metadata.validation_accuracy = validation_accuracy;
      model.ExportParameters(&best_checkpoint.parameters);
      SaveCourseModel(options.work_dir, best_checkpoint, options.input_mode);
    }

    WriteEpochRecord(diagnostics, epoch, epoch_loss, validation_accuracy,
                     ElapsedMilliseconds(epoch_start, TrainingClock::now()));
  }

  if (best.best_epoch() == 0) {
    throw std::runtime_error("course training produced no best checkpoint");
  }

  // Reload and validate the exact published best bundle before success.
  const CourseModel reloaded = LoadCourseModel(options.work_dir);
  if (reloaded.input_mode != options.input_mode) {
    throw std::runtime_error(
        "reloaded course bundle input mode does not match training");
  }
  if (reloaded.checkpoint.metadata.best_epoch != best.best_epoch() ||
      !SameBits(reloaded.checkpoint.metadata.validation_accuracy,
                best.best_accuracy())) {
    throw std::runtime_error(
        "reloaded course bundle best metadata does not match selection");
  }
  if (!SameBits(reloaded.checkpoint.metadata.normalization_mean,
                kNormalizationMean) ||
      !SameBits(reloaded.checkpoint.metadata.normalization_stddev,
                kNormalizationStddev)) {
    throw std::runtime_error(
        "reloaded course bundle normalization contract is invalid");
  }
  if (!ParametersMatch(reloaded.checkpoint.parameters,
                       best_checkpoint.parameters)) {
    throw std::runtime_error(
        "reloaded course bundle parameters do not match the best export");
  }
  model.ImportParameters(reloaded.checkpoint.parameters);
  model.RequireFinite("reloaded course best bundle");

  const std::string checkpoint_crc32 = ReadValidatedCrc(options.work_dir);
  WriteCompletionRecord(diagnostics, options.epochs,
                        reloaded.checkpoint.metadata.best_epoch,
                        reloaded.checkpoint.metadata.validation_accuracy,
                        reloaded.input_mode, checkpoint_crc32);
  return 0;
}

}  // namespace course
