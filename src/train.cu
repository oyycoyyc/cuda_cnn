#include "train.h"

#include "checkpoint.h"
#include "cuda_check.h"
#include "dataset.h"
#include "layers.h"
#include "lenet.h"
#include "random.h"
#include "reporting.h"
#include "tensor.h"
#include "training_data.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

const std::size_t kImagePixels = 28 * 28;
const int kClassCount = 10;
const std::uint32_t kEvaluationBatchSize = 128;
const float kNormalizationMean = 0.1307F;
const float kNormalizationStddev = 0.3081F;

class WorkflowStream {
 public:
  WorkflowStream() : stream_(nullptr) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  ~WorkflowStream() {
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  WorkflowStream(const WorkflowStream&) = delete;
  WorkflowStream& operator=(const WorkflowStream&) = delete;

  cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_;
};

class WorkflowEvents {
 public:
  WorkflowEvents() : start_(nullptr), stop_(nullptr) {
    CUDA_CHECK(cudaEventCreate(&start_));
    try {
      CUDA_CHECK(cudaEventCreate(&stop_));
    } catch (...) {
      static_cast<void>(cudaEventDestroy(start_));
      start_ = nullptr;
      throw;
    }
  }

  ~WorkflowEvents() {
    if (stop_ != nullptr) {
      static_cast<void>(cudaEventDestroy(stop_));
    }
    if (start_ != nullptr) {
      static_cast<void>(cudaEventDestroy(start_));
    }
  }

  WorkflowEvents(const WorkflowEvents&) = delete;
  WorkflowEvents& operator=(const WorkflowEvents&) = delete;

  void RecordStart(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(start_, stream));
  }

  void RecordStop(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(stop_, stream));
  }

  float ElapsedMilliseconds() {
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float elapsed = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_, stop_));
    if (!std::isfinite(elapsed) || elapsed < 0.0F) {
      throw std::runtime_error("CUDA Event forward timing is not finite");
    }
    return elapsed;
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

struct WorkflowStorage {
  explicit WorkflowStorage(std::uint32_t maximum_batch_size)
      : host_batch(),
        device_images(static_cast<std::size_t>(maximum_batch_size) *
                      kImagePixels),
        device_labels(maximum_batch_size),
        device_original_indices(maximum_batch_size),
        normalized_images(static_cast<std::size_t>(maximum_batch_size) *
                          kImagePixels),
        probabilities(static_cast<std::size_t>(maximum_batch_size) *
                      kClassCount),
        per_sample_losses(maximum_batch_size),
        mean_loss(1),
        logits_gradient(static_cast<std::size_t>(maximum_batch_size) *
                        kClassCount),
        predictions(maximum_batch_size),
        correct_flags(maximum_batch_size),
        correct_count(1),
        first_non_finite(1),
        events() {
    host_batch.images.resize(static_cast<std::size_t>(maximum_batch_size) *
                             kImagePixels);
    host_batch.labels.resize(maximum_batch_size);
    host_batch.original_indices.resize(maximum_batch_size);
    host_batch.size = 0;
  }

  HostBatch host_batch;
  DeviceBuffer<std::uint8_t> device_images;
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
  WorkflowEvents events;
};

struct EvaluationResult {
  std::uint64_t correct;
  double total_forward_ms;
  std::uint64_t timed_images;
  std::uint32_t timed_batches;
};

std::string ParentPath(const std::string& path) {
  const std::string::size_type separator = path.find_last_of("/\\");
  if (separator == std::string::npos) {
    return ".";
  }
  if (separator == 0) {
    return path.substr(0, 1);
  }
#ifdef _WIN32
  if (separator == 2 && path.size() >= 3 && path[1] == ':') {
    return path.substr(0, 3);
  }
#endif
  return path.substr(0, separator);
}

void RequireExistingOutputParent(const std::string& path) {
  if (path.empty()) {
    throw std::invalid_argument("output path must not be empty");
  }
  const std::string parent = ParentPath(path);
  struct stat information;
  if (stat(parent.c_str(), &information) != 0 ||
      (information.st_mode & S_IFMT) != S_IFDIR) {
    throw std::runtime_error("output parent directory does not exist: " +
                             parent);
  }
}

void SelectAndReportDevice(int requested, std::ostream& output) {
  int count = 0;
  const cudaError_t count_result = cudaGetDeviceCount(&count);
  if (count_result != cudaSuccess) {
    std::ostringstream message;
    message << "CUDA device query failed: code="
            << static_cast<int>(count_result)
            << " text=" << cudaGetErrorString(count_result);
    throw std::runtime_error(message.str());
  }
  if (count == 0) {
    throw std::runtime_error("no CUDA devices are available");
  }
  if (requested < 0 || requested >= count) {
    std::ostringstream message;
    message << "requested CUDA device " << requested
            << " is unavailable; available device count=" << count;
    throw std::runtime_error(message.str());
  }
  CUDA_CHECK(cudaSetDevice(requested));
  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDeviceProperties(&properties, requested));
  std::ostringstream record;
  record.imbue(std::locale::classic());
  record << "event=device index=" << requested << " name=" << properties.name
         << " compute_capability=" << properties.major << '.'
         << properties.minor << '\n';
  const std::string text = record.str();
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
}

void CopyBatchToDevice(const HostBatch& batch, WorkflowStorage* storage,
                       cudaStream_t stream) {
  const std::size_t image_bytes =
      static_cast<std::size_t>(batch.size) * kImagePixels;
  CUDA_CHECK(cudaMemcpyAsync(storage->device_images.get(), batch.images.data(),
                             image_bytes * sizeof(std::uint8_t),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(storage->device_labels.get(), batch.labels.data(),
                             batch.size * sizeof(std::uint8_t),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(storage->device_original_indices.get(),
                             batch.original_indices.data(),
                             batch.size * sizeof(std::uint32_t),
                             cudaMemcpyHostToDevice, stream));
}

void RequireFiniteDevice(const float* values, std::size_t count,
                         const std::string& phase, WorkflowStorage* storage,
                         cudaStream_t stream) {
  LaunchFindFirstNonFinite(values, count, storage->first_non_finite.get(),
                           stream);
  int first_bad = 0;
  CUDA_CHECK(cudaMemcpyAsync(&first_bad, storage->first_non_finite.get(),
                             sizeof(first_bad), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (first_bad != static_cast<int>(count)) {
    std::ostringstream message;
    message << phase << " contains a non-finite value at index " << first_bad;
    throw std::runtime_error(message.str());
  }
}

void NormalizeBatch(const HostBatch& batch, std::uint64_t seed,
                    std::uint32_t epoch, bool augment,
                    WorkflowStorage* storage, cudaStream_t stream) {
  CopyBatchToDevice(batch, storage, stream);
  LaunchNormalizeTranslate(
      storage->device_images.get(), storage->device_original_indices.get(),
      storage->normalized_images.get(), static_cast<int>(batch.size), seed,
      epoch, augment, stream);
  RequireFiniteDevice(storage->normalized_images.get(),
                      static_cast<std::size_t>(batch.size) * kImagePixels,
                      augment ? "training normalized input"
                              : "evaluation normalized input",
                      storage, stream);
}

int CopyCorrectCount(WorkflowStorage* storage, cudaStream_t stream) {
  int correct = 0;
  CUDA_CHECK(cudaMemcpyAsync(&correct, storage->correct_count.get(),
                             sizeof(correct), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return correct;
}

EvaluationResult EvaluateBatches(const MnistDataset& dataset,
                                 const std::vector<std::uint32_t>& order,
                                 std::uint32_t batch_size, bool measure,
                                 LeNet* model, WorkflowStorage* storage,
                                 cudaStream_t stream) {
  EvaluationResult result{0, 0.0, 0, 0};
  bool first_batch = true;
  for (std::uint32_t offset = 0; offset < order.size(); offset += batch_size) {
    const std::uint32_t actual =
        PackBatch(dataset, order, offset, batch_size, &storage->host_batch);
    NormalizeBatch(storage->host_batch, 0, 1, false, storage, stream);

    // The warm-up forward still contributes metrics once; only its timing is
    // excluded. This prevents either dropping or replaying the first samples.
    const bool timed = measure && !first_batch;
    if (timed) {
      storage->events.RecordStart(stream);
    }
    const float* logits =
        model->Forward(storage->normalized_images.get(), static_cast<int>(actual));
    if (timed) {
      storage->events.RecordStop(stream);
    }
    model->RequireFinite(measure ? "evaluation forward" : "validation forward");
    LaunchSoftmax(logits, storage->probabilities.get(),
                  static_cast<int>(actual), kClassCount, stream);
    RequireFiniteDevice(storage->probabilities.get(),
                        static_cast<std::size_t>(actual) * kClassCount,
                        measure ? "evaluation probabilities"
                                : "validation probabilities",
                        storage, stream);
    LaunchArgmaxAndCountCorrect(
        logits, storage->device_labels.get(), storage->predictions.get(),
        storage->correct_flags.get(), storage->correct_count.get(),
        static_cast<int>(actual), kClassCount, stream);
    const int batch_correct = CopyCorrectCount(storage, stream);
    if (batch_correct < 0 || batch_correct > static_cast<int>(actual)) {
      throw std::runtime_error("GPU correct count is outside batch bounds");
    }
    result.correct += static_cast<std::uint32_t>(batch_correct);
    if (timed) {
      result.total_forward_ms += storage->events.ElapsedMilliseconds();
      result.timed_images += actual;
      ++result.timed_batches;
    }
    first_batch = false;
  }
  return result;
}

void PrintFinalTestSummary(std::ostream& output, std::uint32_t samples,
                           float accuracy,
                           const CheckpointMetadata& metadata) {
  if (!std::isfinite(accuracy)) {
    throw std::runtime_error("final test accuracy is not finite");
  }
  std::ostringstream record;
  record.imbue(std::locale::classic());
  record << std::fixed << "event=final_test samples=" << samples
         << " final_test_accuracy=" << std::setprecision(6) << accuracy
         << " best_epoch=" << metadata.best_epoch
         << " validation_accuracy=" << metadata.validation_accuracy << '\n';
  const std::string text = record.str();
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::vector<std::uint32_t> CanonicalOrder(std::uint32_t count) {
  std::vector<std::uint32_t> order(count);
  for (std::uint32_t index = 0; index < count; ++index) {
    order[index] = index;
  }
  return order;
}

}  // namespace

float LearningRateForEpoch(std::uint32_t one_based_epoch) {
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

int RunTrain(const TrainOptions& options, std::ostream& output,
             std::ostream& error) {
  static_cast<void>(error);
  const MnistDataset training_dataset = LoadMnistDataset(options.train_path);
  RequireDatasetCount(training_dataset, 60000, options.allow_nonstandard_count,
                      "training dataset");
  const MnistDataset test_dataset = LoadMnistDataset(options.test_path);
  RequireDatasetCount(test_dataset, 10000, options.allow_nonstandard_count,
                      "test dataset");
  RequireExistingOutputParent(options.output_path);
  if (options.epochs == 0 || options.epochs > 1000 ||
      options.batch_size == 0 ||
      options.batch_size > 1024) {
    throw std::invalid_argument("training epochs and batch size are invalid");
  }
  const DatasetSplit split = MakeWorkflowSplit(
      training_dataset.sample_count, options.seed,
      options.allow_nonstandard_count);
  const std::vector<std::uint32_t> test_order =
      CanonicalOrder(test_dataset.sample_count);

  SelectAndReportDevice(options.device, output);
  WorkflowStream stream;
  WorkflowStorage storage(options.batch_size);
  LeNet model(static_cast<int>(options.batch_size), options.seed, stream.get());
  const AdamWConfig optimizer{0.9F, 0.999F, 1.0e-8F, 1.0e-4F};
  std::uint64_t global_step = 0;
  float best_accuracy = -1.0F;

  for (std::uint32_t epoch = 1; epoch <= options.epochs; ++epoch) {
    const std::chrono::steady_clock::time_point epoch_start =
        std::chrono::steady_clock::now();
    const std::vector<std::uint32_t> training_order =
        ShuffledTrainingIndices(split.training_indices, options.seed, epoch);
    double weighted_loss = 0.0;
    std::uint64_t trained_samples = 0;
    for (std::uint32_t offset = 0; offset < training_order.size();
         offset += options.batch_size) {
      const std::uint32_t actual = PackBatch(
          training_dataset, training_order, offset, options.batch_size,
          &storage.host_batch);
      NormalizeBatch(storage.host_batch, options.seed, epoch, true, &storage,
                     stream.get());
      const float* logits = model.Forward(storage.normalized_images.get(),
                                          static_cast<int>(actual));
      model.RequireFinite("training forward");
      LaunchSoftmaxCrossEntropy(
          logits, storage.device_labels.get(), storage.probabilities.get(),
          storage.per_sample_losses.get(), storage.mean_loss.get(),
          storage.logits_gradient.get(), static_cast<int>(actual), kClassCount,
          stream.get());
      RequireFiniteDevice(storage.probabilities.get(),
                          static_cast<std::size_t>(actual) * kClassCount,
                          "training probabilities", &storage, stream.get());
      RequireFiniteDevice(storage.per_sample_losses.get(), actual,
                          "training per-sample loss", &storage, stream.get());
      RequireFiniteDevice(storage.mean_loss.get(), 1, "training mean loss",
                          &storage, stream.get());
      RequireFiniteDevice(storage.logits_gradient.get(),
                          static_cast<std::size_t>(actual) * kClassCount,
                          "training logits gradient", &storage, stream.get());
      float batch_loss = 0.0F;
      CUDA_CHECK(cudaMemcpyAsync(&batch_loss, storage.mean_loss.get(),
                                 sizeof(batch_loss), cudaMemcpyDeviceToHost,
                                 stream.get()));
      CUDA_CHECK(cudaStreamSynchronize(stream.get()));
      model.Backward(storage.logits_gradient.get(), static_cast<int>(actual));
      model.RequireFinite("training backward");
      ++global_step;
      model.AdamWStep(global_step, LearningRateForEpoch(epoch), optimizer);
      model.RequireFinite("training optimizer step");
      weighted_loss += static_cast<double>(batch_loss) * actual;
      trained_samples += actual;
    }
    if (trained_samples != split.training_indices.size()) {
      throw std::runtime_error("training batches did not cover every sample");
    }
    const float epoch_loss =
        static_cast<float>(weighted_loss / static_cast<double>(trained_samples));
    if (!std::isfinite(epoch_loss)) {
      throw std::runtime_error("sample-weighted epoch loss is not finite");
    }
    const EvaluationResult validation = EvaluateBatches(
        training_dataset, split.validation_indices, options.batch_size, false,
        &model, &storage, stream.get());
    const float validation_accuracy =
        static_cast<float>(validation.correct) /
        static_cast<float>(split.validation_indices.size());
    if (!std::isfinite(validation_accuracy)) {
      throw std::runtime_error("validation accuracy is not finite");
    }
    if (validation_accuracy > best_accuracy) {
      const Checkpoint checkpoint{{epoch, validation_accuracy,
                                   kNormalizationMean,
                                   kNormalizationStddev},
                                  model.ExportParameters()};
      SaveCheckpoint(options.output_path, checkpoint);
      best_accuracy = validation_accuracy;
    }
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - epoch_start)
            .count();
    PrintEpochSummary(output, epoch, epoch_loss, validation_accuracy,
                      elapsed_ms);
  }

  const Checkpoint best = LoadCheckpoint(options.output_path);
  model.ImportParameters(best.parameters);
  model.RequireFinite("reloaded best checkpoint");
  const EvaluationResult test = EvaluateBatches(
      test_dataset, test_order, options.batch_size, false, &model, &storage,
      stream.get());
  const float test_accuracy = static_cast<float>(test.correct) /
                              static_cast<float>(test_dataset.sample_count);
  PrintFinalTestSummary(output, test_dataset.sample_count, test_accuracy,
                        best.metadata);
  return kSuccess;
}

int RunEvaluate(const EvaluateOptions& options, std::ostream& output,
                std::ostream& error) {
  static_cast<void>(error);
  const MnistDataset dataset = LoadMnistDataset(options.data_path);
  RequireDatasetCount(dataset, 10000, options.allow_nonstandard_count,
                      "evaluation dataset");
  const Checkpoint checkpoint = LoadCheckpoint(options.weights_path);
  if (!std::isfinite(options.minimum_accuracy) ||
      options.minimum_accuracy < 0.0F || options.minimum_accuracy > 1.0F) {
    throw std::invalid_argument("minimum accuracy must be finite and in [0, 1]");
  }
  const std::vector<std::uint32_t> order =
      CanonicalOrder(dataset.sample_count);

  SelectAndReportDevice(options.device, output);
  WorkflowStream stream;
  WorkflowStorage storage(kEvaluationBatchSize);
  LeNet model(static_cast<int>(kEvaluationBatchSize), 0, stream.get());
  model.ImportParameters(checkpoint.parameters);
  model.RequireFinite("evaluation checkpoint import");
  const EvaluationResult result = EvaluateBatches(
      dataset, order, kEvaluationBatchSize, true, &model, &storage,
      stream.get());
  const float accuracy = static_cast<float>(result.correct) /
                         static_cast<float>(dataset.sample_count);
  const float mean_forward_ms =
      result.timed_batches == 0
          ? 0.0F
          : static_cast<float>(result.total_forward_ms /
                               static_cast<double>(result.timed_batches));
  const float images_per_second =
      result.total_forward_ms > 0.0
          ? static_cast<float>(static_cast<double>(result.timed_images) *
                               1000.0 / result.total_forward_ms)
          : 0.0F;
  if (!std::isfinite(accuracy) || !std::isfinite(mean_forward_ms) ||
      !std::isfinite(images_per_second)) {
    throw std::runtime_error("evaluation metrics are not finite");
  }
  const bool passed = accuracy >= options.minimum_accuracy;
  PrintEvaluationSummary(output, dataset.sample_count, accuracy,
                         mean_forward_ms, images_per_second,
                         options.minimum_accuracy, passed);
  return passed ? kSuccess : kAcceptanceFailure;
}

int RunInfer(const InferOptions& options, std::ostream& output,
             std::ostream& error) {
  static_cast<void>(error);
  const MnistDataset dataset = LoadMnistDataset(options.data_path);
  const Checkpoint checkpoint = LoadCheckpoint(options.weights_path);
  if (options.index >= dataset.sample_count) {
    std::ostringstream message;
    message << "inference index " << options.index
            << " is outside loaded sample count " << dataset.sample_count;
    throw std::out_of_range(message.str());
  }
  const std::vector<std::uint32_t> order{
      static_cast<std::uint32_t>(options.index)};

  SelectAndReportDevice(options.device, output);
  WorkflowStream stream;
  WorkflowStorage storage(1);
  LeNet model(1, 0, stream.get());
  model.ImportParameters(checkpoint.parameters);
  model.RequireFinite("inference checkpoint import");
  PackBatch(dataset, order, 0, 1, &storage.host_batch);
  NormalizeBatch(storage.host_batch, 0, 1, false, &storage, stream.get());
  const float* device_logits = model.Forward(storage.normalized_images.get(), 1);
  model.RequireFinite("inference forward");
  LaunchSoftmax(device_logits, storage.probabilities.get(), 1, kClassCount,
                stream.get());
  RequireFiniteDevice(storage.probabilities.get(), kClassCount,
                      "inference probabilities", &storage, stream.get());
  LaunchArgmaxAndCountCorrect(
      device_logits, storage.device_labels.get(), storage.predictions.get(),
      storage.correct_flags.get(), storage.correct_count.get(), 1, kClassCount,
      stream.get());

  std::array<float, kClassCount> logits;
  std::array<float, kClassCount> probabilities;
  std::uint8_t prediction = 0;
  std::uint8_t label = 0;
  CUDA_CHECK(cudaMemcpyAsync(logits.data(), device_logits,
                             logits.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream.get()));
  CUDA_CHECK(cudaMemcpyAsync(probabilities.data(), storage.probabilities.get(),
                             probabilities.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream.get()));
  CUDA_CHECK(cudaMemcpyAsync(&prediction, storage.predictions.get(),
                             sizeof(prediction), cudaMemcpyDeviceToHost,
                             stream.get()));
  CUDA_CHECK(cudaMemcpyAsync(&label, storage.device_labels.get(), sizeof(label),
                             cudaMemcpyDeviceToHost, stream.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  for (int class_index = 0; class_index < kClassCount; ++class_index) {
    if (!std::isfinite(logits[class_index]) ||
        !std::isfinite(probabilities[class_index])) {
      throw std::runtime_error("inference output contains a non-finite value");
    }
  }

  std::ostringstream record;
  record.imbue(std::locale::classic());
  record << std::fixed << std::setprecision(9)
         << "event=infer index=" << options.index << " logits=";
  for (int class_index = 0; class_index < kClassCount; ++class_index) {
    if (class_index != 0) {
      record << ',';
    }
    record << logits[class_index];
  }
  record << " probabilities=";
  for (int class_index = 0; class_index < kClassCount; ++class_index) {
    if (class_index != 0) {
      record << ',';
    }
    record << probabilities[class_index];
  }
  record << " prediction=" << static_cast<unsigned int>(prediction)
         << " label=" << static_cast<unsigned int>(label) << '\n';
  const std::string text = record.str();
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  return kSuccess;
}
