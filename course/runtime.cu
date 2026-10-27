// Implements the official course inference runtime: host validation first,
// one explicit device-0 setup, a single fixed set of device buffers, all batches
// enqueued without intermediate readback, one final prediction/sticky-flag
// readback, and a single buffered stdout publication.
#include "course/runtime.h"

#include "course/gpu_ops.h"
#include "course/inference.h"
#include "course/model_io.h"
#include "cuda_check.h"
#include "tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kChannels = 1;
constexpr std::uint32_t kHeight = 28;
constexpr std::uint32_t kWidth = 28;
constexpr std::uint32_t kClasses = 10;
constexpr std::uint32_t kImagePixels = kHeight * kWidth;
constexpr std::uint32_t kMinimumBatch = 1;
constexpr std::uint32_t kMaximumBatch = 16384;

using Clock = std::chrono::steady_clock;

// Converts a clock duration to fractional milliseconds.
double Milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

// Multiplies two sizes, rejecting overflow before it can wrap.
std::size_t CheckedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("course inference extent overflows size_t");
  }
  return left * right;
}

// Validates every direct-API option before any host file or CUDA work.
void ValidateOptions(const course::InferOptions& options) {
  if (options.images_path.empty()) {
    throw std::invalid_argument("inference images path must not be empty");
  }
  if (options.model_dir.empty()) {
    throw std::invalid_argument("COURSE_MODEL_DIR must not be empty");
  }
  if (options.shape.n == 0) {
    throw std::invalid_argument("inference N must be positive");
  }
  if (options.shape.c != kChannels || options.shape.h != kHeight ||
      options.shape.w != kWidth || options.shape.k != kClasses) {
    throw std::invalid_argument("only C=1, H=28, W=28, K=10 are supported");
  }
  if (options.batch_size < kMinimumBatch ||
      options.batch_size > kMaximumBatch) {
    throw std::invalid_argument("COURSE_INFER_BATCH must be in [1, 16384]");
  }
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
  throw std::logic_error("unknown course input mode");
}

// Owns the non-blocking stream and destroys it without letting cleanup errors
// escape. The stream remains alive until every buffer/model above it unwinds.
class InferenceStream {
 public:
  InferenceStream() : stream_(nullptr) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  ~InferenceStream() {
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  InferenceStream(const InferenceStream&) = delete;
  InferenceStream& operator=(const InferenceStream&) = delete;

  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_;
};

// Owns the four timing events for every batch. Allocation happens once, before
// the batch loop; the fast path never allocates this object's storage.
class BatchTimingEvents {
 public:
  BatchTimingEvents() = default;

  ~BatchTimingEvents() { Destroy(); }

  BatchTimingEvents(const BatchTimingEvents&) = delete;
  BatchTimingEvents& operator=(const BatchTimingEvents&) = delete;

  // Allocates exactly four default (timing) events per batch. On failure the
  // object stays alive in the caller, so its destructor releases any events
  // created before the failing call.
  void Allocate(std::size_t batch_count) {
    events_.assign(batch_count, BatchEvents{nullptr, nullptr, nullptr, nullptr});
    for (BatchEvents& batch : events_) {
      CUDA_CHECK(cudaEventCreateWithFlags(&batch.upload_start, cudaEventDefault));
      CUDA_CHECK(cudaEventCreateWithFlags(&batch.upload_end, cudaEventDefault));
      CUDA_CHECK(cudaEventCreateWithFlags(&batch.prepared, cudaEventDefault));
      CUDA_CHECK(
          cudaEventCreateWithFlags(&batch.forward_complete, cudaEventDefault));
    }
  }

  struct BatchEvents {
    cudaEvent_t upload_start;
    cudaEvent_t upload_end;
    cudaEvent_t prepared;
    cudaEvent_t forward_complete;
  };

  BatchEvents& at(std::size_t index) { return events_[index]; }

 private:
  void Destroy() noexcept {
    for (BatchEvents& batch : events_) {
      if (batch.upload_start != nullptr) {
        static_cast<void>(cudaEventDestroy(batch.upload_start));
      }
      if (batch.upload_end != nullptr) {
        static_cast<void>(cudaEventDestroy(batch.upload_end));
      }
      if (batch.prepared != nullptr) {
        static_cast<void>(cudaEventDestroy(batch.prepared));
      }
      if (batch.forward_complete != nullptr) {
        static_cast<void>(cudaEventDestroy(batch.forward_complete));
      }
    }
    events_.clear();
  }

  std::vector<BatchEvents> events_;
};

// Requires one device-event duration to be finite and nonnegative.
double EventMilliseconds(cudaEvent_t start, cudaEvent_t end,
                         const char* label) {
  float milliseconds = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
  if (!std::isfinite(milliseconds) || milliseconds < 0.0F) {
    throw std::runtime_error(std::string("invalid GPU timing for ") + label);
  }
  return static_cast<double>(milliseconds);
}

// Appends one fixed-precision profile record using the classic locale.
void AppendProfileRecord(std::ostringstream* output, const char* event,
                         std::size_t index, std::size_t batch_samples,
                         double h2d_ms, double prepare_ms,
                         double forward_ms) {
  *output << "event=" << event << " index=" << index
          << " batch_samples=" << batch_samples << " h2d_ms=" << h2d_ms
          << " prepare_ms=" << prepare_ms
          << " forward_argmax_ms=" << forward_ms << '\n';
}

}  // namespace

namespace course {

int RunCourseInfer(const InferOptions& options, std::ostream& predictions,
                   std::ostream& diagnostics) {
  ValidateOptions(options);

  const std::size_t sample_count = static_cast<std::size_t>(options.shape.n);
  const std::size_t image_element_count =
      CheckedMultiply(sample_count, static_cast<std::size_t>(kImagePixels));
  const std::size_t prediction_bytes =
      CheckedMultiply(sample_count, sizeof(std::int32_t));
  const std::uint32_t batch_size = options.batch_size;
  const std::size_t batch_element_count =
      CheckedMultiply(static_cast<std::size_t>(batch_size),
                      static_cast<std::size_t>(kImagePixels));
  const std::size_t batch_count =
      1 + (sample_count - 1) / static_cast<std::size_t>(batch_size);

  // Host-only model and input validation. No CUDA call occurs before both
  // complete; the persisted metadata mode is the only preprocessing selector.
  double host_load_model_ms = 0.0;
  double host_load_images_ms = 0.0;
  const Clock::time_point load_model_start = Clock::now();
  const CourseModel bundle = LoadCourseModel(options.model_dir);
  host_load_model_ms = Milliseconds(Clock::now() - load_model_start);

  const Clock::time_point load_images_start = Clock::now();
  const std::vector<float> host_images =
      ReadImages(options.images_path, options.shape, bundle.input_mode);
  host_load_images_ms = Milliseconds(Clock::now() - load_images_start);
  if (host_images.size() != image_element_count) {
    throw std::runtime_error(
        "inference image buffer size does not match the declared shape");
  }

  // Host prediction landing storage is allocated before CUDA setup so the
  // final readback bucket contains only copies and synchronization.
  std::vector<std::int32_t> host_predictions(sample_count);
  int host_bad_output = 0;

  double host_cuda_setup_ms = 0.0;
  const Clock::time_point cuda_setup_start = Clock::now();

  // Explicit device 0 selection precedes the first stream/allocation call. No
  // banner is written to either supplied stream.
  CUDA_CHECK(cudaSetDevice(0));
  InferenceStream stream;
  InferenceModel model(static_cast<int>(batch_size),
                       bundle.checkpoint.parameters, stream.get());
  // Input and normalized buffers hold one configured batch capacity and are
  // reused at pointer zero for every batch. Only predictions span all N.
  DeviceBuffer<float> packed(batch_element_count);
  DeviceBuffer<float> normalized(batch_element_count);
  DeviceBuffer<std::int32_t> device_predictions(sample_count);
  DeviceBuffer<int> device_bad_output(1);

  // The sticky flag is zeroed exactly once per inference request. Argmax never
  // clears it, so any early non-finite output remains visible at final readback.
  CUDA_CHECK(cudaMemsetAsync(device_bad_output.get(), 0, sizeof(int),
                             stream.get()));

  BatchTimingEvents timing;
  if (options.profile) {
    timing.Allocate(batch_count);
  }
  host_cuda_setup_ms = Milliseconds(Clock::now() - cuda_setup_start);

  // Enqueue every batch. No device allocation, readback, or synchronization
  // occurs in this loop; the fast path also creates no events.
  double host_enqueue_batches_ms = 0.0;
  const Clock::time_point enqueue_start = Clock::now();
  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const std::size_t offset = batch_index * static_cast<std::size_t>(batch_size);
    const std::size_t remaining = sample_count - offset;
    const std::size_t current_batch = std::min<std::size_t>(
        static_cast<std::size_t>(batch_size), remaining);
    const std::size_t host_offset =
        CheckedMultiply(offset, static_cast<std::size_t>(kImagePixels));
    const int batch = static_cast<int>(current_batch);

    // Record the upload window around the actual H2D slice copy.
    if (options.profile) {
      BatchTimingEvents::BatchEvents& events = timing.at(batch_index);
      CUDA_CHECK(cudaEventRecord(events.upload_start, stream.get()));
    }
    CUDA_CHECK(cudaMemcpyAsync(
        packed.get(), host_images.data() + host_offset,
        CheckedMultiply(current_batch, sizeof(float) * kImagePixels),
        cudaMemcpyHostToDevice, stream.get()));
    if (options.profile) {
      BatchTimingEvents::BatchEvents& events = timing.at(batch_index);
      CUDA_CHECK(cudaEventRecord(events.upload_end, stream.get()));
    }

    LaunchPrepareInput(packed.get(), nullptr, normalized.get(), batch,
                       bundle.input_mode, 0, 0, false, stream.get());
    if (options.profile) {
      BatchTimingEvents::BatchEvents& events = timing.at(batch_index);
      CUDA_CHECK(cudaEventRecord(events.prepared, stream.get()));
    }

    const float* const logits = model.Forward(normalized.get(), batch);
    LaunchArgmax(logits, device_predictions.get() + offset,
                 device_bad_output.get(), batch, static_cast<int>(kClasses),
                 stream.get());
    if (options.profile) {
      BatchTimingEvents::BatchEvents& events = timing.at(batch_index);
      CUDA_CHECK(cudaEventRecord(events.forward_complete, stream.get()));
    }
  }
  host_enqueue_batches_ms = Milliseconds(Clock::now() - enqueue_start);

  // One final readback boundary: both device-to-host copies are enqueued after
  // every batch, followed by the only stream synchronization in the fast path.
  double host_readback_sync_ms = 0.0;
  const Clock::time_point readback_start = Clock::now();
  CUDA_CHECK(cudaMemcpyAsync(host_predictions.data(),
                             device_predictions.get(), prediction_bytes,
                             cudaMemcpyDeviceToHost, stream.get()));
  CUDA_CHECK(cudaMemcpyAsync(&host_bad_output, device_bad_output.get(),
                             sizeof(int), cudaMemcpyDeviceToHost,
                             stream.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  host_readback_sync_ms = Milliseconds(Clock::now() - readback_start);

  if (host_bad_output != 0) {
    throw std::runtime_error(
        "inference produced a non-finite output; no predictions were written");
  }
  for (const std::int32_t prediction : host_predictions) {
    if (prediction < 0 ||
        prediction >= static_cast<std::int32_t>(kClasses)) {
      throw std::runtime_error(
          "inference produced a prediction outside [0, 10)");
    }
  }

  // Profile event durations are computed before the prediction publication, so
  // a timing failure can never occur after stdout has been committed. GPU
  // values are event durations; host values are separate wall buckets and
  // enqueue is never labeled as GPU execution.
  std::vector<double> h2d_per_batch;
  std::vector<double> prepare_per_batch;
  std::vector<double> forward_per_batch;
  double total_h2d_ms = 0.0;
  double total_prepare_ms = 0.0;
  double total_forward_ms = 0.0;
  if (options.profile) {
    h2d_per_batch.resize(batch_count);
    prepare_per_batch.resize(batch_count);
    forward_per_batch.resize(batch_count);
    for (std::size_t batch_index = 0; batch_index < batch_count;
         ++batch_index) {
      BatchTimingEvents::BatchEvents& events = timing.at(batch_index);
      h2d_per_batch[batch_index] = EventMilliseconds(
          events.upload_start, events.upload_end, "H2D");
      prepare_per_batch[batch_index] = EventMilliseconds(
          events.upload_end, events.prepared, "prepare");
      forward_per_batch[batch_index] = EventMilliseconds(
          events.prepared, events.forward_complete, "forward+argmax");
      total_h2d_ms += h2d_per_batch[batch_index];
      total_prepare_ms += prepare_per_batch[batch_index];
      total_forward_ms += forward_per_batch[batch_index];
    }
  }

  // Format every line in a classic-locale buffer and publish it with one write.
  double host_write_predictions_ms = 0.0;
  const Clock::time_point write_start = Clock::now();
  std::ostringstream prediction_record;
  prediction_record.imbue(std::locale::classic());
  for (const std::int32_t prediction : host_predictions) {
    prediction_record << prediction << '\n';
  }
  predictions << prediction_record.str();
  // Flush before any success diagnostics so a buffered sink failure is
  // observable while the function can still return a failure.
  predictions.flush();
  if (!predictions) {
    throw std::runtime_error("could not write inference predictions");
  }
  host_write_predictions_ms = Milliseconds(Clock::now() - write_start);

  // Diagnostics are assembled after successful publication. The property
  // profile records cover every batch, including the first.
  std::ostringstream diagnostic_record;
  diagnostic_record.imbue(std::locale::classic());
  if (options.profile) {
    std::ostringstream profile_record;
    profile_record.imbue(std::locale::classic());
    profile_record << std::fixed << std::setprecision(3);
    for (std::size_t batch_index = 0; batch_index < batch_count;
         ++batch_index) {
      const std::size_t offset =
          batch_index * static_cast<std::size_t>(batch_size);
      const std::size_t remaining = sample_count - offset;
      const std::size_t current_batch = std::min<std::size_t>(
          static_cast<std::size_t>(batch_size), remaining);
      AppendProfileRecord(&profile_record, "course_profile_batch",
                          batch_index + 1, current_batch,
                          h2d_per_batch[batch_index],
                          prepare_per_batch[batch_index],
                          forward_per_batch[batch_index]);
    }

    profile_record << "event=course_profile profile_batches=" << batch_count
                   << std::fixed << std::setprecision(3)
                   << " gpu_h2d_ms=" << total_h2d_ms
                   << " gpu_prepare_ms=" << total_prepare_ms
                   << " gpu_forward_argmax_ms=" << total_forward_ms
                   << " host_load_model_ms=" << host_load_model_ms
                   << " host_load_images_ms=" << host_load_images_ms
                   << " host_cuda_setup_ms=" << host_cuda_setup_ms
                   << " host_enqueue_batches_ms=" << host_enqueue_batches_ms
                   << " host_readback_sync_ms=" << host_readback_sync_ms
                   << " host_write_predictions_ms=" << host_write_predictions_ms
                   << '\n';
    diagnostic_record << profile_record.str();
  }

  diagnostic_record << "event=course_infer status=pass samples=" << sample_count
                    << " batches=" << batch_count << " input_mode="
                    << InputModeName(bundle.input_mode) << '\n';
  diagnostics << diagnostic_record.str();
  // The success return is valid only after the diagnostics sink has accepted
  // and flushed the record. Predictions may already have been published above,
  // so this failure is not a rollback guarantee.
  diagnostics.flush();
  if (!diagnostics) {
    throw std::runtime_error("could not write inference diagnostics");
  }
  return 0;
}

}  // namespace course
