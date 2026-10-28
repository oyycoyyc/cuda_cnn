// Development-only warm full-request benchmark tool.
//
// One inference request is the entire supplied image set. The tool loads and
// validates the model and images once, preallocates all buffers once, performs
// an untimed setup synchronization, then repeats the complete request: reset
// the sticky flag, copy every image again in configured batch slices, prepare
// on the GPU, run the original forward chain and global-offset argmax, read all
// predictions and the flag back, synchronize, and validate the host results.
//
// The host steady clock brackets the flag reset through the post-sync range and
// prediction-invariance checks. Record formatting, stream writes, and stdout
// publication happen outside that window. No device allocation, timing event,
// disk access, or host output formatting occurs inside a round.
//
// Usage: course_warm_benchmark images.f32 N C H W K warmups runs
// The first seven arguments are parsed by course::ParseInferArgs with the
// existing COURSE_MODEL_DIR and COURSE_INFER_BATCH environment; the persisted
// model input mode is authoritative. Training and profile variables are never
// consulted. Usage errors exit 2; model, input, CUDA, and output failures exit 1.

#include "course/gpu_ops.h"
#include "course/inference.h"
#include "course/interface.h"
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
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kHeight = 28;
constexpr std::uint32_t kWidth = 28;
constexpr std::uint32_t kClasses = 10;
constexpr std::uint32_t kImagePixels = kHeight * kWidth;
constexpr std::uint32_t kMinimumRoundCount = 1;
constexpr std::uint32_t kMaximumRoundCount = 100000;

using Clock = std::chrono::steady_clock;

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

// Parses a positive ASCII decimal round count in [1, 100000], rejecting signs,
// spaces, fractions, missing values, and overflow before narrowing.
std::uint32_t ParseRoundCount(const char* text, const char* field) {
  if (text == nullptr || *text == '\0') {
    throw std::invalid_argument(std::string(field) +
                                " must be a positive decimal integer in "
                                "[1, 100000]");
  }
  std::uint64_t value = 0;
  for (const char* digit = text; *digit != '\0'; ++digit) {
    if (*digit < '0' || *digit > '9') {
      throw std::invalid_argument(std::string(field) +
                                  " must contain only decimal digits");
    }
    value = value * 10 + static_cast<std::uint64_t>(*digit - '0');
    if (value > kMaximumRoundCount) {
      throw std::invalid_argument(std::string(field) +
                                  " must be in [1, 100000]");
    }
  }
  if (value < kMinimumRoundCount) {
    throw std::invalid_argument(std::string(field) +
                                " must be a positive decimal integer in "
                                "[1, 100000]");
  }
  return static_cast<std::uint32_t>(value);
}

// Multiplies two sizes, rejecting overflow before it can wrap.
std::size_t CheckedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("warm benchmark extent overflows size_t");
  }
  return left * right;
}

// Converts a clock duration to fractional milliseconds.
double Milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

// Owns the single non-blocking stream. Destruction never lets a cleanup error
// escape, and the stream outlives every buffer and the model above it.
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

int RunWarmBenchmark(int argc, char** argv, std::ostream& predictions,
                     std::ostream& diagnostics) {
  if (argc != 9) {
    throw std::invalid_argument(
        "usage: course_warm_benchmark images.f32 N C H W K warmups runs");
  }

  // Exactly the first seven arguments are the official inference request. The
  // profile input is deliberately null, so COURSE_PROFILE cannot enable profile
  // records or alter the request.
  const char* const model_dir = std::getenv("COURSE_MODEL_DIR");
  const char* const batch = std::getenv("COURSE_INFER_BATCH");
  const course::InferOptions options = course::ParseInferArgs(
      7, const_cast<const char* const*>(argv), model_dir, batch, nullptr);

  const std::uint32_t warmups = ParseRoundCount(argv[7], "warmups");
  const std::uint32_t runs = ParseRoundCount(argv[8], "runs");

  // Complete host validation, including every input value and the persisted
  // model mode, happens before the first CUDA call or device selection.
  const course::CourseModel bundle = course::LoadCourseModel(options.model_dir);
  const std::vector<float> host_images = course::ReadImages(
      options.images_path, options.shape, bundle.input_mode);
  const std::size_t sample_count = static_cast<std::size_t>(options.shape.n);
  const std::size_t image_element_count =
      CheckedMultiply(sample_count, kImagePixels);
  if (host_images.size() != image_element_count) {
    throw std::runtime_error(
        "warm benchmark image buffer size does not match the declared shape");
  }

  const std::uint32_t batch_size = options.batch_size;
  const std::size_t batch_element_count =
      CheckedMultiply(static_cast<std::size_t>(batch_size), kImagePixels);
  const std::size_t prediction_bytes =
      CheckedMultiply(sample_count, sizeof(std::int32_t));
  const std::size_t batch_count =
      1 + (sample_count - 1) / static_cast<std::size_t>(batch_size);

  // All host landing storage and the invariant baseline are allocated before
  // device setup. The only host prediction work inside a round is the
  // preallocated baseline copy on the first warmup and exact comparisons later.
  std::vector<std::int32_t> host_predictions(sample_count);
  std::vector<std::int32_t> baseline_predictions(sample_count);
  int host_bad_output = 0;

  // One explicit device, one stream, one model, and one fixed buffer set. The
  // model constructor uploads parameters and synchronizes before this returns.
  CUDA_CHECK(cudaSetDevice(0));
  InferenceStream stream;
  course::InferenceModel model(static_cast<int>(batch_size),
                               bundle.checkpoint.parameters, stream.get());
  DeviceBuffer<float> packed(batch_element_count);
  DeviceBuffer<float> normalized(batch_element_count);
  DeviceBuffer<std::int32_t> device_predictions(sample_count);
  DeviceBuffer<int> device_bad_output(1);

  // Drain any setup work before the first timed round. No warmup can inherit a
  // previous queue, and every later round ends with its own synchronization.
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  const std::uint32_t total_rounds = warmups + runs;
  for (std::uint32_t round_index = 0; round_index < total_rounds;
       ++round_index) {
    const bool is_warmup = round_index < warmups;
    const char* const phase = is_warmup ? "warmup" : "measure";
    const std::uint32_t iteration =
        is_warmup ? round_index + 1 : round_index - warmups + 1;

    // The timed window starts before the per-round sticky-flag reset.
    const Clock::time_point start = Clock::now();
    CUDA_CHECK(cudaMemsetAsync(device_bad_output.get(), 0, sizeof(int),
                               stream.get()));

    for (std::size_t batch_index = 0; batch_index < batch_count;
         ++batch_index) {
      const std::size_t offset =
          batch_index * static_cast<std::size_t>(batch_size);
      const std::size_t remaining = sample_count - offset;
      const std::size_t current_batch = std::min<std::size_t>(
          static_cast<std::size_t>(batch_size), remaining);
      const std::size_t host_offset =
          CheckedMultiply(offset, kImagePixels);
      const int batch_samples = static_cast<int>(current_batch);

      // Every round transfers the actual host-resident input again. Each slice
      // lands at device offset zero; the global sample offset is host_offset.
      CUDA_CHECK(cudaMemcpyAsync(
          packed.get(), host_images.data() + host_offset,
          CheckedMultiply(current_batch, sizeof(float) * kImagePixels),
          cudaMemcpyHostToDevice, stream.get()));
      course::LaunchPrepareInput(packed.get(), nullptr, normalized.get(),
                                 batch_samples, bundle.input_mode, 0, 0, false,
                                 stream.get());
      const float* const logits = model.Forward(normalized.get(), batch_samples);
      course::LaunchArgmax(logits, device_predictions.get() + offset,
                           device_bad_output.get(), batch_samples,
                           static_cast<int>(kClasses), stream.get());
    }

    // One complete readback and synchronization boundary per round.
    CUDA_CHECK(cudaMemcpyAsync(host_predictions.data(),
                               device_predictions.get(), prediction_bytes,
                               cudaMemcpyDeviceToHost, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(&host_bad_output, device_bad_output.get(),
                               sizeof(int), cudaMemcpyDeviceToHost,
                               stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    // Host validation remains inside the timed window. Failures are reported
    // only after the clock stops, so a failed round is never timed as success.
    const char* failure = nullptr;
    if (host_bad_output != 0) {
      failure =
          "warm benchmark produced a non-finite output; no predictions were "
          "written";
    } else {
      for (const std::int32_t prediction : host_predictions) {
        if (prediction < 0 ||
            prediction >= static_cast<std::int32_t>(kClasses)) {
          failure =
              "warm benchmark produced a prediction outside [0, 10); no "
              "predictions were written";
          break;
        }
      }
    }

    bool predictions_identical = true;
    if (failure == nullptr) {
      if (round_index == 0) {
        std::copy(host_predictions.begin(), host_predictions.end(),
                  baseline_predictions.begin());
      } else {
        predictions_identical =
            std::equal(host_predictions.begin(), host_predictions.end(),
                       baseline_predictions.begin());
        if (!predictions_identical) {
          failure =
              "warm benchmark predictions changed from the first warmup round";
        }
      }
    }

    const Clock::time_point end = Clock::now();
    const double wall_ms = Milliseconds(end - start);
    if (failure != nullptr) {
      throw std::runtime_error(failure);
    }
    if (!std::isfinite(wall_ms) || !(wall_ms > 0.0)) {
      throw std::runtime_error("warm benchmark measured a non-positive wall time");
    }

    // Formatting and stream I/O are outside the clock. Every warmup record is
    // retained before the first measured record and is flushed immediately.
    std::ostringstream record;
    record.imbue(std::locale::classic());
    record << "event=course_warm_iteration phase=" << phase
           << " iteration=" << iteration << " samples=" << sample_count
           << " wall_ms=" << std::setprecision(17) << wall_ms
           << " predictions_identical=" << (predictions_identical ? 1 : 0)
           << '\n';
    diagnostics << record.str();
    diagnostics.flush();
    if (!diagnostics) {
      throw std::runtime_error("could not write warm benchmark diagnostics");
    }
  }

  // Predictions remain buffered until every warmup and measured round passed.
  // Publish and flush them before the completion record so a stdout sink
  // failure can never leave a passing completion record behind.
  std::ostringstream prediction_record;
  prediction_record.imbue(std::locale::classic());
  for (const std::int32_t prediction : host_predictions) {
    prediction_record << prediction << '\n';
  }
  predictions << prediction_record.str();
  predictions.flush();
  if (!predictions) {
    throw std::runtime_error("could not write warm benchmark predictions");
  }

  std::ostringstream completion;
  completion.imbue(std::locale::classic());
  completion << "event=course_warm_complete status=pass warmups=" << warmups
             << " runs=" << runs << " samples=" << sample_count
             << " batch=" << batch_size
             << " input_mode=" << InputModeName(bundle.input_mode) << '\n';
  diagnostics << completion.str();
  diagnostics.flush();
  if (!diagnostics) {
    throw std::runtime_error("could not write warm benchmark completion");
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::cout.imbue(std::locale::classic());
  std::cerr.imbue(std::locale::classic());
  try {
    return RunWarmBenchmark(argc, argv, std::cout, std::cerr);
  } catch (const std::invalid_argument& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "error: unknown failure\n";
    return 1;
  }
}
