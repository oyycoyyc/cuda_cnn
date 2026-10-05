#include "reporting.h"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Reporters emit stable newline-terminated key=value records through a private
// scratch stream so numeric formatting and locale never leak to callers.
namespace {

// Requires a finite, nonnegative measurement, naming it in the error.
void RequireNonnegativeFinite(double value, const char* name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and nonnegative");
  }
}

// Requires a finite unit-interval accuracy value, naming it in the error.
void RequireAccuracy(float value, const char* name) {
  if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and in [0, 1]");
  }
}

// Writes one formatted record without touching the caller's stream state.
void WriteRecord(std::ostream& output, const std::ostringstream& formatted) {
  const std::string record = formatted.str();
  output.write(record.data(), static_cast<std::streamsize>(record.size()));
}

}  // namespace

// Emits one device record; index and capability fields must be nonnegative and
// name must be a nonempty single line.
void PrintDeviceSummary(std::ostream& output, int index, const std::string& name,
                        int compute_major, int compute_minor) {
  if (index < 0 || compute_major < 0 || compute_minor < 0) {
    throw std::invalid_argument("device fields must be nonnegative");
  }
  if (name.empty() || name.find('\n') != std::string::npos ||
      name.find('\r') != std::string::npos) {
    throw std::invalid_argument("device name must be a nonempty single line");
  }

  std::ostringstream formatted;
  // Format on a scratch stream with the classic locale so the caller's locale
  // and formatting flags are preserved.
  formatted.imbue(std::locale::classic());
  formatted << "event=device index=" << index << " name=" << name
            << " compute_capability=" << compute_major << '.' << compute_minor
            << '\n';
  WriteRecord(output, formatted);
}

// Emits one epoch record with six fractional digits for loss/accuracy and
// three for elapsed milliseconds.
void PrintEpochSummary(std::ostream& output, std::uint32_t epoch,
                       float training_loss, float validation_accuracy,
                       double elapsed_ms) {
  if (epoch == 0) {
    throw std::invalid_argument("epoch must be nonzero");
  }
  RequireNonnegativeFinite(training_loss, "training_loss");
  RequireAccuracy(validation_accuracy, "validation_accuracy");
  RequireNonnegativeFinite(elapsed_ms, "elapsed_ms");

  std::ostringstream formatted;
  formatted.imbue(std::locale::classic());
  formatted << std::fixed << "event=epoch epoch=" << epoch
            << " train_loss=" << std::setprecision(6) << training_loss
            << " validation_accuracy=" << validation_accuracy
            << " elapsed_ms=" << std::setprecision(3) << elapsed_ms << '\n';
  WriteRecord(output, formatted);
}

// Emits one evaluation record; accuracy fields use six digits and timing/rate
// fields three.
void PrintEvaluationSummary(std::ostream& output, std::uint32_t samples,
                            float accuracy, float mean_forward_ms,
                            float images_per_second, float minimum_accuracy,
                            bool passed) {
  if (samples == 0) {
    throw std::invalid_argument("samples must be nonzero");
  }
  RequireAccuracy(accuracy, "accuracy");
  RequireNonnegativeFinite(mean_forward_ms, "mean_forward_ms");
  RequireNonnegativeFinite(images_per_second, "images_per_second");
  RequireAccuracy(minimum_accuracy, "minimum_accuracy");

  std::ostringstream formatted;
  formatted.imbue(std::locale::classic());
  formatted << std::fixed << "event=evaluate samples=" << samples
            << " accuracy=" << std::setprecision(6) << accuracy
            << " mean_forward_ms=" << std::setprecision(3) << mean_forward_ms
            << " images_per_second=" << images_per_second
            << " min_accuracy=" << std::setprecision(6) << minimum_accuracy
            << " status=" << (passed ? "pass" : "fail") << '\n';
  WriteRecord(output, formatted);
}

// Emits one final-test record with six fractional digits for both accuracies.
void PrintFinalTestSummary(std::ostream& output, std::uint32_t samples,
                           float accuracy, std::uint32_t best_epoch,
                           float validation_accuracy) {
  if (samples == 0 || best_epoch == 0) {
    throw std::invalid_argument("samples and best_epoch must be nonzero");
  }
  RequireAccuracy(accuracy, "accuracy");
  RequireAccuracy(validation_accuracy, "validation_accuracy");

  std::ostringstream formatted;
  formatted.imbue(std::locale::classic());
  formatted << std::fixed << "event=final_test samples=" << samples
            << " final_test_accuracy=" << std::setprecision(6) << accuracy
            << " best_epoch=" << best_epoch
            << " validation_accuracy=" << validation_accuracy << '\n';
  WriteRecord(output, formatted);
}

// Emits one inference record: ten logits then ten probabilities at nine
// fractional digits, followed by prediction and label classes.
void PrintInferenceSummary(std::ostream& output, std::uint32_t index,
                           const std::array<float, 10>& logits,
                           const std::array<float, 10>& probabilities,
                           std::uint8_t prediction, std::uint8_t label) {
  if (prediction >= 10 || label >= 10) {
    throw std::invalid_argument("prediction and label must be in [0, 9]");
  }
  for (std::size_t class_index = 0; class_index < logits.size(); ++class_index) {
    if (!std::isfinite(logits[class_index]) ||
        !std::isfinite(probabilities[class_index])) {
      throw std::invalid_argument("inference values must be finite");
    }
  }

  std::ostringstream formatted;
  formatted.imbue(std::locale::classic());
  formatted << std::fixed << std::setprecision(9)
            << "event=infer index=" << index << " logits=";
  for (std::size_t class_index = 0; class_index < logits.size(); ++class_index) {
    if (class_index != 0) {
      formatted << ',';
    }
    formatted << logits[class_index];
  }
  formatted << " probabilities=";
  for (std::size_t class_index = 0; class_index < probabilities.size();
       ++class_index) {
    if (class_index != 0) {
      formatted << ',';
    }
    formatted << probabilities[class_index];
  }
  formatted << " prediction=" << static_cast<unsigned int>(prediction)
            << " label=" << static_cast<unsigned int>(label) << '\n';
  WriteRecord(output, formatted);
}
