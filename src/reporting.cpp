#include "reporting.h"

#include <cmath>
#include <iomanip>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void RequireNonnegativeFinite(double value, const char* name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and nonnegative");
  }
}

void RequireAccuracy(float value, const char* name) {
  if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and in [0, 1]");
  }
}

void WriteRecord(std::ostream& output, const std::ostringstream& formatted) {
  const std::string record = formatted.str();
  output.write(record.data(), static_cast<std::streamsize>(record.size()));
}

}  // namespace

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
