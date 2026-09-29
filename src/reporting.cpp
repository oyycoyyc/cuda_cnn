#include "reporting.h"

#include <cmath>
#include <iomanip>
#include <ios>
#include <locale>
#include <ostream>
#include <stdexcept>
#include <string>

namespace {

class StreamState {
 public:
  explicit StreamState(std::ostream& stream)
      : stream_(stream),
        flags_(stream.flags()),
        precision_(stream.precision()),
        width_(stream.width()),
        fill_(stream.fill()),
        locale_(stream.getloc()) {}

  ~StreamState() {
    stream_.imbue(locale_);
    stream_.flags(flags_);
    stream_.precision(precision_);
    stream_.width(width_);
    stream_.fill(fill_);
  }

 private:
  std::ostream& stream_;
  std::ios::fmtflags flags_;
  std::streamsize precision_;
  std::streamsize width_;
  char fill_;
  std::locale locale_;
};

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

void Prepare(std::ostream& output) {
  output.imbue(std::locale::classic());
  output.setf(std::ios::fixed, std::ios::floatfield);
  output.width(0);
  output.fill(' ');
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

  StreamState restore(output);
  Prepare(output);
  output << "event=epoch epoch=" << epoch << " train_loss="
         << std::setprecision(6) << training_loss
         << " validation_accuracy=" << validation_accuracy << " elapsed_ms="
         << std::setprecision(3) << elapsed_ms << '\n';
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

  StreamState restore(output);
  Prepare(output);
  output << "event=evaluate samples=" << samples << " accuracy="
         << std::setprecision(6) << accuracy << " mean_forward_ms="
         << std::setprecision(3) << mean_forward_ms << " images_per_second="
         << images_per_second << " min_accuracy=" << std::setprecision(6)
         << minimum_accuracy << " status=" << (passed ? "pass" : "fail")
         << '\n';
}
