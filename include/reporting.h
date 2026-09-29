#ifndef INCLUDE_REPORTING_H_
#define INCLUDE_REPORTING_H_

#include <cstdint>
#include <iosfwd>

// Writes one stable newline-terminated epoch key=value record using the
// classic locale, six fractional digits for loss/accuracy, and three for
// elapsed milliseconds. epoch must be nonzero; training_loss and elapsed_ms
// must be finite and nonnegative; validation_accuracy must be finite and in
// [0, 1]. Throws std::invalid_argument for invalid values. The caller owns the
// stream; its locale and formatting state are restored before return.
void PrintEpochSummary(std::ostream& output, std::uint32_t epoch,
                       float training_loss, float validation_accuracy,
                       double elapsed_ms);

// Writes one stable newline-terminated evaluation key=value record using the
// classic locale, six fractional digits for accuracy thresholds, and three
// for timing/rate values. samples must be nonzero; accuracy and
// minimum_accuracy must be finite and in [0, 1]; timing and rate values must
// be finite and nonnegative. passed selects the reported pass/fail status.
// Throws std::invalid_argument for invalid values. The caller owns the stream;
// its locale and formatting state are restored before return.
void PrintEvaluationSummary(std::ostream& output, std::uint32_t samples,
                            float accuracy, float mean_forward_ms,
                            float images_per_second, float minimum_accuracy,
                            bool passed);

#endif  // INCLUDE_REPORTING_H_
