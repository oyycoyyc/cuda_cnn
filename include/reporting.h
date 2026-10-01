#ifndef INCLUDE_REPORTING_H_
#define INCLUDE_REPORTING_H_

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string>

// Writes one stable newline-terminated CUDA device record. The device index
// and compute-capability components must be nonnegative, and name must be a
// nonempty single-line value. Throws std::invalid_argument otherwise.
void PrintDeviceSummary(std::ostream& output, int index, const std::string& name,
                        int compute_major, int compute_minor);

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

// Writes one stable newline-terminated final-test record with six fractional
// digits for both accuracies. Counts and best_epoch must be nonzero and both
// accuracies must be finite and in [0, 1].
void PrintFinalTestSummary(std::ostream& output, std::uint32_t samples,
                           float accuracy, std::uint32_t best_epoch,
                           float validation_accuracy);

// Writes one stable newline-terminated inference record with ten logits and
// probabilities at nine fractional digits. Values must be finite and the
// prediction and label must be decimal classes in [0, 9].
void PrintInferenceSummary(std::ostream& output, std::uint32_t index,
                           const std::array<float, 10>& logits,
                           const std::array<float, 10>& probabilities,
                           std::uint8_t prediction, std::uint8_t label);

#endif  // INCLUDE_REPORTING_H_
