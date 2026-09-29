#ifndef INCLUDE_TRAIN_H_
#define INCLUDE_TRAIN_H_

#include "cli.h"

#include <cstdint>
#include <iosfwd>

// Stable process exit statuses shared by the workflow API and CLI executable.
// Success is zero; runtime failures, command usage failures, and unmet
// evaluation acceptance thresholds have distinct nonzero values.
enum ExitCode {
  kSuccess = 0,
  kRuntimeError = 1,
  kUsageError = 2,
  kAcceptanceFailure = 3
};

// Loads and validates both datasets and the output parent before selecting the
// requested CUDA device. It trains modern LeNet using the owned options,
// validates after every epoch, atomically saves strict accuracy improvements,
// reloads the persisted best checkpoint, and evaluates the untouched test set.
// output receives stable device/epoch/final records; error is caller-owned and
// reserved for workflow diagnostics. Paths and options are not retained. CUDA,
// data, checkpoint, non-finite, or I/O failures throw std::runtime_error;
// invalid direct-call options may throw std::invalid_argument. No CPU fallback
// is attempted, and all device storage is allocated before batch processing.
int RunTrain(const TrainOptions& options, std::ostream& output,
             std::ostream& error);

// Loads one validated test dataset and checkpoint, selects the requested CUDA
// device, and evaluates fixed-order batches without augmentation. The first
// forward batch is an unmeasured warm-up; CUDA Events time only later forward
// launches. output receives stable device/evaluation records and error is
// caller-owned and reserved for diagnostics. Paths are not retained. Runtime
// validation/CUDA failures throw; accuracy below minimum_accuracy returns
// kAcceptanceFailure. Device storage is fixed before the first batch.
int RunEvaluate(const EvaluateOptions& options, std::ostream& output,
                std::ostream& error);

// Loads one validated dataset/checkpoint, checks index against the loaded row
// count, selects the requested CUDA device, and runs an unaugmented batch of
// one. It copies exactly ten logits, ten stable-softmax probabilities, one
// prediction, and one label for a stable machine-readable output record. error
// is caller-owned and reserved for diagnostics; paths/options are not retained.
// Runtime validation/CUDA/non-finite failures throw and never fall back to CPU.
int RunInfer(const InferOptions& options, std::ostream& output,
             std::ostream& error);

// Returns the fixed FP32 schedule for a one-based training epoch: 1e-3 for
// epochs 1-12, 1e-4 for 13-17, and 1e-5 from epoch 18 onward. Epoch zero throws
// std::invalid_argument. The function owns no storage and performs no CUDA work.
float LearningRateForEpoch(std::uint32_t one_based_epoch);

#endif  // INCLUDE_TRAIN_H_
