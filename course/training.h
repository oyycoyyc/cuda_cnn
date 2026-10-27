#ifndef COURSE_TRAINING_H_
#define COURSE_TRAINING_H_

#include "course/interface.h"
#include "lenet.h"

#include <cstdint>
#include <ostream>

// Course training over the caller-supplied raw FP32 dataset. The workflow uses
// the original LeNet arithmetic, loss/metrics, optimizer launcher, and the
// original deterministic split/shuffle helpers, but it never loads an external
// test dataset and never calls the original RunTrain workflow. The small pure
// policies below are production inputs to that workflow and are declared here
// so their pinned schedule, optimizer configuration, split size, and
// best-epoch rule can be tested without a twenty-epoch fixture run.

namespace course {

// Returns the deterministic training-only validation partition size. The
// official 60000-sample course dataset keeps 5000 samples for validation;
// every other supported training count keeps max(1, sample_count / 5).
// sample_count must be at least two, because training requires at least one
// sample in each partition. A smaller count throws std::invalid_argument.
std::uint32_t CourseValidationCount(std::uint32_t sample_count);

// Returns the pinned decoupled AdamW configuration used by course training:
// beta1 0.9F, beta2 0.999F, epsilon 1.0e-8F, and weight decay 1.0e-4F. The
// original LeNet optimizer applies zero decay to all five bias tensors.
AdamWConfig CourseAdamWConfig();

// Returns the pinned piecewise-constant learning rate for a one-based epoch:
// 1.0e-3F for epochs 1 through 12, 1.0e-4F for epochs 13 through 17, and
// 1.0e-5F from epoch 18 onward. Epoch zero is invalid and throws
// std::invalid_argument.
float CourseLearningRate(std::uint32_t one_based_epoch);

// Tracks the strictly improving validation accuracy used to choose the saved
// checkpoint. The initial state has best_epoch() == 0 and best_accuracy() ==
// -1.0F, so the first valid observation is always selected.
class BestValidationSelection {
 public:
  BestValidationSelection() noexcept : best_epoch_(0), best_accuracy_(-1.0F) {}

  // Validates one_based_epoch >= 1 and a finite accuracy in [0, 1], then
  // records the observation only when it is strictly greater than the current
  // best. Returns true exactly when the best was replaced. Equal or lower
  // accuracies are ignored. Invalid arguments throw std::invalid_argument
  // without changing state.
  bool Consider(std::uint32_t one_based_epoch, float validation_accuracy);

  // Returns the one-based epoch of the current best, or zero before any
  // selection. It performs no validation and cannot fail.
  std::uint32_t best_epoch() const noexcept { return best_epoch_; }

  // Returns the current best validation accuracy, or -1.0F before any
  // selection. It performs no validation and cannot fail.
  float best_accuracy() const noexcept { return best_accuracy_; }

 private:
  std::uint32_t best_epoch_;
  float best_accuracy_;
};

// Trains LeNet on exactly the raw FP32 images and int32 labels named by options
// and publishes a validated best model bundle in options.work_dir. Returns 0
// only after the selected bundle has been saved and successfully reloaded and
// validated; all failures throw before that return.
//
// options must name non-empty images, labels, and work paths; shape must be
// N >= 2, C == 1, H == 28, W == 28, K == 10; epochs must be in [1, 1000];
// batch_size must be positive and fit the existing LeNet int capacity; and
// input_mode must be a known enumerator. There is no fixed upper batch limit
// beyond that representability check, so a large valid direct-API batch may
// fail normally on device memory. Violations throw std::invalid_argument. The
// raw files are read once through
// ReadImages(options.images_path, shape, input_mode) and
// ReadLabels(options.labels_path, shape), so no value is quantized through
// uint8 and every label is already validated into [0, 10).
//
// The selected directory is created when missing, and neither an existing
// directory nor its unrelated entries are cleared. The reserved lenet.bin and
// course.meta pair is published through SaveCourseModel; an incomplete or
// malformed existing reserved pair is refused rather than overwritten.
//
// The deterministic training-only split is
// MakeTrainValidationSplit(N, CourseValidationCount(N), options.seed). Each
// epoch shuffles the training indices with ShuffledTrainingIndices and the
// supplied seed, prepares each batch on the GPU with the persisted input mode
// and deterministic translation augmentation, computes the weighted loss, and
// performs one monotonic global AdamW step per batch with CourseLearningRate
// and CourseAdamWConfig. Validation uses only split.validation_indices, in
// split order and without augmentation, between completed optimizer steps.
// The best bundle is replaced only by strictly improving validation accuracy
// as decided by BestValidationSelection; this is never a test-set choice.
//
// Every diagnostic, including epoch progress, goes to diagnostics. No test
// accuracy is claimed. On success the final record is exactly one line of the
// form:
//   event=course_train status=pass epochs=E best_epoch=B
//   validation_accuracy=A input_mode=M checkpoint_crc32=H
// (space-separated on one line, split here only for readability), where M is
// the persisted "unit", "byte", or "normalized" spelling and H is the eight
// lowercase hexadecimal CRC32 of the saved lenet.bin bytes, identical to the
// validated course.meta value. File, data, CUDA, and bundle failures throw
// std::runtime_error; usage/shape violations throw std::invalid_argument.
int RunCourseTrain(const TrainOptions& options, std::ostream& diagnostics);

}  // namespace course

#endif  // COURSE_TRAINING_H_
