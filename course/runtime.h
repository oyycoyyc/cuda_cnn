#ifndef COURSE_RUNTIME_H_
#define COURSE_RUNTIME_H_

#include "course/interface.h"

#include <ostream>

// Course inference runtime behind the official `infer` executable. The request
// is the already-parsed InferOptions value: positional paths and shapes plus the
// optional COURSE_MODEL_DIR, COURSE_INFER_BATCH, and COURSE_PROFILE inputs that
// ParseInferArgs validated. Training-only variables are never consulted here;
// the persisted course.meta input mode is the only preprocessing selection.
//
// The complete request is validated and buffered before any prediction byte is
// published, so a successful call writes all shape.n class integers together.
// Argument, model, input, CUDA, and non-finite-output failures occur before
// that publication and leave the predictions stream untouched. A downstream
// output/diagnostics sink I/O error is reported but cannot roll back bytes the
// sink already accepted.

namespace course {

// Runs one complete FP32 LeNet inference request on CUDA device 0 and writes
// the class predictions for the supplied images.
//
// Required inputs:
// - options.images_path and options.model_dir are non-empty.
// - options.shape is N >= 1, C == 1, H == 28, W == 28, K == 10.
// - options.batch_size is in [1, 16384].
// - options.model_dir holds a valid lenet.bin/course.meta bundle. The model
//   metadata selects unit, byte, or normalized preprocessing and is used
//   exactly as persisted; COURSE_INPUT_MODE is not an inference input.
// - options.images_path holds exactly N*1*28*28 little-endian FP32 values that
//   satisfy the persisted mode's bounds. No labels file is read or required.
//
// Failure taxonomy for direct callers and for the CLI's exit mapping:
// - std::invalid_argument for a usage/shape/capacity violation: an empty path
//   or model directory, unsupported geometry, an out-of-range batch size, or
//   otherwise invalid options.
// - std::runtime_error for an unreadable or malformed model bundle, bad model
//   CRC, unreadable or out-of-range image data, CUDA setup/launch/copy failure,
//   a non-finite output detected by the GPU sticky flag, or a diagnostics/
//   prediction stream failure.
//
// The model bundle and every host input value are validated before the first
// runtime CUDA allocation, device selection, or launch. The device is selected
// without writing a banner to either supplied stream. All batches are enqueued
// with int32 predictions at their global sample offsets; the predictions and
// the single sticky non-finite flag are read back once after the final stream
// synchronization. Only after validation succeeds are all N lines formatted
// and written in one publication. Each line is exactly one decimal digit in
// [0, 10) followed by '\n', preserving input order. There is no partial
// prediction output, no label comparison, no softmax, and no accuracy claim.
//
// Returns 0 only after the complete N-line prediction payload has been written.
// No validation, model, input, CUDA, or non-finite-output failure writes a
// prediction byte. Diagnostics are written only to `diagnostics`; successful
// diagnostics include the final record
//   event=course_infer status=pass samples=N batches=B input_mode=M
// where B is ceil(N / batch_size) and M is "unit", "byte", or "normalized".
//
// When options.profile is true, all CUDA timing events are allocated before the
// batch loop and every batch, including the first, contributes to the profile.
// One ordered record per batch is emitted:
//   event=course_profile_batch index=I batch_samples=S h2d_ms=H prepare_ms=P
//   forward_argmax_ms=F
// followed by the aggregate record:
//   event=course_profile profile_batches=B gpu_h2d_ms=H gpu_prepare_ms=P
//   gpu_forward_argmax_ms=F host_load_model_ms=L host_load_images_ms=I
//   host_cuda_setup_ms=S host_enqueue_batches_ms=E host_readback_sync_ms=R
//   host_write_predictions_ms=W
// (each record is one space-separated line, split here only for readability).
// profile_batches equals the completion record's batches. GPU values are
// device-event durations summed over all batches; host values are separate
// monotonic wall buckets. host_enqueue_batches_ms measures host enqueue cost,
// never device execution. With options.profile false, the fast path creates no
// timing events, performs no per-batch synchronization, and emits neither
// profile record.
int RunCourseInfer(const InferOptions& options, std::ostream& predictions,
                   std::ostream& diagnostics);

}  // namespace course

#endif  // COURSE_RUNTIME_H_
