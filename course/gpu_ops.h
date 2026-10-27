#ifndef COURSE_GPU_OPS_H_
#define COURSE_GPU_OPS_H_

#include "course/interface.h"

#include <cstdint>

#include <cuda_runtime_api.h>

// GPU helpers for the course inference path: FP32 input preparation without a
// uint8 intermediate representation, and a label-free int32 argmax with a
// sticky non-finite diagnostic. Both launchers only validate their host
// arguments and enqueue work on the caller's stream; neither synchronizes, and
// neither reads back device data.

namespace course {

// Converts batch contiguous FP32 source images in packed NCHW order
// [batch][1][28][28] into the exact normalized FP32 form consumed by LeNet:
//   kUnit:       (value - 0.1307F) / 0.3081F
//   kByte:       (value / 255.0F - 0.1307F) / 0.3081F
//   kNormalized: value unchanged
// Source values are consumed as FP32 in every mode; no mode passes through a
// uint8 or other quantized representation, so fractional kByte values are
// preserved with their FP32 precision. With augment == true each sample is
// translated by TranslationOffsetForSample(seed, epoch,
// original_indices[sample]); positive dx/dy move content right/down, and
// out-of-range sources supply the normalized black pixel
// (0.0F - 0.1307F) / 0.3081F in every mode, including kNormalized.
//
// packed and normalized are caller-owned device buffers of at least
// batch*784 elements. Both are non-null, disjoint, and remain live through
// stream work; packed is read-only while normalized is the written output.
// original_indices is a device buffer of at least batch uint32 values, is
// read-only, and is required only when augment is true; it may be null
// otherwise. epoch is ignored unless augment is true, where it must be at least
// one. batch must be positive and fit one CUDA x-grid. Invalid batch/epoch
// values, an unknown InputMode enumerator, a required null pointer, or
// overlapping packed/normalized extents throw std::invalid_argument before
// launch; dimensions whose element extent or grid exceeds the representable
// capacity throw std::overflow_error.
// Immediate launch failure throws std::runtime_error; no synchronization is
// performed, so asynchronous errors surface at a later boundary.
void LaunchPrepareInput(const float* packed,
                        const std::uint32_t* original_indices,
                        float* normalized, int batch, InputMode mode,
                        std::uint64_t seed, std::uint32_t epoch, bool augment,
                        cudaStream_t stream);

// Writes the smallest-index argmax of every row of finite FP32 row-major logits
// [batch][classes] into int32 predictions[batch]. Ties are broken by the
// smallest class index (strict greater-than comparison), and no label is read
// or required. Sets *bad_output to 1 (sticky) when any logit in any row is NaN
// or infinite; the flag is never cleared here, so the caller zeroes it exactly
// once before one full inference request and any later argmax call in that
// request leaves a previously observed failure visible. All rows and all
// classes are examined, including rows whose argmax has already been found.
//
// logits, predictions, and bad_output are caller-owned device buffers that
// remain live through stream work. logits has at least batch*classes elements,
// predictions exactly batch int32 elements, and bad_output one int element; all
// three are non-null and pairwise disjoint. batch must be positive and classes
// must be in [1, 256]. Invalid dimensions or null pointers or overlapping
// buffer extents throw std::invalid_argument before launch; unrepresentable
// element extents throw std::overflow_error. Device logit values are not
// host-validated. Immediate launch failure throws std::runtime_error; no
// synchronization is performed, so asynchronous errors surface at a later
// boundary.
void LaunchArgmax(const float* logits, std::int32_t* predictions,
                  int* bad_output, int batch, int classes,
                  cudaStream_t stream);

}  // namespace course

#endif  // COURSE_GPU_OPS_H_
