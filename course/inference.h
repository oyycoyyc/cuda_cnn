#ifndef COURSE_INFERENCE_H_
#define COURSE_INFERENCE_H_

#include "parameters.h"

#include <cstddef>
#include <memory>

#include <cuda_runtime_api.h>

// Forward-only LeNet inference model for the course `infer` command. It owns a
// single fixed device arena sized for the supplied capacity and never allocates
// device storage after construction. Its forward chain is the original LeNet
// Conv-ReLU-Pool, Conv-ReLU-Pool, and three-layer MLP chain launched in the
// original order with FP32 arithmetic, so logits agree bit for bit with the
// original reference model given the same already-normalized device input and
// the same parameters.

namespace course {

// Owns one checked, aligned device arena holding only the ten supplied
// parameter payloads plus forward activations and pooling winners for
// maximum_batch samples. No gradients, optimizer moments, random
// initialization, or training-only storage is allocated. The borrowed stream
// is used for every copy and launch and must remain valid for this object's
// lifetime.
class InferenceModel {
 public:
  // Validates maximum_batch >= 1 and the exact canonical ten-tensor schema,
  // rejects any non-finite supplied value, checks available device memory, then
  // reserves the single arena and copies the validated parameters to device in
  // canonical order. The constructor does not randomize weights and does not
  // zero the whole arena. maximum_batch must be positive and every parameter
  // value must be finite. Invalid capacity, schema, or parameter values throw
  // std::invalid_argument before any device allocation; insufficient memory or
  // CUDA query/copy/synchronization failures throw std::runtime_error, and
  // insufficient memory reports required, free, and total bytes. The stream is
  // synchronized once after the parameter copy so Forward may be called
  // immediately.
  InferenceModel(int maximum_batch, const ParameterSet& parameters,
                 cudaStream_t stream);

  // Releases the model-owned arena. The borrowed stream is neither
  // synchronized nor destroyed, and CUDA deallocation errors cannot escape.
  ~InferenceModel();

  // Copy construction is forbidden because the arena has unique ownership.
  InferenceModel(const InferenceModel&) = delete;

  // Copy assignment is forbidden because the arena has unique ownership.
  InferenceModel& operator=(const InferenceModel&) = delete;

  // Enqueues the forward-only LeNet chain for the first batch samples of
  // normalized and returns the stable model-owned device FP32 logits pointer
  // for contiguous [batch][10] output. normalized is a non-null caller-owned
  // device [batch][1][28][28] buffer containing already-normalized values that
  // remains live through stream work. batch must be in [1, maximum_batch].
  // Repeated calls with any batch size in that range are supported, and unused
  // capacity rows hold no observable output. No device allocation or
  // synchronization occurs; invalid arguments throw std::invalid_argument and
  // immediate launch failure throws std::runtime_error.
  const float* Forward(const float* normalized, int batch);

  // Returns the exact number of device bytes in the single arena, including the
  // ten parameter payloads, forward activations, and pooling winners at
  // maximum_batch capacity. It performs no CUDA call, allocation, or
  // synchronization and cannot fail. At every supported capacity it is
  // strictly smaller than the original trainable LeNet arena, which additionally
  // holds gradients, moments, and finite-scan storage.
  std::size_t RequiredDeviceBytes() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace course

#endif  // COURSE_INFERENCE_H_
