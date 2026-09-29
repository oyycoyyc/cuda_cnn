#ifndef INCLUDE_LENET_H_
#define INCLUDE_LENET_H_

#include "parameters.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

// Configures one decoupled AdamW update. beta1 and beta2 must be finite in
// [0,1), epsilon finite and positive, and weight_decay finite and nonnegative.
// Weight decay is applied to model weights only; all five biases use zero.
struct AdamWConfig {
  float beta1;
  float beta2;
  float epsilon;
  float weight_decay;
};

class LeNetTestAccess;

// Owns the fixed-capacity device state for modern LeNet. The constructor's
// stream is borrowed, must remain valid for this object's lifetime, and is the
// only stream used. The class never allocates device storage after construction
// and is neither copyable nor assignable.
class LeNet {
 public:
  // Constructs capacity for batch sizes in [1, maximum_batch_size], checks
  // available device memory before its single allocation, initializes the ten
  // canonical parameter tensors from seed, and zeroes gradients and moments.
  // maximum_batch_size must be positive. CUDA/query/allocation/copy failures
  // throw std::runtime_error; insufficient memory reports required, free, and
  // total bytes. stream is caller-owned and may be the valid null stream.
  LeNet(int maximum_batch_size, std::uint64_t seed, cudaStream_t stream);

  // Releases all model-owned device storage. The borrowed stream is neither
  // synchronized nor destroyed; CUDA deallocation errors cannot escape.
  ~LeNet();

  // Copy construction is forbidden because the model uniquely owns device
  // parameters, optimizer state, activations, and workspace.
  LeNet(const LeNet&) = delete;

  // Copy assignment is forbidden because device ownership is unique.
  LeNet& operator=(const LeNet&) = delete;

  // Enqueues the exact Conv-ReLU-Pool, Conv-ReLU-Pool, and three-layer MLP
  // forward chain and returns the stable model-owned device logits pointer for
  // contiguous [batch_size][10] FP32 output. normalized_images is a non-null
  // caller-owned device [batch_size][1][28][28] buffer that remains live until
  // Backward completes. batch_size must be within constructed capacity. A
  // successful call replaces any prior forward/gradient state and permits one
  // matching Backward. No synchronization or allocation occurs; launch
  // failures throw runtime_error and leave no consumable state.
  const float* Forward(const float* normalized_images, int batch_size);

  // Enqueues the exact reverse chain for the most recent Forward, populating
  // all ten parameter gradients and model-owned input-gradient storage.
  // logits_gradient is a non-null caller-owned device [batch_size][10] buffer,
  // batch_size must match that Forward, and both external buffers remain live
  // through stream work. The call consumes that Forward and, on success,
  // permits one AdamWStep. Repeated or stale calls throw std::invalid_argument.
  // No synchronization or allocation occurs.
  void Backward(const float* logits_gradient, int batch_size);

  // Enqueues AdamW updates for all ten canonical tensors using gradients from
  // the most recent Backward. global_step must be at least one; learning_rate
  // must be finite and nonnegative and config must satisfy AdamWConfig's
  // contract. Bias corrections are computed on the host in double precision.
  // Decay is config.weight_decay for weights and zero for biases. No allocation
  // or synchronization occurs. The call consumes current gradients and
  // invalidates saved forward state; stale/repeated calls and invalid arguments
  // throw std::invalid_argument.
  void AdamWStep(std::uint64_t global_step, float learning_rate,
                 const AdamWConfig& config);

  // Returns a newly host-owned canonical ParameterSet copied from device state.
  // The copy is enqueued on the borrowed stream and that stream is synchronized
  // before return. CUDA copy or synchronization failures throw runtime_error.
  ParameterSet ExportParameters() const;

  // Copies device parameters into an existing caller-owned canonical
  // ParameterSet without resizing any tensor payload. destination must be
  // non-null and already satisfy the exact ten-tensor LeNet schema; it remains
  // owned by the caller and may be reused across calls. Validation failures
  // throw std::invalid_argument before CUDA work. Copies use the borrowed stream
  // and synchronize it before return; CUDA failures throw std::runtime_error.
  void ExportParameters(ParameterSet* destination) const;

  // Validates exact canonical schema before enqueueing copies from host-owned
  // parameters into device state, then synchronizes the borrowed stream before
  // return. A schema-valid call invalidates saved forward/gradient state before
  // copying; input storage is not retained. Schema violations throw
  // std::invalid_argument without changing state; CUDA copy/synchronization
  // failures throw runtime_error and leave no consumable state.
  void ImportParameters(const ParameterSet& parameters);

  // Deterministically scans owned FP32 tensors and synchronizes the borrowed
  // stream once at this phase boundary. phase is included verbatim in a
  // runtime_error naming the first non-finite tensor and linear index. Empty
  // phase strings are allowed. The method allocates no device storage.
  void RequireFinite(const std::string& phase) const;

  // Returns the exact number of device bytes in the model's single fixed arena,
  // including parameters, gradients, moments, maximum-batch activations and
  // activation gradients, pool winners, and finite-scan storage. It performs no
  // CUDA calls, allocation, or synchronization and cannot fail.
  std::size_t RequiredDeviceBytes() const;

 private:
  friend class LeNetTestAccess;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Provides one narrow structural assertion required by the CUDA operator tests.
// It grants no pointer or ownership access and is always declared identically,
// independent of test macros, so LeNet's class definition remains ODR-safe.
class LeNetTestAccess {
 public:
  // Returns true only after a successful Forward whose exact pointer supplied
  // to FC1 was the model's pool2 storage. No CUDA call or synchronization occurs.
  static bool Fc1InputAliasesPool2(const LeNet& model) noexcept;

  // Returns true when all model finite-scan gradient/moment labels were built
  // during construction, before any forward/backward batch phase. It performs
  // no allocation, CUDA call, synchronization, or ownership transfer.
  static bool FiniteDiagnosticNamesPrepared(const LeNet& model) noexcept;
};

#endif  // INCLUDE_LENET_H_
