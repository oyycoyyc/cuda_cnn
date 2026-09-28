#ifndef INCLUDE_CHECKPOINT_H_
#define INCLUDE_CHECKPOINT_H_

#include "parameters.h"

#include <cstdint>
#include <string>

// Owns the scalar training metadata stored in the 40-byte checkpoint header;
// it has no dynamic storage or aliases. Fields are serialized individually,
// not by object representation. SaveCheckpoint requires the epoch and accuracy
// to be valid and the normalization constants to match the model contract;
// constructing or copying this aggregate does not throw.
struct CheckpointMetadata {
  // One-based epoch at which validation accuracy was measured.
  std::uint32_t best_epoch;
  // Finite validation accuracy in the inclusive range [0, 1].
  float validation_accuracy;
  // Input normalization mean, required to equal FP32 0.1307.
  float normalization_mean;
  // Input normalization standard deviation, required to equal FP32 0.3081.
  float normalization_stddev;
};

// Owns checkpoint metadata and the canonical ten row-major FP32 parameter
// tensors described by LenetParameterSpecs. Its buffers do not alias one
// another or external storage. SaveCheckpoint validates all fields and may
// throw; ordinary aggregate construction and copying have container semantics.
struct Checkpoint {
  // Scalar header metadata owned directly by this checkpoint.
  CheckpointMetadata metadata;
  // Canonically ordered tensor names, shapes, and values owned by this object.
  ParameterSet parameters;
};

// Validates checkpoint metadata, canonical parameter layout, and every finite
// FP32 value, then writes the documented little-endian 40-byte header followed
// by ten 60-byte metadata records and row-major payloads. path names caller-
// managed destination storage but is never retained; checkpoint is read
// without mutation or aliasing. The destination parent must exist. It writes
// and closes a same-directory temporary before atomically replacing path.
// Validation, I/O, or replacement errors throw a path-qualified
// std::runtime_error;
// standard allocation exceptions may also propagate.
void SaveCheckpoint(const std::string& path, const Checkpoint& checkpoint);

// Reads and validates exactly one documented little-endian checkpoint from
// path, returning ownership of independent metadata and ten canonical
// row-major FP32 tensor buffers. The path string and file storage are not
// retained or aliased. The file must be readable, complete, contain no trailing
// bytes, and satisfy every header, tensor-layout, and finiteness invariant;
// violations throw a path-qualified std::runtime_error, while standard
// allocation exceptions may propagate.
Checkpoint LoadCheckpoint(const std::string& path);

#endif  // INCLUDE_CHECKPOINT_H_
