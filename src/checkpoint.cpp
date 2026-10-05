#include "checkpoint.h"

#include "binary_io.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {

// On-disk identity: the LNETC01 magic, format and architecture versions, the
// canonical tensor count, and the fixed normalization pair.
const std::array<unsigned char, 8> kMagic{{'L', 'N', 'E', 'T', 'C', '0', '1',
                                           0}};
const std::uint32_t kFormatVersion = 1;
const std::uint32_t kArchitectureId = 1;
const std::uint32_t kTensorCount = 10;
const float kNormalizationMean = 0.1307F;
const float kNormalizationStddev = 0.3081F;

// Requires a one-based epoch, finite unit-interval accuracy, and the fixed
// normalization constants.
void ValidateMetadata(const std::string& path,
                      const CheckpointMetadata& metadata) {
  if (metadata.best_epoch == 0) {
    binary_io::Fail(path, "best epoch must be one-based");
  }
  if (!std::isfinite(metadata.validation_accuracy) ||
      metadata.validation_accuracy < 0.0F ||
      metadata.validation_accuracy > 1.0F) {
    binary_io::Fail(path, "validation accuracy must be finite and in [0, 1]");
  }
  if (metadata.normalization_mean != kNormalizationMean) {
    binary_io::Fail(path, "normalization mean must equal 0.1307");
  }
  if (metadata.normalization_stddev != kNormalizationStddev) {
    binary_io::Fail(path,
                    "normalization standard deviation must equal 0.3081");
  }
}

// Validates metadata, canonical schema, and value finiteness before any write.
void ValidateCheckpoint(const std::string& path,
                        const Checkpoint& checkpoint) {
  ValidateMetadata(path, checkpoint.metadata);
  try {
    ValidateLenetParameters(checkpoint.parameters);
  } catch (const std::invalid_argument& error) {
    binary_io::Fail(path, error.what());
  }
  for (std::size_t tensor = 0; tensor < checkpoint.parameters.size(); ++tensor) {
    for (float value : checkpoint.parameters[tensor].values) {
      if (!std::isfinite(value)) {
        std::ostringstream invariant;
        invariant << "parameter tensor " << tensor
                  << " values must be finite";
        binary_io::Fail(path, invariant.str());
      }
    }
  }
}

// Names a unique temporary beside the destination so replacement stays within
// one filesystem and never exposes a partial checkpoint.
std::string TemporaryPath(const std::string& path) {
  static std::atomic<unsigned long> sequence{0};
#ifdef _WIN32
  const unsigned long process = GetCurrentProcessId();
#else
  const unsigned long process = static_cast<unsigned long>(getpid());
#endif
  std::ostringstream temporary;
  temporary << path << ".tmp." << process << '.' << sequence.fetch_add(1);
  return temporary.str();
}

// Atomically replaces the destination, leaving the prior checkpoint intact on
// failure.
bool ReplaceFile(const std::string& temporary, const std::string& path) {
#ifdef _WIN32
  // MoveFileEx supplies replace-existing behavior on Windows, where
  // std::rename does not replace a destination file.
  return MoveFileExA(temporary.c_str(), path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
  // temporary is in path's directory, so POSIX rename atomically replaces an
  // existing destination without exposing a partial checkpoint.
  return std::rename(temporary.c_str(), path.c_str()) == 0;
#endif
}

// Writes the full checkpoint: the 40-byte header, then ten 60-byte metadata
// records each followed by its contiguous float payload.
void WriteCheckpointContents(std::ostream& output, const std::string& path,
                             const Checkpoint& checkpoint) {
  // The 40-byte header is written field-by-field in little-endian order:
  // magic[8], version u32, architecture u32, tensor count u32, one-based epoch
  // u32, accuracy f32, normalization mean f32, normalization stddev f32, and
  // reserved-zero u32. No in-memory aggregate representation reaches disk.
  binary_io::WriteExact(output, kMagic.data(), kMagic.size(), path,
                        "could not write checkpoint magic");
  binary_io::WriteU32LE(output, kFormatVersion, path,
                        "could not write checkpoint version");
  binary_io::WriteU32LE(output, kArchitectureId, path,
                        "could not write checkpoint architecture");
  binary_io::WriteU32LE(output, kTensorCount, path,
                        "could not write checkpoint tensor count");
  binary_io::WriteU32LE(output, checkpoint.metadata.best_epoch, path,
                        "could not write checkpoint epoch");
  binary_io::WriteF32LE(output, checkpoint.metadata.validation_accuracy, path,
                        "could not write checkpoint accuracy");
  binary_io::WriteF32LE(output, checkpoint.metadata.normalization_mean, path,
                        "could not write normalization mean");
  binary_io::WriteF32LE(output, checkpoint.metadata.normalization_stddev, path,
                        "could not write normalization standard deviation");
  binary_io::WriteU32LE(output, 0, path,
                        "could not write checkpoint reserved field");

  // Emit tensor records in canonical order, zero-padding each 32-byte name.
  const std::array<ParameterSpec, 10>& specs = LenetParameterSpecs();
  for (std::size_t index = 0; index < specs.size(); ++index) {
    const ParameterSpec& spec = specs[index];
    const ParameterTensor& tensor = checkpoint.parameters[index];
    std::array<unsigned char, 32> name{{0}};
    for (std::size_t character = 0; spec.name[character] != '\0'; ++character) {
      name[character] = static_cast<unsigned char>(spec.name[character]);
    }

    // Each 60-byte tensor metadata record is name[32], rank u32, four shape
    // u32 values (unused slots are one), then element_count u64, all integers
    // little-endian. Its contiguous row-major FP32 payload immediately follows.
    binary_io::WriteExact(output, name.data(), name.size(), path,
                          "could not write tensor name");
    binary_io::WriteU32LE(output, spec.rank, path,
                          "could not write tensor rank");
    for (std::uint32_t dimension : spec.dimensions) {
      binary_io::WriteU32LE(output, dimension, path,
                            "could not write tensor dimensions");
    }
    binary_io::WriteU64LE(output, spec.element_count, path,
                          "could not write tensor element count");
    for (float value : tensor.values) {
      binary_io::WriteF32LE(output, value, path,
                            "could not write parameter value");
    }
  }
}

// Reads one 32-byte tensor name and checks it against the canonical spec,
// reporting the tensor index when name or order is invalid.
void ReadAndValidateName(std::istream& input, const std::string& path,
                         const ParameterSpec& spec, std::size_t index) {
  std::array<unsigned char, 32> actual;
  binary_io::ReadExact(input, actual.data(), actual.size(), path,
                       "tensor metadata name is truncated");
  std::array<unsigned char, 32> expected{{0}};
  for (std::size_t character = 0; spec.name[character] != '\0'; ++character) {
    expected[character] = static_cast<unsigned char>(spec.name[character]);
  }
  if (actual != expected) {
    std::ostringstream invariant;
    invariant << "tensor " << index << " name or canonical order is invalid";
    binary_io::Fail(path, invariant.str());
  }
}

}  // namespace

// Validates before writing, saves through a same-directory temporary, and
// keeps the destination untouched if any step fails.
void SaveCheckpoint(const std::string& path, const Checkpoint& checkpoint) {
  ValidateCheckpoint(path, checkpoint);

  // Write to a unique temporary so a partial checkpoint never replaces the
  // destination, and remove it on any failure.
  const std::string temporary = TemporaryPath(path);
  try {
    std::ofstream output(temporary,
                         std::ios::binary | std::ios::out | std::ios::trunc);
    if (!output) {
      binary_io::Fail(path, "could not open temporary checkpoint for writing");
    }
    WriteCheckpointContents(output, path, checkpoint);
    output.flush();
    if (!output) {
      binary_io::Fail(path, "could not flush temporary checkpoint");
    }
    output.close();
    if (output.fail()) {
      binary_io::Fail(path, "could not close temporary checkpoint");
    }
  } catch (...) {
    std::remove(temporary.c_str());
    throw;
  }

  // Publish by replacement only after the temporary is flushed and closed.
  if (!ReplaceFile(temporary, path)) {
    std::remove(temporary.c_str());
    binary_io::Fail(path, "could not atomically replace checkpoint");
  }
}

// Parses and validates the header before allocating any parameter storage; the
// ten metadata records are validated in order afterward.
Checkpoint LoadCheckpoint(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    binary_io::Fail(path, "could not open checkpoint for reading");
  }

  // Parse the 40-byte little-endian header field-by-field in its documented
  // disk order; values are validated before any tensor allocation or payload.
  std::array<unsigned char, 8> magic;
  binary_io::ReadExact(input, magic.data(), magic.size(), path,
                       "checkpoint header magic is truncated");
  if (magic != kMagic) {
    binary_io::Fail(path, "checkpoint magic is invalid");
  }
  const std::uint32_t version = binary_io::ReadU32LE(
      input, path, "checkpoint header version is truncated");
  if (version != kFormatVersion) {
    binary_io::Fail(path, "checkpoint version is unsupported");
  }
  const std::uint32_t architecture = binary_io::ReadU32LE(
      input, path, "checkpoint header architecture is truncated");
  if (architecture != kArchitectureId) {
    binary_io::Fail(path, "checkpoint architecture is unsupported");
  }
  const std::uint32_t tensor_count = binary_io::ReadU32LE(
      input, path, "checkpoint header tensor count is truncated");
  if (tensor_count != kTensorCount) {
    binary_io::Fail(path, "checkpoint tensor count must equal 10");
  }

  Checkpoint checkpoint;
  checkpoint.metadata.best_epoch = binary_io::ReadU32LE(
      input, path, "checkpoint header epoch is truncated");
  checkpoint.metadata.validation_accuracy = binary_io::ReadF32LE(
      input, path, "checkpoint header accuracy is truncated");
  checkpoint.metadata.normalization_mean = binary_io::ReadF32LE(
      input, path, "checkpoint header normalization mean is truncated");
  checkpoint.metadata.normalization_stddev = binary_io::ReadF32LE(
      input, path,
      "checkpoint header normalization standard deviation is truncated");
  const std::uint32_t reserved = binary_io::ReadU32LE(
      input, path, "checkpoint header reserved field is truncated");
  if (reserved != 0) {
    binary_io::Fail(path, "checkpoint reserved field must be zero");
  }
  ValidateMetadata(path, checkpoint.metadata);

  // Allocate canonical storage only after the header validates.
  checkpoint.parameters = CreateLenetParameters();
  const std::array<ParameterSpec, 10>& specs = LenetParameterSpecs();
  for (std::size_t index = 0; index < specs.size(); ++index) {
    const ParameterSpec& spec = specs[index];

    // Parse one 60-byte metadata record in little-endian field order, checking
    // it against the sole canonical ParameterSpec before reading its FP32 data.
    ReadAndValidateName(input, path, spec, index);
    const std::uint32_t rank = binary_io::ReadU32LE(
        input, path, "tensor metadata rank is truncated");
    if (rank != spec.rank) {
      std::ostringstream invariant;
      invariant << "tensor " << index << " rank is invalid";
      binary_io::Fail(path, invariant.str());
    }
    for (std::size_t dimension = 0; dimension < spec.dimensions.size();
         ++dimension) {
      const std::uint32_t value = binary_io::ReadU32LE(
          input, path, "tensor metadata dimensions are truncated");
      if (value != spec.dimensions[dimension]) {
        std::ostringstream invariant;
        invariant << "tensor " << index << " dimensions are invalid";
        binary_io::Fail(path, invariant.str());
      }
    }
    const std::uint64_t element_count = binary_io::ReadU64LE(
        input, path, "tensor metadata element count is truncated");
    if (element_count != spec.element_count) {
      std::ostringstream invariant;
      invariant << "tensor " << index << " element count is invalid";
      binary_io::Fail(path, invariant.str());
    }

    for (float& value : checkpoint.parameters[index].values) {
      value = binary_io::ReadF32LE(input, path,
                                   "parameter value payload is truncated");
      if (!std::isfinite(value)) {
        std::ostringstream invariant;
        invariant << "tensor " << index << " values must be finite";
        binary_io::Fail(path, invariant.str());
      }
    }
  }

  // Require end-of-stream with no trailing bytes.
  char trailing = 0;
  if (input.get(trailing)) {
    binary_io::Fail(path, "checkpoint has trailing bytes");
  }
  if (input.bad()) {
    binary_io::Fail(path, "could not check checkpoint trailing bytes");
  }
  return checkpoint;
}
