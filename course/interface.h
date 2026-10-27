#ifndef COURSE_INTERFACE_H_
#define COURSE_INTERFACE_H_

#include <cstdint>
#include <string>
#include <vector>

// Strict host-side contract for the course `train` and `infer` commands and
// for reading the raw array files they consume. Argument, environment, and
// shape violations are usage errors reported by throwing std::invalid_argument;
// file access or content violations throw std::runtime_error. No function in
// this interface allocates based on an unvalidated file extent.

namespace course {

// Selects how raw FP32 image values are interpreted before the fixed
// normalization (x - 0.1307F) / 0.3081F is applied on the GPU.
// kUnit: values in [0, 1] used as-is.
// kByte: values in [0, 255] scaled by 1/255.
// kNormalized: already-normalized finite values used as-is, including
//              negative values.
enum class InputMode { kUnit, kByte, kNormalized };

// Maps a mode name to its enum value. A null pointer selects kUnit. Only the
// exact spellings "unit", "byte", and "normalized" are accepted; empty,
// padded, mixed-case, or otherwise unknown names are rejected instead of
// being guessed or defaulted.
InputMode ParseInputMode(const char* value);

// The fixed supported problem geometry: batch, channels, height, width, and
// class count. C=1, H=28, W=28, and K=10 are the only supported dimensions.
// N is a positive uint32 and must be at least 2 for training.
struct Shape {
  std::uint32_t n;
  std::uint32_t c;
  std::uint32_t h;
  std::uint32_t w;
  std::uint32_t k;
};

// Owns the parsed `train` request. Defaults: input mode kUnit, 20 epochs,
// batch size 128, and seed 1337.
struct TrainOptions {
  std::string images_path;
  std::string labels_path;
  std::string work_dir;
  Shape shape;
  InputMode input_mode;
  std::uint32_t epochs;
  std::uint32_t batch_size;
  std::uint64_t seed;
};

// Owns the parsed `infer` request. Defaults: batch size 1024 and profiling
// disabled.
struct InferOptions {
  std::string images_path;
  std::string model_dir;
  Shape shape;
  std::uint32_t batch_size;
  bool profile;
};

// Parses exactly the documented train form:
//   train images.f32 labels.i32 N C H W K work_dir
// argc must therefore be exactly 9. input_mode is the optional
// COURSE_INPUT_MODE value and epochs is the optional COURSE_EPOCHS value
// (1..1000); a null optional pointer selects the default. The batch size and
// seed are fixed by the course contract.
//
// Rejects otherwise: a different argc, empty required paths, a non-numeric,
// signed, zero, or out-of-range N, training N < 2, unsupported C/H/W/K, an
// empty or unknown input mode, and empty, unknown, or out-of-range epochs.
// The parsed file extents are not inspected here and no file data is read.
TrainOptions ParseTrainArgs(int argc, const char* const* argv,
                            const char* input_mode, const char* epochs);

// Parses exactly the documented infer form:
//   infer images.f32 N C H W K
// argc must therefore be exactly 7. model_dir is the required
// COURSE_MODEL_DIR value and may not be null or empty. batch is the optional
// COURSE_INFER_BATCH value (1..16384; null selects 1024). profile is the
// optional COURSE_PROFILE switch: only the exact value "1" enables it, and
// every other spelling is a disabled diagnostic switch rather than an error.
// There is deliberately no labels or input-mode argument: inference always
// uses the persisted model mode.
//
// Rejects otherwise: a different argc, empty or missing paths/model
// directory, a non-numeric, signed, zero, or out-of-range N, unsupported
// C/H/W/K, and an empty, unknown, or out-of-range batch value.
InferOptions ParseInferArgs(int argc, const char* const* argv,
                            const char* model_dir, const char* batch,
                            const char* profile);

// Reads exactly shape.n * shape.c * shape.h * shape.w little-endian FP32
// values from path in NCHW order and returns them bit-for-bit unchanged; raw
// inputs are never quantized through uint8. shape and the mode-appropriate
// value bounds are validated first: kUnit requires [0, 1], kByte requires
// [0, 255], kNormalized accepts every finite value (including negatives), and
// NaN or infinity is always rejected. The host must provide little-endian
// storage with 32-bit IEEE-754 binary32 floats; any other representation is
// rejected at runtime rather than decoded with a guessed fallback. Unsupported
// dimensions, zero batch, empty path, or an unknown mode throw
// std::invalid_argument; an unreadable, short, long, non-finite, or
// out-of-mode-range payload throws std::runtime_error. File size is checked
// before allocating the result.
std::vector<float> ReadImages(const std::string& path, const Shape& shape,
                              InputMode input_mode);

// Reads exactly shape.n little-endian int32 labels from path and returns them
// narrowed to uint8. Every int32 value is validated to lie in [0, 10) before
// narrowing, so negative or oversized labels can never wrap into a valid
// class; the host must use little-endian 32-bit signed two's-complement int32
// or the read is rejected at runtime. Unsupported or zero-batch shapes and an
// empty path throw std::invalid_argument; an unreadable, short, or long file,
// or any out-of-range label, throws std::runtime_error.
std::vector<std::uint8_t> ReadLabels(const std::string& path,
                                     const Shape& shape);

}  // namespace course

#endif  // COURSE_INTERFACE_H_
