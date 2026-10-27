#include "course/interface.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Strict host-side parsing and raw-array reading for the course interface.
// Argument, environment, shape, and mode violations are std::invalid_argument;
// file access, size, and end-of-stream violations are std::runtime_error.
// Image and label payloads are consumed only after the exact file size and the
// checked extent have been validated, so a malformed declaration is rejected
// before any large result buffer is allocated.

namespace course {
namespace {

const std::uint32_t kChannels = 1;
const std::uint32_t kHeight = 28;
const std::uint32_t kWidth = 28;
const std::uint32_t kClasses = 10;
const std::uint32_t kDefaultBatchSize = 128;
const std::uint32_t kDefaultInferenceBatch = 1024;
const std::uint32_t kDefaultEpochs = 20;
const std::uint64_t kDefaultSeed = 1337;
const std::uint32_t kMinimumEpochs = 1;
const std::uint32_t kMaximumEpochs = 1000;
const std::uint32_t kMinimumInferenceBatch = 1;
const std::uint32_t kMaximumInferenceBatch = 16384;

[[noreturn]] void ArgumentFail(const std::string& message) {
  throw std::invalid_argument(message);
}

[[noreturn]] void FileFail(const std::string& path,
                           const std::string& message) {
  throw std::runtime_error(path + ": " + message);
}

// Parses a non-empty unsigned decimal uint32 with no sign, padding, or
// trailing characters, rejecting overflow before narrowing.
std::uint32_t ParseU32(const char* text, const char* field) {
  if (text == nullptr || *text == '\0') {
    ArgumentFail(std::string(field) + " must be a positive decimal integer");
  }
  std::uint64_t value = 0;
  for (const char* digit = text; *digit != '\0'; ++digit) {
    if (*digit < '0' || *digit > '9') {
      ArgumentFail(std::string(field) +
                   " must contain only decimal digits");
    }
    value = value * 10 + static_cast<std::uint64_t>(*digit - '0');
    if (value > std::numeric_limits<std::uint32_t>::max()) {
      ArgumentFail(std::string(field) + " is out of range for uint32");
    }
  }
  return static_cast<std::uint32_t>(value);
}

// Requires one supported positive geometry; training additionally requires
// N >= 2 at the call site.
void ValidateShape(const Shape& shape) {
  if (shape.n == 0) {
    ArgumentFail("N must be positive");
  }
  if (shape.c != kChannels || shape.h != kHeight || shape.w != kWidth ||
      shape.k != kClasses) {
    ArgumentFail("only C=1, H=28, W=28, K=10 are supported");
  }
}

// Parses N,C,H,W,K from five consecutive argument slots and validates them.
Shape ParseShape(const char* const* values) {
  Shape shape;
  shape.n = ParseU32(values[0], "N");
  shape.c = ParseU32(values[1], "C");
  shape.h = ParseU32(values[2], "H");
  shape.w = ParseU32(values[3], "W");
  shape.k = ParseU32(values[4], "K");
  ValidateShape(shape);
  return shape;
}

std::string RequirePath(const char* value, const char* name) {
  if (value == nullptr || *value == '\0') {
    ArgumentFail(std::string(name) + " must not be empty");
  }
  return std::string(value);
}

// The raw array contract is little-endian IEEE-754 binary32 images and
// little-endian two's-complement 32-bit labels. The target is Linux/x86-64;
// other representations are rejected explicitly instead of being decoded with
// a guessed portable fallback. The non-little-endian branch is a runtime guard
// that cannot be exercised on this host.
void ValidateScalarContract(const std::string& path) {
  if (sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559 ||
      std::numeric_limits<float>::radix != 2 ||
      std::numeric_limits<float>::digits != 24 ||
      std::numeric_limits<float>::max_exponent != 128 ||
      sizeof(std::int32_t) != 4 ||
      !std::numeric_limits<std::int32_t>::is_exact ||
      !std::numeric_limits<std::int32_t>::is_signed ||
      std::numeric_limits<std::int32_t>::digits != 31) {
    FileFail(path,
             "unsupported host scalar representation; raw arrays require "
             "32-bit IEEE-754 binary32 and 32-bit signed int32");
  }
  const std::uint32_t marker = 0x01020304U;
  unsigned char bytes[sizeof(marker)];
  std::memcpy(bytes, &marker, sizeof(marker));
  if (bytes[0] != 0x04U || bytes[1] != 0x03U || bytes[2] != 0x02U ||
      bytes[3] != 0x01U) {
    FileFail(path,
             "unsupported host byte order; raw arrays require little-endian "
             "storage");
  }
}

// Returns the exact image element count after shape validation. Because the
// supported geometry is 1x28x28, the product cannot overflow uint64.
std::uint64_t ImageElementCount(const Shape& shape) {
  return static_cast<std::uint64_t>(shape.n) * shape.c * shape.h * shape.w;
}

// Rejects non-finite values and applies the mode-specific inclusive bounds.
// kNormalized accepts every finite value, including negatives.
void ValidateImageValue(float value, InputMode input_mode, std::size_t index,
                        const std::string& path) {
  if (!std::isfinite(value)) {
    std::ostringstream message;
    message << "image value " << index << " must be finite";
    FileFail(path, message.str());
  }
  bool in_bounds = false;
  switch (input_mode) {
    case InputMode::kUnit:
      in_bounds = value >= 0.0F && value <= 1.0F;
      break;
    case InputMode::kByte:
      in_bounds = value >= 0.0F && value <= 255.0F;
      break;
    case InputMode::kNormalized:
      in_bounds = true;
      break;
    default:
      ArgumentFail("unsupported input mode");
  }
  if (!in_bounds) {
    std::ostringstream message;
    message << "image value " << index
            << " is outside the bounds for the selected input mode";
    FileFail(path, message.str());
  }
}

// Opens path, verifies its exact byte size, and leaves it positioned at zero.
std::ifstream OpenExactSize(const std::string& path,
                            std::uint64_t expected_bytes) {
  std::ifstream input(path.c_str(), std::ios::binary);
  if (!input) {
    FileFail(path, "could not open file for reading");
  }
  input.seekg(0, std::ios::end);
  const std::streamoff end = input.tellg();
  if (end < 0) {
    FileFail(path, "could not determine file size");
  }
  if (static_cast<std::uint64_t>(end) != expected_bytes) {
    std::ostringstream message;
    message << "expected exactly " << expected_bytes << " bytes, found "
            << static_cast<std::uint64_t>(end);
    FileFail(path, message.str());
  }
  input.seekg(0, std::ios::beg);
  if (!input) {
    FileFail(path, "could not rewind file for reading");
  }
  return input;
}

}  // namespace

InputMode ParseInputMode(const char* value) {
  if (value == nullptr) {
    return InputMode::kUnit;
  }
  if (std::strcmp(value, "unit") == 0) {
    return InputMode::kUnit;
  }
  if (std::strcmp(value, "byte") == 0) {
    return InputMode::kByte;
  }
  if (std::strcmp(value, "normalized") == 0) {
    return InputMode::kNormalized;
  }
  ArgumentFail("input mode must be exactly unit, byte, or normalized");
}

TrainOptions ParseTrainArgs(int argc, const char* const* argv,
                            const char* input_mode, const char* epochs) {
  if (argv == nullptr) {
    ArgumentFail("train arguments are missing");
  }
  if (argc != 9) {
    ArgumentFail("usage: train images.f32 labels.i32 N C H W K work_dir");
  }

  TrainOptions options;
  options.images_path = RequirePath(argv[1], "images path");
  options.labels_path = RequirePath(argv[2], "labels path");
  options.shape = ParseShape(argv + 3);
  if (options.shape.n < 2) {
    ArgumentFail("training N must be at least 2");
  }
  options.work_dir = RequirePath(argv[8], "work directory");
  options.input_mode = ParseInputMode(input_mode);
  options.epochs = kDefaultEpochs;
  if (epochs != nullptr) {
    options.epochs = ParseU32(epochs, "COURSE_EPOCHS");
    if (options.epochs < kMinimumEpochs ||
        options.epochs > kMaximumEpochs) {
      ArgumentFail("COURSE_EPOCHS must be in [1, 1000]");
    }
  }
  options.batch_size = kDefaultBatchSize;
  options.seed = kDefaultSeed;
  return options;
}

InferOptions ParseInferArgs(int argc, const char* const* argv,
                            const char* model_dir, const char* batch,
                            const char* profile) {
  if (argv == nullptr) {
    ArgumentFail("infer arguments are missing");
  }
  if (argc != 7) {
    ArgumentFail("usage: infer images.f32 N C H W K");
  }

  InferOptions options;
  options.images_path = RequirePath(argv[1], "images path");
  options.shape = ParseShape(argv + 2);
  options.model_dir = RequirePath(model_dir, "COURSE_MODEL_DIR");
  options.batch_size = kDefaultInferenceBatch;
  if (batch != nullptr) {
    options.batch_size = ParseU32(batch, "COURSE_INFER_BATCH");
    if (options.batch_size < kMinimumInferenceBatch ||
        options.batch_size > kMaximumInferenceBatch) {
      ArgumentFail("COURSE_INFER_BATCH must be in [1, 16384]");
    }
  }
  options.profile = profile != nullptr && std::strcmp(profile, "1") == 0;
  return options;
}

std::vector<float> ReadImages(const std::string& path, const Shape& shape,
                              InputMode input_mode) {
  if (path.empty()) {
    ArgumentFail("image path must not be empty");
  }
  ValidateScalarContract(path);
  ValidateShape(shape);
  // Reject an unknown enum value before opening or allocating anything.
  switch (input_mode) {
    case InputMode::kUnit:
    case InputMode::kByte:
    case InputMode::kNormalized:
      break;
    default:
      ArgumentFail("unsupported input mode");
  }

  const std::uint64_t elements = ImageElementCount(shape);
  if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
    FileFail(path, "declared image extent overflows byte count");
  }
  const std::uint64_t expected_bytes = elements * sizeof(float);
  std::ifstream input = OpenExactSize(path, expected_bytes);

  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    FileFail(path, "declared image extent does not fit addressable storage");
  }
  std::vector<float> values(static_cast<std::size_t>(elements));

  // The target is little-endian x86; raw FP32 bytes are transferred directly
  // so every fractional bit pattern is retained without quantization.
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(expected_bytes));
  if (!input ||
      static_cast<std::uint64_t>(input.gcount()) != expected_bytes) {
    FileFail(path, "image payload is truncated");
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    ValidateImageValue(values[index], input_mode, index, path);
  }
  return values;
}

std::vector<std::uint8_t> ReadLabels(const std::string& path,
                                     const Shape& shape) {
  if (path.empty()) {
    ArgumentFail("labels path must not be empty");
  }
  ValidateScalarContract(path);
  ValidateShape(shape);
  const std::uint64_t count = static_cast<std::uint64_t>(shape.n);
  if (count > std::numeric_limits<std::uint64_t>::max() /
                  sizeof(std::int32_t)) {
    FileFail(path, "declared label extent overflows byte count");
  }
  const std::uint64_t expected_bytes = count * sizeof(std::int32_t);
  std::ifstream input = OpenExactSize(path, expected_bytes);

  if (count > std::numeric_limits<std::size_t>::max() /
                  sizeof(std::int32_t)) {
    FileFail(path, "declared label extent does not fit addressable storage");
  }
  std::vector<std::int32_t> raw(static_cast<std::size_t>(count));
  input.read(reinterpret_cast<char*>(raw.data()),
             static_cast<std::streamsize>(expected_bytes));
  if (!input ||
      static_cast<std::uint64_t>(input.gcount()) != expected_bytes) {
    FileFail(path, "label payload is truncated");
  }

  // Validate the full int32 range before narrowing so -1 can never become 255
  // and 10 can never become a valid class.
  std::vector<std::uint8_t> labels;
  labels.reserve(raw.size());
  for (std::size_t index = 0; index < raw.size(); ++index) {
    if (raw[index] < 0 || raw[index] >= static_cast<std::int32_t>(kClasses)) {
      std::ostringstream message;
      message << "label " << index << " must be in [0, 10)";
      FileFail(path, message.str());
    }
    labels.push_back(static_cast<std::uint8_t>(raw[index]));
  }
  return labels;
}

}  // namespace course
