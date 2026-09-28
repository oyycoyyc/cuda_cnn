#ifndef SRC_BINARY_IO_H_
#define SRC_BINARY_IO_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>

namespace binary_io {

[[noreturn]] inline void Fail(const std::string& path,
                              const std::string& invariant) {
  throw std::runtime_error(path + ": " + invariant);
}

inline std::streamsize CheckedStreamSize(std::size_t size,
                                         const std::string& path,
                                         const std::string& invariant) {
  if (size > static_cast<std::size_t>(
                 std::numeric_limits<std::streamsize>::max())) {
    Fail(path, invariant);
  }
  return static_cast<std::streamsize>(size);
}

inline void ReadExact(std::istream& input, void* destination, std::size_t size,
                      const std::string& path,
                      const std::string& invariant) {
  input.read(static_cast<char*>(destination),
             CheckedStreamSize(size, path, invariant));
  if (!input) {
    Fail(path, invariant);
  }
}

inline std::uint32_t ReadU32LE(std::istream& input, const std::string& path,
                               const std::string& invariant) {
  std::uint8_t bytes[4];
  ReadExact(input, bytes, sizeof(bytes), path, invariant);
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8) |
         (static_cast<std::uint32_t>(bytes[2]) << 16) |
         (static_cast<std::uint32_t>(bytes[3]) << 24);
}

inline std::uint64_t ReadU64LE(std::istream& input, const std::string& path,
                               const std::string& invariant) {
  std::uint8_t bytes[8];
  ReadExact(input, bytes, sizeof(bytes), path, invariant);
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < sizeof(bytes); ++index) {
    value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8);
  }
  return value;
}

inline float ReadF32LE(std::istream& input, const std::string& path,
                       const std::string& invariant) {
  static_assert(sizeof(float) == sizeof(std::uint32_t),
                "binary float I/O requires 32-bit float");
  const std::uint32_t bits = ReadU32LE(input, path, invariant);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline void WriteExact(std::ostream& output, const void* source,
                       std::size_t size, const std::string& path,
                       const std::string& invariant) {
  output.write(static_cast<const char*>(source),
               CheckedStreamSize(size, path, invariant));
  if (!output) {
    Fail(path, invariant);
  }
}

inline void WriteU32LE(std::ostream& output, std::uint32_t value,
                       const std::string& path,
                       const std::string& invariant) {
  const std::uint8_t bytes[4] = {
      static_cast<std::uint8_t>(value),
      static_cast<std::uint8_t>(value >> 8),
      static_cast<std::uint8_t>(value >> 16),
      static_cast<std::uint8_t>(value >> 24),
  };
  WriteExact(output, bytes, sizeof(bytes), path, invariant);
}

inline void WriteU64LE(std::ostream& output, std::uint64_t value,
                       const std::string& path,
                       const std::string& invariant) {
  std::uint8_t bytes[8];
  for (std::size_t index = 0; index < sizeof(bytes); ++index) {
    bytes[index] = static_cast<std::uint8_t>(value >> (index * 8));
  }
  WriteExact(output, bytes, sizeof(bytes), path, invariant);
}

inline void WriteF32LE(std::ostream& output, float value,
                       const std::string& path,
                       const std::string& invariant) {
  static_assert(sizeof(float) == sizeof(std::uint32_t),
                "binary float I/O requires 32-bit float");
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  WriteU32LE(output, bits, path, invariant);
}

}  // namespace binary_io

#endif  // SRC_BINARY_IO_H_
