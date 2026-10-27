#include "course/model_io.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

// Model bundle publication and loading. The checkpoint file uses the original
// serialization unchanged; course.meta is the small exact metadata file whose
// CRC32 binds it to those checkpoint bytes. Publication uses same-directory
// temporaries and renames the metadata last, so failures before publication
// leave previous entries intact and failures in between leave a mixed pair
// that the loader rejects.

namespace course {
namespace {

const char kCheckpointName[] = "lenet.bin";
const char kMetadataName[] = "course.meta";
const char kVersionLine[] = "course_lenet_v1";
const char kShapeLine[] = "shape=1,28,28,10";
const char kCrcPrefix[] = "checkpoint_crc32=";
const std::size_t kMaxMetadataBytes = 256;

[[noreturn]] void Fail(const std::string& path, const std::string& message) {
  throw std::runtime_error(path + ": " + message);
}

std::string Join(const std::string& directory, const char* name) {
  return directory + "/" + name;
}

const char* ModeName(InputMode input_mode) {
  switch (input_mode) {
    case InputMode::kUnit:
      return "unit";
    case InputMode::kByte:
      return "byte";
    case InputMode::kNormalized:
      return "normalized";
  }
  throw std::runtime_error("unsupported input mode for course model");
}

// Inspects lstat metadata so a destination symlink or hard-link alias is
// detected before any write; only regular single-link artifacts survive.
void InspectArtifact(const std::string& path) {
  struct stat info;
#if defined(_WIN32)
  if (::stat(path.c_str(), &info) != 0) {
    Fail(path, "could not inspect existing model artifact");
  }
#else
  if (::lstat(path.c_str(), &info) != 0) {
    Fail(path, "could not inspect existing model artifact");
  }
  if (S_ISLNK(info.st_mode)) {
    Fail(path, "refusing symlink model artifact");
  }
#endif
  if (!S_ISREG(info.st_mode)) {
    Fail(path, "existing model artifact must be a regular file");
  }
  if (info.st_nlink != 1) {
    Fail(path, "refusing hard-link model artifact alias");
  }
}

// Which reserved artifact names currently exist after type/alias validation.
struct ReservedArtifacts {
  bool checkpoint_exists;
  bool metadata_exists;
};

// Requires directory to exist and, when present, validates only the two
// reserved artifact names. Unrelated caller files and directories in the work
// directory are preserved untouched; course /work is not guaranteed empty.
ReservedArtifacts InspectReservedArtifacts(const std::string& directory) {
  struct stat info;
  if (::stat(directory.c_str(), &info) != 0) {
    Fail(directory, "could not inspect model directory");
  }
  if (!S_ISDIR(info.st_mode)) {
    Fail(directory, "model path is not a directory");
  }

  DIR* handle = ::opendir(directory.c_str());
  if (handle == nullptr) {
    Fail(directory, "could not open model directory");
  }
  std::vector<std::string> reserved_paths;
  ReservedArtifacts artifacts{false, false};
  while (struct dirent* entry = ::readdir(handle)) {
    const std::string name(entry->d_name);
    if (name == kCheckpointName) {
      artifacts.checkpoint_exists = true;
      reserved_paths.push_back(directory + "/" + name);
    } else if (name == kMetadataName) {
      artifacts.metadata_exists = true;
      reserved_paths.push_back(directory + "/" + name);
    }
  }
  ::closedir(handle);

  // Never follow a reserved-name alias: symlinks and shared hard links are
  // refused before any temporary or destination write.
  for (const std::string& path : reserved_paths) {
    InspectArtifact(path);
  }
  return artifacts;
}

// Creates a unique temporary name beside its destination, keeping publication
// on one filesystem.
std::string TemporaryPath(const std::string& destination) {
  static std::atomic<unsigned long> sequence{0};
#ifdef _WIN32
  const unsigned long process = static_cast<unsigned long>(::_getpid());
#else
  const unsigned long process = static_cast<unsigned long>(::getpid());
#endif
  std::ostringstream temporary;
  temporary << destination << ".tmp." << process << '.'
            << sequence.fetch_add(1);
  return temporary.str();
}

// Chooses a temporary name that does not currently exist, so an unrelated
// caller file is never truncated or replaced merely because its name matches
// a candidate.
std::string UniqueTemporaryPath(const std::string& destination) {
  for (int attempt = 0; attempt < 100; ++attempt) {
    const std::string path = TemporaryPath(destination);
    struct stat info;
#ifdef _WIN32
    const int status = ::stat(path.c_str(), &info);
#else
    const int status = ::lstat(path.c_str(), &info);
#endif
    if (status != 0) {
      if (errno == ENOENT) {
        return path;
      }
      Fail(path, "could not inspect candidate temporary path");
    }
  }
  Fail(destination, "could not choose a free temporary path");
}

// Atomically replaces destination with temporary on the same filesystem.
bool ReplaceFile(const std::string& temporary, const std::string& destination) {
#ifdef _WIN32
  return MoveFileExA(temporary.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
  return std::rename(temporary.c_str(), destination.c_str()) == 0;
#endif
}

// IEEE CRC32 (reflected polynomial 0xedb88320, initial/final XOR 0xffffffff),
// matching Python zlib.crc32, computed over the checkpoint file's exact bytes.
std::uint32_t Crc32File(const std::string& path) {
  std::ifstream input(path.c_str(), std::ios::binary);
  if (!input) {
    Fail(path, "could not open checkpoint bytes for CRC32");
  }
  std::uint32_t crc = 0xffffffffu;
  char buffer[65536];
  while (input) {
    input.read(buffer, static_cast<std::streamsize>(sizeof(buffer)));
    const std::streamsize count = input.gcount();
    for (std::streamsize index = 0; index < count; ++index) {
      crc ^= static_cast<unsigned char>(buffer[index]);
      for (int bit = 0; bit < 8; ++bit) {
        const std::uint32_t mask = 0u - (crc & 1u);
        crc = (crc >> 1) ^ (0xedb88320u & mask);
      }
    }
  }
  if (!input.eof()) {
    Fail(path, "could not read checkpoint bytes for CRC32");
  }
  return crc ^ 0xffffffffu;
}

std::string Hex8(std::uint32_t value) {
  std::ostringstream output;
  output << std::hex << std::nouppercase << value;
  const std::string digits = output.str();
  if (digits.size() >= 8) {
    return digits;
  }
  return std::string(8 - digits.size(), '0') + digits;
}

std::string BuildMetadata(const char* input_mode,
                          std::uint32_t checkpoint_crc32) {
  return std::string(kVersionLine) + "\ninput_mode=" + input_mode + "\n" +
         kShapeLine + "\n" + kCrcPrefix + Hex8(checkpoint_crc32) + "\n";
}

void WriteTextFile(const std::string& path, const std::string& text) {
  std::ofstream output(path.c_str(),
                       std::ios::binary | std::ios::out | std::ios::trunc);
  if (!output) {
    Fail(path, "could not open temporary metadata for writing");
  }
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  output.flush();
  if (!output) {
    Fail(path, "could not write temporary metadata");
  }
  output.close();
  if (output.fail()) {
    Fail(path, "could not close temporary metadata");
  }
}

// Reads course.meta only after confirming it is at most 256 bytes.
std::string ReadMetadataText(const std::string& path) {
  std::ifstream input(path.c_str(), std::ios::binary);
  if (!input) {
    Fail(path, "could not open course metadata for reading");
  }
  input.seekg(0, std::ios::end);
  const std::streamoff end = input.tellg();
  if (end < 0) {
    Fail(path, "could not determine course metadata size");
  }
  if (static_cast<std::uint64_t>(end) > kMaxMetadataBytes) {
    Fail(path, "course metadata must be at most 256 bytes");
  }
  input.seekg(0, std::ios::beg);
  if (!input) {
    Fail(path, "could not rewind course metadata");
  }
  std::string text(static_cast<std::size_t>(end), '\0');
  if (!text.empty()) {
    input.read(&text[0], static_cast<std::streamsize>(text.size()));
    if (!input ||
        static_cast<std::size_t>(input.gcount()) != text.size()) {
      Fail(path, "could not read course metadata");
    }
  }
  return text;
}

struct ParsedMetadata {
  InputMode input_mode;
  std::uint32_t checkpoint_crc32;
};

// Requires exactly four LF-terminated lines in the documented order, with no
// extra, unknown, duplicate, or reordered fields and eight lowercase hex CRC
// digits.
ParsedMetadata ParseMetadata(const std::string& text,
                             const std::string& path) {
  std::vector<std::string> lines;
  std::string current;
  for (char character : text) {
    if (character == '\n') {
      lines.push_back(current);
      current.clear();
    } else {
      current.push_back(character);
    }
  }
  if (!current.empty()) {
    Fail(path, "course metadata must end with a line feed");
  }
  if (lines.size() != 4) {
    Fail(path, "course metadata must contain exactly four lines");
  }
  if (lines[0] != kVersionLine) {
    Fail(path, "course metadata version line is invalid");
  }

  ParsedMetadata parsed;
  if (lines[1] == "input_mode=unit") {
    parsed.input_mode = InputMode::kUnit;
  } else if (lines[1] == "input_mode=byte") {
    parsed.input_mode = InputMode::kByte;
  } else if (lines[1] == "input_mode=normalized") {
    parsed.input_mode = InputMode::kNormalized;
  } else {
    Fail(path, "input_mode field is invalid");
  }

  if (lines[2] != kShapeLine) {
    Fail(path, "shape field is invalid");
  }

  const std::string prefix(kCrcPrefix);
  if (lines[3].size() != prefix.size() + 8 ||
      lines[3].compare(0, prefix.size(), prefix) != 0) {
    Fail(path, "checkpoint_crc32 field must have eight lowercase hex digits");
  }
  std::uint32_t crc = 0;
  for (std::size_t index = prefix.size(); index < lines[3].size(); ++index) {
    const char digit = lines[3][index];
    std::uint32_t value = 0;
    if (digit >= '0' && digit <= '9') {
      value = static_cast<std::uint32_t>(digit - '0');
    } else if (digit >= 'a' && digit <= 'f') {
      value = static_cast<std::uint32_t>(digit - 'a') + 10u;
    } else {
      Fail(path, "checkpoint_crc32 must use lowercase hexadecimal digits");
    }
    crc = crc * 16u + value;
  }
  parsed.checkpoint_crc32 = crc;
  return parsed;
}

// A destination directory is replaceable only when both reserved artifacts are
// absent, or when they form a valid existing pair: both regular single-link
// files, exact metadata, matching CRC over lenet.bin, and a loadable original
// checkpoint. Lone, junk, and mixed reserved files are refused before any
// temporary or destination write, while unrelated entries remain untouched.
void ValidateExistingBundle(const std::string& directory) {
  const ReservedArtifacts existing = InspectReservedArtifacts(directory);
  if (!existing.checkpoint_exists && !existing.metadata_exists) {
    return;
  }
  if (existing.checkpoint_exists != existing.metadata_exists) {
    Fail(directory,
         "reserved model artifacts must exist as a complete valid pair");
  }

  const std::string checkpoint_path = Join(directory, kCheckpointName);
  const std::string metadata_path = Join(directory, kMetadataName);
  const ParsedMetadata metadata =
      ParseMetadata(ReadMetadataText(metadata_path), metadata_path);
  if (Crc32File(checkpoint_path) != metadata.checkpoint_crc32) {
    Fail(metadata_path,
         "existing checkpoint_crc32 does not match the lenet.bin bytes");
  }
  // CRC binding alone cannot distinguish a junk checkpoint from a real one, so
  // the existing checkpoint must also satisfy the original loader invariants.
  (void)LoadCheckpoint(checkpoint_path);
}

}  // namespace

CourseModel LoadCourseModel(const std::string& directory) {
  const std::string metadata_path = Join(directory, kMetadataName);
  const ParsedMetadata metadata =
      ParseMetadata(ReadMetadataText(metadata_path), metadata_path);

  // The checkpoint loader independently validates the complete original
  // format before any bundle fields are trusted for inference.
  const std::string checkpoint_path = Join(directory, kCheckpointName);
  CourseModel model{LoadCheckpoint(checkpoint_path), metadata.input_mode};

  if (Crc32File(checkpoint_path) != metadata.checkpoint_crc32) {
    Fail(metadata_path,
         "checkpoint_crc32 does not match the lenet.bin bytes");
  }
  return model;
}

void SaveCourseModel(const std::string& directory,
                     const Checkpoint& checkpoint, InputMode input_mode) {
  const char* mode = ModeName(input_mode);
  ValidateExistingBundle(directory);

  const std::string checkpoint_path = Join(directory, kCheckpointName);
  const std::string metadata_path = Join(directory, kMetadataName);
  const std::string temporary_checkpoint =
      UniqueTemporaryPath(checkpoint_path);
  const std::string temporary_metadata = UniqueTemporaryPath(metadata_path);

  try {
    // SaveCheckpoint validates metadata, schema, and finiteness before it
    // writes anything, so an invalid checkpoint never reaches the directory.
    SaveCheckpoint(temporary_checkpoint, checkpoint);
    const std::string metadata =
        BuildMetadata(mode, Crc32File(temporary_checkpoint));
    if (metadata.size() > kMaxMetadataBytes) {
      Fail(metadata_path, "course metadata would exceed 256 bytes");
    }
    WriteTextFile(temporary_metadata, metadata);

    // Checkpoint first, metadata last: a failure between the two leaves a
    // mixed pair, which the loader rejects through the CRC.
    if (!ReplaceFile(temporary_checkpoint, checkpoint_path)) {
      Fail(checkpoint_path, "could not atomically publish checkpoint");
    }
    if (!ReplaceFile(temporary_metadata, metadata_path)) {
      Fail(metadata_path, "could not atomically publish metadata");
    }
  } catch (...) {
    std::remove(temporary_checkpoint.c_str());
    std::remove(temporary_metadata.c_str());
    throw;
  }
}

}  // namespace course
