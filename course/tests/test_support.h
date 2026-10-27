#ifndef COURSE_TESTS_TEST_SUPPORT_H_
#define COURSE_TESTS_TEST_SUPPORT_H_

// Self-contained harness for the course host-interface tests.
//
// It intentionally uses no third-party test library. Cases self-register with
// TEST_CASE, report by throwing, and the runner prints an explicit executed
// and failure count. Each run exclusively creates a fresh
// "<executable directory>/course_io_scratch.XXXXXX" tree, so fixtures never
// touch /train, /test, /gemini data, the repository data directories, or /tmp,
// and concurrent suites can never remove one another's files. The harness is
// POSIX-oriented, matching the deployment target.

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#include <process.h>
#define COURSE_TEST_HAS_SYMLINKS 0
#else
#include <unistd.h>
#define COURSE_TEST_HAS_SYMLINKS 1
#endif

namespace course_test_support {

// Test entry points take no arguments and report failure by throwing.
using TestFunction = void (*)();

// Pairs a stable case name with the entry point registered for it.
struct TestCase {
  const char* name;
  TestFunction function;
};

// Function-local static registry avoids cross-translation-unit init order.
inline std::vector<TestCase>& Tests() {
  static std::vector<TestCase> tests;
  return tests;
}

// Appends a case to the registry during static initialization.
class TestRegistration {
 public:
  TestRegistration(const char* name, TestFunction function) {
    Tests().push_back({name, function});
  }
};

// Carries formatted assertion diagnostics from a case to the runner.
class AssertionFailure : public std::runtime_error {
 public:
  explicit AssertionFailure(const std::string& message)
      : std::runtime_error(message) {}
};

// Formats failures as file:line: message, then throws AssertionFailure.
[[noreturn]] inline void Fail(const char* file, int line,
                              const std::string& message) {
  std::ostringstream output;
  output << file << ':' << line << ": " << message;
  throw AssertionFailure(output.str());
}

// Reduces an executable path to a bare, extension-free suite name.
inline std::string SuiteName(const char* executable) {
  std::string name(executable == nullptr ? "tests" : executable);
  const std::string::size_type separator = name.find_last_of("/\\");
  if (separator != std::string::npos) {
    name.erase(0, separator + 1);
  }
  if (name.size() > 4 && name.compare(name.size() - 4, 4, ".exe") == 0) {
    name.erase(name.size() - 4);
  }
  return name;
}

// Directory containing path, without a trailing separator.
inline std::string DirectoryOf(const std::string& path) {
  const std::string::size_type separator = path.find_last_of("/\\");
  if (separator == std::string::npos) {
    return ".";
  }
  if (separator == 0) {
    return path.substr(0, 1);
  }
  return path.substr(0, separator);
}

// lstat on POSIX so symlinks are inspected rather than followed; plain stat
// is sufficient on non-POSIX hosts where symlink fixtures never run.
inline int LStat(const char* path, struct stat* info) {
#if defined(_WIN32)
  return ::stat(path, info);
#else
  return ::lstat(path, info);
#endif
}

// Creates one directory, tolerating an existing directory of that name.
inline void MakeDirectory(const std::string& path) {
#if defined(_WIN32)
  const int status = ::_mkdir(path.c_str());
#else
  const int status = ::mkdir(path.c_str(), 0700);
#endif
  if (status != 0 && errno != EEXIST) {
    throw std::runtime_error("could not create directory " + path);
  }
}

// Removes one empty directory.
inline int RemoveDirectory(const std::string& path) {
#if defined(_WIN32)
  return ::_rmdir(path.c_str());
#else
  return ::rmdir(path.c_str());
#endif
}

// Recursively removes path without ever following a symlink. Missing paths are
// already clean, so ENOENT is not an error.
inline void RemoveTree(const std::string& path) {
  struct stat info;
  if (LStat(path.c_str(), &info) != 0) {
    if (errno == ENOENT) {
      return;
    }
    throw std::runtime_error("could not inspect scratch path " + path);
  }
#if COURSE_TEST_HAS_SYMLINKS
  if (S_ISLNK(info.st_mode)) {
    if (std::remove(path.c_str()) != 0) {
      throw std::runtime_error("could not remove symlink " + path);
    }
    return;
  }
#endif
  if (S_ISDIR(info.st_mode)) {
    DIR* directory = ::opendir(path.c_str());
    if (directory == nullptr) {
      throw std::runtime_error("could not open scratch directory " + path);
    }
    std::vector<std::string> names;
    while (struct dirent* entry = ::readdir(directory)) {
      const std::string name(entry->d_name);
      if (name != "." && name != "..") {
        names.push_back(name);
      }
    }
    ::closedir(directory);
    for (const std::string& name : names) {
      RemoveTree(path + "/" + name);
    }
    if (RemoveDirectory(path) != 0) {
      throw std::runtime_error("could not remove scratch directory " + path);
    }
    return;
  }
  if (std::remove(path.c_str()) != 0) {
    throw std::runtime_error("could not remove scratch file " + path);
  }
}

inline bool PathExists(const std::string& path) {
  struct stat info;
  return LStat(path.c_str(), &info) == 0;
}

// True only when path itself is a symbolic link.
inline bool IsSymlink(const std::string& path) {
#if COURSE_TEST_HAS_SYMLINKS
  struct stat info;
  if (::lstat(path.c_str(), &info) != 0) {
    throw std::runtime_error("could not inspect " + path);
  }
  return S_ISLNK(info.st_mode);
#else
  (void)path;
  return false;
#endif
}

// Creates link_path pointing at target_path. POSIX-only fixture support.
inline void CreateSymlink(const std::string& target_path,
                          const std::string& link_path) {
#if COURSE_TEST_HAS_SYMLINKS
  if (::symlink(target_path.c_str(), link_path.c_str()) != 0) {
    throw std::runtime_error("could not create symlink " + link_path);
  }
#else
  (void)target_path;
  (void)link_path;
  throw std::runtime_error("symlink fixtures are unsupported on this host");
#endif
}

// Creates link_path as a hard-link alias of target_path. POSIX-only fixture
// support, used to prove destinations are not mutated through aliases.
inline void CreateHardLink(const std::string& target_path,
                           const std::string& link_path) {
#if COURSE_TEST_HAS_SYMLINKS
  if (::link(target_path.c_str(), link_path.c_str()) != 0) {
    throw std::runtime_error("could not create hard link " + link_path);
  }
#else
  (void)target_path;
  (void)link_path;
  throw std::runtime_error("hard-link fixtures are unsupported on this host");
#endif
}

// Reads a whole file as raw bytes.
inline std::vector<unsigned char> ReadBytes(const std::string& path) {
  std::ifstream input(path.c_str(), std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not read " + path);
  }
  return std::vector<unsigned char>(std::istreambuf_iterator<char>(input),
                                    std::istreambuf_iterator<char>());
}

// Replaces a whole file with raw bytes, truncating any previous content.
inline void WriteBytes(const std::string& path,
                       const std::vector<unsigned char>& bytes) {
  std::ofstream output(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("could not write " + path);
  }
  if (!bytes.empty()) {
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
  }
  output.close();
  if (!output) {
    throw std::runtime_error("could not finish writing " + path);
  }
}

inline std::string ReadText(const std::string& path) {
  const std::vector<unsigned char> bytes = ReadBytes(path);
  return std::string(bytes.begin(), bytes.end());
}

inline void WriteText(const std::string& path, const std::string& text) {
  WriteBytes(path, std::vector<unsigned char>(text.begin(), text.end()));
}

// Writes host-order (little-endian on the target) raw FP32 fixture values.
inline void WriteFloats(const std::string& path,
                        const std::vector<float>& values) {
  std::vector<unsigned char> bytes(values.size() * sizeof(float));
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  WriteBytes(path, bytes);
}

// Writes host-order (little-endian on the target) raw int32 fixture values.
inline void WriteInt32s(const std::string& path,
                        const std::vector<std::int32_t>& values) {
  std::vector<unsigned char> bytes(values.size() * sizeof(std::int32_t));
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  WriteBytes(path, bytes);
}

// Compares two FP32 values by representation so signed zero, denormals, and
// every other bit pattern survive round-trips unchanged.
inline bool SameBits(float left, float right) {
  std::uint32_t left_bits = 0;
  std::uint32_t right_bits = 0;
  std::memcpy(&left_bits, &left, sizeof(left_bits));
  std::memcpy(&right_bits, &right, sizeof(right_bits));
  return left_bits == right_bits;
}

// IEEE 802.3 CRC32: reflected polynomial 0xedb88320, initial and final XOR
// 0xffffffff, i.e. the same value as Python zlib.crc32.
inline std::uint32_t Crc32(const std::vector<unsigned char>& bytes) {
  std::uint32_t crc = 0xffffffffu;
  for (const unsigned char byte : bytes) {
    crc ^= static_cast<std::uint32_t>(byte);
    for (int bit = 0; bit < 8; ++bit) {
      const std::uint32_t mask = 0u - (crc & 1u);
      crc = (crc >> 1) ^ (0xedb88320u & mask);
    }
  }
  return crc ^ 0xffffffffu;
}

// Formats a CRC as exactly eight lowercase hexadecimal digits.
inline std::string Hex8(std::uint32_t value) {
  std::ostringstream output;
  output << std::hex << std::nouppercase << std::setw(8) << std::setfill('0')
         << value;
  return output.str();
}

inline std::string ToUpper(std::string text) {
  for (char& character : text) {
    if (character >= 'a' && character <= 'f') {
      character = static_cast<char>(character - 'a' + 'A');
    }
  }
  return text;
}

inline void ReplaceAll(std::string* text, const std::string& from,
                       const std::string& to) {
  std::string::size_type position = 0;
  while ((position = text->find(from, position)) != std::string::npos) {
    text->replace(position, from.size(), to);
    position += to.size();
  }
}

// Root of all scratch trees, set by main to this run's exclusive root.
inline std::string& ScratchRootStorage() {
  static std::string root;
  return root;
}

inline std::string ScratchRoot() { return ScratchRootStorage(); }

inline void SetScratchRoot(const std::string& root) {
  ScratchRootStorage() = root;
}

// Exclusively creates a fresh per-run scratch root below parent. POSIX uses
// mkdtemp, which fails rather than reusing an existing path; the Windows
// fallback retries unique pid/sequence names and never deletes pre-existing
// directories. Callers may recursively clean only the returned root.
inline std::string CreateScratchRoot(const std::string& parent) {
#if !defined(_WIN32)
  std::string pattern = parent + "/course_io_scratch.XXXXXX";
  std::vector<char> buffer(pattern.begin(), pattern.end());
  buffer.push_back('\0');
  if (::mkdtemp(buffer.data()) == nullptr) {
    throw std::runtime_error("could not create unique scratch root under " +
                             parent);
  }
  return std::string(buffer.data());
#else
  static std::atomic<unsigned long> sequence{0};
  const unsigned long process = static_cast<unsigned long>(::_getpid());
  for (int attempt = 0; attempt < 100; ++attempt) {
    std::ostringstream name;
    name << parent << "/course_io_scratch." << process << '.'
         << sequence.fetch_add(1);
    const std::string path = name.str();
    if (::_mkdir(path.c_str()) == 0) {
      return path;
    }
    if (errno != EEXIST) {
      throw std::runtime_error("could not create unique scratch root under " +
                               parent);
    }
  }
  throw std::runtime_error("could not create unique scratch root under " +
                           parent);
#endif
}

// Creates (or re-creates) a uniquely named directory below the build-owned
// scratch root and returns its path.
inline std::string MakeScratchDirectory(const std::string& name) {
  const std::string root = ScratchRoot();
  if (root.empty()) {
    throw std::runtime_error("scratch root was not configured");
  }
  const std::string path = root + "/" + name;
  RemoveTree(path);
  MakeDirectory(path);
  return path;
}

// Runs selected cases, counts executed cases and failures, and emits the suite
// result. A filter that selects nothing is a failure: the suite must never
// report success with zero executed cases.
inline int Run(const char* executable, const std::string& filter) {
  int executed = 0;
  int failures = 0;
  for (const TestCase& test : Tests()) {
    if (!filter.empty() &&
        std::string(test.name).find(filter) == std::string::npos) {
      continue;
    }
    ++executed;
    try {
      test.function();
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "test failure: " << test.name << ": " << error.what()
                << '\n';
    } catch (...) {
      ++failures;
      std::cerr << "test failure: " << test.name
                << ": non-standard exception\n";
    }
  }

  const bool passed = failures == 0 && executed > 0;
  std::cout << "event=test_suite name=" << SuiteName(executable)
            << " cases=" << Tests().size() << " executed=" << executed
            << " failures=" << failures
            << " status=" << (passed ? "pass" : "fail") << '\n';
  return passed ? 0 : 1;
}

}  // namespace course_test_support

// Token-pasting helpers give each generated symbol a unique name.
#define COURSE_TEST_JOIN_INNER(left, right) left##right
// Two-level indirection expands arguments before concatenation.
#define COURSE_TEST_JOIN(left, right) COURSE_TEST_JOIN_INNER(left, right)

// Declares, self-registers, and defines one named static test case.
#define TEST_CASE(name)                                                 \
  static void COURSE_TEST_JOIN(CourseTestFunction_, name)();            \
  static ::course_test_support::TestRegistration                        \
      COURSE_TEST_JOIN(CourseTestRegistration_, name)(                  \
          #name, &COURSE_TEST_JOIN(CourseTestFunction_, name));         \
  static void COURSE_TEST_JOIN(CourseTestFunction_, name)()

// Evaluates value exactly once and fails when it is false.
#define EXPECT_TRUE(value)                                                 \
  do {                                                                     \
    if (!(value)) {                                                        \
      ::course_test_support::Fail(__FILE__, __LINE__,                      \
                                  "EXPECT_TRUE(" #value ")");              \
    }                                                                      \
  } while (false)

// Evaluates value exactly once and fails when it is true.
#define EXPECT_FALSE(value)                                                \
  do {                                                                     \
    if ((value)) {                                                         \
      ::course_test_support::Fail(__FILE__, __LINE__,                      \
                                  "EXPECT_FALSE(" #value ")");             \
    }                                                                      \
  } while (false)

// Binds each operand exactly once, then checks equality.
#define EXPECT_EQ(expected, actual)                                       \
  do {                                                                    \
    const auto& course_test_expected = (expected);                        \
    const auto& course_test_actual = (actual);                            \
    if (!(course_test_expected == course_test_actual)) {                  \
      ::course_test_support::Fail(                                        \
          __FILE__, __LINE__, "EXPECT_EQ(" #expected ", " #actual ")");   \
    }                                                                     \
  } while (false)

// Requires the expression to fail through a validation/runtime exception. A
// bad_alloc is reported separately: rejecting a malformed declaration must be
// decided before large result storage is requested.
#define EXPECT_THROWS(expression)                                         \
  do {                                                                    \
    bool course_test_threw = false;                                       \
    try {                                                                 \
      expression;                                                         \
    } catch (const std::bad_alloc&) {                                     \
      ::course_test_support::Fail(                                        \
          __FILE__, __LINE__,                                             \
          "EXPECT_THROWS(" #expression                                     \
          ") threw std::bad_alloc instead of rejecting the input");        \
    } catch (const std::exception&) {                                     \
      course_test_threw = true;                                           \
    } catch (...) {                                                       \
      ::course_test_support::Fail(__FILE__, __LINE__,                     \
                                  "expression threw a non-standard "       \
                                  "exception");                           \
    }                                                                     \
    if (!course_test_threw) {                                             \
      ::course_test_support::Fail(__FILE__, __LINE__,                     \
                                  "EXPECT_THROWS(" #expression            \
                                  ") did not throw");                     \
    }                                                                     \
  } while (false)

// Requires the expression to throw exactly the named standard exception type,
// pinning usage errors to std::invalid_argument and corrupt-data errors to
// std::runtime_error so the eventual CLI can map them to exit 2 versus 1.
#define EXPECT_THROWS_AS(expression, exception_type)                      \
  do {                                                                    \
    bool course_test_expected_thrown = false;                             \
    try {                                                                 \
      expression;                                                         \
    } catch (const exception_type&) {                                     \
      course_test_expected_thrown = true;                                 \
    } catch (const std::exception&) {                                     \
      ::course_test_support::Fail(                                        \
          __FILE__, __LINE__,                                             \
          "EXPECT_THROWS_AS(" #expression ", " #exception_type            \
          ") threw a different std::exception type");                    \
    } catch (...) {                                                       \
      ::course_test_support::Fail(__FILE__, __LINE__,                     \
                                  "EXPECT_THROWS_AS(" #expression         \
                                  ") threw a non-standard exception");    \
    }                                                                     \
    if (!course_test_expected_thrown) {                                   \
      ::course_test_support::Fail(                                        \
          __FILE__, __LINE__,                                             \
          "EXPECT_THROWS_AS(" #expression ", " #exception_type            \
          ") did not throw");                                             \
    }                                                                     \
  } while (false)

// Parses the optional --filter argument, exclusively creates a fresh
// build-owned scratch root for this run, dispatches the suite, and removes
// only that root again.
int main(int argc, char** argv) {
  std::string filter;
  if (argc == 3 && std::string(argv[1]) == "--filter") {
    filter = argv[2];
  } else if (argc != 1) {
    std::cerr << "usage: " << argv[0] << " [--filter text]\n";
    std::cout << "event=test_suite name="
              << course_test_support::SuiteName(argv[0])
              << " cases=" << course_test_support::Tests().size()
              << " executed=0 failures=0 status=fail\n";
    return 1;
  }

  std::string executable_directory =
      course_test_support::DirectoryOf(argv[0] == nullptr ? "tests" : argv[0]);
#if !defined(_WIN32)
  char working_directory[4096];
  if (!executable_directory.empty() && executable_directory[0] != '/' &&
      ::getcwd(working_directory, sizeof(working_directory)) != nullptr) {
    executable_directory =
        std::string(working_directory) + "/" + executable_directory;
  }
#endif

  std::string scratch_root;
  bool ran = false;
  int status = 1;
  try {
    scratch_root =
        course_test_support::CreateScratchRoot(executable_directory);
    course_test_support::SetScratchRoot(scratch_root);
    status = course_test_support::Run(argv[0], filter);
    ran = true;
  } catch (const std::exception& error) {
    std::cerr << "test harness error: " << error.what() << '\n';
  }

  // Cleanup is limited to the root this process exclusively created; a failed
  // cleanup is a suite failure, never a reason to remove anything else.
  if (!scratch_root.empty()) {
    try {
      course_test_support::RemoveTree(scratch_root);
    } catch (const std::exception& error) {
      std::cerr << "test harness cleanup error: " << error.what() << '\n';
      status = 1;
    }
  }

  if (!ran) {
    std::cout << "event=test_suite name="
              << course_test_support::SuiteName(argv[0])
              << " cases=" << course_test_support::Tests().size()
              << " executed=0 failures=1 status=fail\n";
    return 1;
  }
  return status;
}

#endif  // COURSE_TESTS_TEST_SUPPORT_H_
