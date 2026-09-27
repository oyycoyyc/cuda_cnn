#ifndef TESTS_TEST_HARNESS_H_
#define TESTS_TEST_HARNESS_H_

#include <cmath>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace test_harness {

using TestFunction = void (*)();

struct TestCase {
  const char* name;
  TestFunction function;
};

inline std::vector<TestCase>& Tests() {
  static std::vector<TestCase> tests;
  return tests;
}

class TestRegistration {
 public:
  TestRegistration(const char* name, TestFunction function) {
    Tests().push_back({name, function});
  }
};

class AssertionFailure : public std::runtime_error {
 public:
  explicit AssertionFailure(const std::string& message)
      : std::runtime_error(message) {}
};

[[noreturn]] inline void Fail(const char* file, int line,
                              const std::string& message) {
  std::ostringstream output;
  output << file << ':' << line << ": " << message;
  throw AssertionFailure(output.str());
}

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

inline int Run(const char* executable, const std::string& filter) {
  int failures = 0;
  for (const TestCase& test : Tests()) {
    if (!filter.empty() &&
        std::string(test.name).find(filter) == std::string::npos) {
      continue;
    }

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

  std::cout << "event=test_suite name=" << SuiteName(executable)
            << " status=" << (failures == 0 ? "pass" : "fail") << '\n';
  return failures == 0 ? 0 : 1;
}

}  // namespace test_harness

#define TEST_HARNESS_JOIN_INNER(left, right) left##right
#define TEST_HARNESS_JOIN(left, right) TEST_HARNESS_JOIN_INNER(left, right)

#define TEST_CASE(name)                                                    \
  static void TEST_HARNESS_JOIN(TestFunction_, name)();                   \
  static ::test_harness::TestRegistration                                 \
      TEST_HARNESS_JOIN(TestRegistration_, name)(                         \
          #name, &TEST_HARNESS_JOIN(TestFunction_, name));                \
  static void TEST_HARNESS_JOIN(TestFunction_, name)()

#define EXPECT_TRUE(value)                                                   \
  do {                                                                       \
    if (!(value)) {                                                          \
      ::test_harness::Fail(__FILE__, __LINE__, "EXPECT_TRUE(" #value ")"); \
    }                                                                        \
  } while (false)

#define EXPECT_EQ(expected, actual)                                    \
  do {                                                                 \
    const auto& test_expected = (expected);                            \
    const auto& test_actual = (actual);                                \
    if (!(test_expected == test_actual)) {                             \
      ::test_harness::Fail(                                            \
          __FILE__, __LINE__, "EXPECT_EQ(" #expected ", " #actual ")"); \
    }                                                                  \
  } while (false)

#define EXPECT_NEAR(expected, actual, tolerance)                         \
  do {                                                                  \
    const auto test_expected = (expected);                              \
    const auto test_actual = (actual);                                  \
    const auto test_tolerance = (tolerance);                            \
    if (!(std::fabs(static_cast<double>(test_expected) -                \
                     static_cast<double>(test_actual)) <=               \
          static_cast<double>(test_tolerance))) {                       \
      ::test_harness::Fail(                                             \
          __FILE__, __LINE__,                                           \
          "EXPECT_NEAR(" #expected ", " #actual ", " #tolerance ")"); \
    }                                                                   \
  } while (false)

#define EXPECT_THROW_CONTAINS(expression, text)                            \
  do {                                                                     \
    bool test_threw = false;                                               \
    const std::string test_text(text);                                     \
    try {                                                                  \
      expression;                                                          \
    } catch (const std::exception& test_error) {                           \
      test_threw = true;                                                   \
      if (std::string(test_error.what()).find(test_text) ==                \
          std::string::npos) {                                             \
        ::test_harness::Fail(__FILE__, __LINE__,                           \
                             "exception did not contain expected text");  \
      }                                                                    \
    } catch (...) {                                                        \
      ::test_harness::Fail(__FILE__, __LINE__,                             \
                           "expression threw a non-standard exception");  \
    }                                                                      \
    if (!test_threw) {                                                     \
      ::test_harness::Fail(__FILE__, __LINE__, "expression did not throw"); \
    }                                                                      \
  } while (false)

int main(int argc, char** argv) {
  std::string filter;
  if (argc == 3 && std::string(argv[1]) == "--filter") {
    filter = argv[2];
  } else if (argc != 1) {
    std::cerr << "usage: " << argv[0] << " [--filter text]\n";
    std::cout << "event=test_suite name="
              << test_harness::SuiteName(argv[0]) << " status=fail\n";
    return 1;
  }
  return test_harness::Run(argv[0], filter);
}

#endif  // TESTS_TEST_HARNESS_H_
