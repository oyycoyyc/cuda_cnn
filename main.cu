// Official course entry point. Exactly one of COURSE_TRAIN or COURSE_INFER is
// defined by the selected CMake target, so each executable contains only its
// own parsing and dispatch path.
#include "course/interface.h"

#if defined(COURSE_TRAIN)
#include "course/training.h"
#elif defined(COURSE_INFER)
#include "course/runtime.h"
#else
#error "define exactly one of COURSE_TRAIN or COURSE_INFER"
#endif

#if defined(COURSE_TRAIN) && defined(COURSE_INFER)
#error "COURSE_TRAIN and COURSE_INFER are mutually exclusive"
#endif

#include <cstdlib>
#include <exception>
#include <iostream>
#include <locale>
#include <stdexcept>

namespace {

// Dispatches the one workflow selected at compile time. Usage errors are
// std::invalid_argument (exit 2); file/data/CUDA failures are
// std::runtime_error (exit 1).
int RunSelectedWorkflow(int argc, char** argv) {
#if defined(COURSE_TRAIN)
  const char* const input_mode = std::getenv("COURSE_INPUT_MODE");
  const char* const epochs = std::getenv("COURSE_EPOCHS");
  const course::TrainOptions options = course::ParseTrainArgs(
      argc, const_cast<const char* const*>(argv), input_mode, epochs);
  return course::RunCourseTrain(options, std::cerr);
#else
  const char* const model_dir = std::getenv("COURSE_MODEL_DIR");
  const char* const batch = std::getenv("COURSE_INFER_BATCH");
  const char* const profile = std::getenv("COURSE_PROFILE");
  const course::InferOptions options = course::ParseInferArgs(
      argc, const_cast<const char* const*>(argv), model_dir, batch, profile);
  return course::RunCourseInfer(options, std::cout, std::cerr);
#endif
}

}  // namespace

int main(int argc, char** argv) {
  // Deterministic decimal formatting for diagnostics and predictions regardless
  // of the caller's global locale; runtime record formatting is independently
  // classic-locale buffered.
  std::cout.imbue(std::locale::classic());
  std::cerr.imbue(std::locale::classic());
  try {
    return RunSelectedWorkflow(argc, argv);
  } catch (const std::invalid_argument& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "error: unknown failure\n";
    return 1;
  }
}
