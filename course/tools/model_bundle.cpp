#include "checkpoint.h"
#include "course/interface.h"
#include "course/model_io.h"

#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif

// Development-only utility: publishes an existing original checkpoint as a
// course model bundle (lenet.bin plus course.meta). Usage errors exit 2,
// runtime errors exit 1, success exits 0. All diagnostics go to stderr; a
// successful run writes nothing to stdout and never modifies the source
// checkpoint.

namespace {

void Fail(const std::string& path, const std::string& message) {
  throw std::runtime_error(path + ": " + message);
}

// Accepts a missing output path (creating it) or an existing empty directory,
// and refuses anything else.
void PrepareEmptyOutputDirectory(const std::string& path) {
  struct stat info;
  if (::stat(path.c_str(), &info) == 0) {
    if (!S_ISDIR(info.st_mode)) {
      Fail(path, "output path exists and is not a directory");
    }
    DIR* handle = ::opendir(path.c_str());
    if (handle == nullptr) {
      Fail(path, "could not open output directory");
    }
    bool empty = true;
    while (struct dirent* entry = ::readdir(handle)) {
      const std::string name(entry->d_name);
      if (name != "." && name != "..") {
        empty = false;
        break;
      }
    }
    ::closedir(handle);
    if (!empty) {
      Fail(path, "output directory must be empty");
    }
    return;
  }
  if (errno != ENOENT) {
    Fail(path, "could not inspect output directory");
  }
#ifdef _WIN32
  const int status = ::_mkdir(path.c_str());
#else
  const int status = ::mkdir(path.c_str(), 0755);
#endif
  if (status != 0) {
    Fail(path, "could not create output directory");
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: model_bundle checkpoint.bin output_dir input_mode\n";
    return 2;
  }

  try {
    const course::InputMode input_mode = course::ParseInputMode(argv[3]);
    const Checkpoint checkpoint = LoadCheckpoint(argv[1]);
    PrepareEmptyOutputDirectory(argv[2]);
    course::SaveCourseModel(argv[2], checkpoint, input_mode);
  } catch (const std::invalid_argument& error) {
    std::cerr << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
