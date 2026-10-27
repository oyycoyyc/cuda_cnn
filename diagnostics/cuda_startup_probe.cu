// Standalone startup diagnostic for the first explicit CUDA Runtime calls in
// main. Build with -DPROBE_HOST_ONLY for a host-only binary that supports only
// empty mode. Records call=1/2 per API; they are explicit call ordering, not
// proof that platform preload libraries did not call CUDA before main.
#include <chrono>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>

#ifndef PROBE_HOST_ONLY
#include <cuda_runtime.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kExitSuccess = 0;
constexpr int kExitRuntimeFailure = 1;
constexpr int kExitUsage = 2;

// Modes are parsed before any CUDA call is issued.
enum class Mode { kEmpty, kCountFirst, kSetFirst, kPropertiesFirst, kSequence };

#ifdef PROBE_HOST_ONLY
constexpr const char* kBuildName = "host";
#else
constexpr const char* kBuildName = "cuda";
#endif

// Returns the stable command-line spelling of a mode.
const char* ModeName(Mode mode) {
  switch (mode) {
    case Mode::kEmpty:
      return "empty";
    case Mode::kCountFirst:
      return "count-first";
    case Mode::kSetFirst:
      return "set-first";
    case Mode::kPropertiesFirst:
      return "properties-first";
    case Mode::kSequence:
      return "sequence";
  }
  return "unknown";
}

// Parses a known mode name without touching CUDA.
bool ParseMode(const std::string& name, Mode* mode) {
  if (name == "empty") {
    *mode = Mode::kEmpty;
  } else if (name == "count-first") {
    *mode = Mode::kCountFirst;
  } else if (name == "set-first") {
    *mode = Mode::kSetFirst;
  } else if (name == "properties-first") {
    *mode = Mode::kPropertiesFirst;
  } else if (name == "sequence") {
    *mode = Mode::kSequence;
  } else {
    return false;
  }
  return true;
}

// Returns the wall time between two steady-clock samples in milliseconds.
double ElapsedMilliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// Writes a concise failure record with no spaces inside field values.
void PrintErrorReason(Mode mode, const char* reason) {
  std::cerr << "event=error mode=" << ModeName(mode) << " reason=" << reason
            << '\n';
}

// Writes exactly one summary record just before a successful exit.
void PrintSummary(Mode mode, int api_calls, double main_ms) {
  std::ostringstream formatted;
  formatted.imbue(std::locale::classic());
  formatted << std::fixed << std::setprecision(6)
            << "event=summary mode=" << ModeName(mode)
            << " build=" << kBuildName << " api_calls=" << api_calls
            << " main_ms=" << main_ms << " status=pass\n";
  std::cout << formatted.str();
}

#ifndef PROBE_HOST_ONLY
// Writes one classic-locale record after the measured API call has returned.
void PrintApiRecord(Mode mode, const char* api, int call, double elapsed_ms,
                    int code, const std::string& extra_fields) {
  std::ostringstream formatted;
  formatted.imbue(std::locale::classic());
  formatted << std::fixed << std::setprecision(6)
            << "event=api mode=" << ModeName(mode) << " api=" << api
            << " call=" << call << " elapsed_ms=" << elapsed_ms
            << " code=" << code;
  if (!extra_fields.empty()) formatted << ' ' << extra_fields;
  formatted << '\n';
  std::cout << formatted.str();
}

// Reports a checked CUDA Runtime failure with its numeric error code.
void PrintApiError(Mode mode, const char* api, int call, int code) {
  std::cerr << "event=error mode=" << ModeName(mode) << " api=" << api
            << " call=" << call << " code=" << code << '\n';
}

// Measures one cudaGetDeviceCount call and requires a visible device.
bool CountStep(Mode mode, int call, int* api_calls) {
  int count = 0;
  const Clock::time_point start = Clock::now();
  const cudaError_t result = cudaGetDeviceCount(&count);
  const Clock::time_point end = Clock::now();
  ++*api_calls;
  if (result != cudaSuccess) {
    PrintApiError(mode, "cudaGetDeviceCount", call, static_cast<int>(result));
    return false;
  }
  PrintApiRecord(mode, "cudaGetDeviceCount", call,
                 ElapsedMilliseconds(start, end), static_cast<int>(result),
                 "count=" + std::to_string(count));
  if (count <= 0) {
    PrintErrorReason(mode, "no_visible_devices");
    return false;
  }
  return true;
}

// Measures one cudaSetDevice(0) call.
bool SetStep(Mode mode, int call, int* api_calls) {
  const Clock::time_point start = Clock::now();
  const cudaError_t result = cudaSetDevice(0);
  const Clock::time_point end = Clock::now();
  ++*api_calls;
  if (result != cudaSuccess) {
    PrintApiError(mode, "cudaSetDevice", call, static_cast<int>(result));
    return false;
  }
  PrintApiRecord(mode, "cudaSetDevice", call, ElapsedMilliseconds(start, end),
                 static_cast<int>(result), "device=0");
  return true;
}

// Measures one cudaGetDeviceProperties(0) call.
bool PropertiesStep(Mode mode, int call, int* api_calls) {
  cudaDeviceProp properties{};
  const Clock::time_point start = Clock::now();
  const cudaError_t result = cudaGetDeviceProperties(&properties, 0);
  const Clock::time_point end = Clock::now();
  ++*api_calls;
  if (result != cudaSuccess) {
    PrintApiError(mode, "cudaGetDeviceProperties", call,
                  static_cast<int>(result));
    return false;
  }
  PrintApiRecord(mode, "cudaGetDeviceProperties", call,
                 ElapsedMilliseconds(start, end), static_cast<int>(result),
                 "major=" + std::to_string(properties.major) +
                     " minor=" + std::to_string(properties.minor));
  return true;
}

// Runs the selected mode using only checked CUDA Runtime calls.
bool RunCudaMode(Mode mode, int* api_calls) {
  switch (mode) {
    case Mode::kEmpty:
      return true;
    case Mode::kCountFirst:
      return CountStep(mode, 1, api_calls) && CountStep(mode, 2, api_calls);
    case Mode::kSetFirst:
      return SetStep(mode, 1, api_calls) && SetStep(mode, 2, api_calls);
    case Mode::kPropertiesFirst:
      return PropertiesStep(mode, 1, api_calls) &&
             PropertiesStep(mode, 2, api_calls);
    case Mode::kSequence:
      for (int call = 1; call <= 2; ++call) {
        if (!CountStep(mode, call, api_calls) ||
            !SetStep(mode, call, api_calls) ||
            !PropertiesStep(mode, call, api_calls)) {
          return false;
        }
      }
      return true;
  }
  return false;
}

#endif
}  // namespace

int main(int argc, char** argv) {
  const Clock::time_point main_start = Clock::now();
  if (argc != 2) {
    std::cerr << "event=error reason=usage expected=MODE\n";
    return kExitUsage;
  }
  Mode mode;
  if (!ParseMode(argv[1], &mode)) {
    std::cerr << "event=error reason=unknown_mode\n";
    return kExitUsage;
  }
#ifdef PROBE_HOST_ONLY
  if (mode != Mode::kEmpty) {
    PrintErrorReason(mode, "host_build_unsupported");
    return kExitUsage;
  }
  const int api_calls = 0;
#else
  int api_calls = 0;
  if (!RunCudaMode(mode, &api_calls)) {
    return kExitRuntimeFailure;
  }
#endif
  const Clock::time_point main_end = Clock::now();
  PrintSummary(mode, api_calls, ElapsedMilliseconds(main_start, main_end));
  return kExitSuccess;
}
