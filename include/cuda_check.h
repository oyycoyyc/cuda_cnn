#ifndef INCLUDE_CUDA_CHECK_H_
#define INCLUDE_CUDA_CHECK_H_

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace cuda_support {

// Checks one CUDA Runtime result and returns normally on cudaSuccess. On
// failure it throws std::runtime_error containing expression, file, line,
// numeric CUDA code, and CUDA error text. String inputs must remain valid only
// for this call; the function retains no pointers and performs no implicit
// device synchronization.
inline void CheckCuda(cudaError_t result, const char* expression,
                      const char* file, int line) {
  if (result == cudaSuccess) {
    return;
  }
  std::ostringstream message;
  message << "CUDA error: expression=" << expression << " file=" << file
          << " line=" << line << " code=" << static_cast<int>(result)
          << " text=" << cudaGetErrorString(result);
  throw std::runtime_error(message.str());
}

}  // namespace cuda_support

// Checks a CUDA Runtime expression without changing its evaluation count.
#define CUDA_CHECK(expression) \
  ::cuda_support::CheckCuda((expression), #expression, __FILE__, __LINE__)

// Checks and clears the calling host thread's pending kernel-launch error via
// cudaGetLastError; it does not synchronize the stream or device.
#define CUDA_KERNEL_CHECK() \
  ::cuda_support::CheckCuda(cudaGetLastError(), "cudaGetLastError()", __FILE__, \
                            __LINE__)

#endif  // INCLUDE_CUDA_CHECK_H_
