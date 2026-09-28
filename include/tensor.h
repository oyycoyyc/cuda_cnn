#ifndef INCLUDE_TENSOR_H_
#define INCLUDE_TENSOR_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace device_buffer_detail {

#ifdef CUDA_LENET_ENABLE_TEST_HOOKS
// Returns the test-only internal live-allocation counter shared by DeviceBuffer
// specializations in this process. The reference is implementation-owned and
// retained for program lifetime.
inline std::size_t& LiveAllocationCount() noexcept {
  static std::size_t count = 0;
  return count;
}
#endif

// Records one successful allocation for test instrumentation when enabled.
// It owns no storage, retains no caller data, and cannot fail or synchronize.
inline void RecordAllocation() noexcept {
#ifdef CUDA_LENET_ENABLE_TEST_HOOKS
  ++LiveAllocationCount();
#endif
}

// Records one ownership release for test instrumentation when enabled. It
// owns no storage, retains no caller data, and cannot fail or synchronize.
inline void RecordRelease() noexcept {
#ifdef CUDA_LENET_ENABLE_TEST_HOOKS
  --LiveAllocationCount();
#endif
}

// Allocates exactly requested_bytes of device storage and returns unique
// ownership to the caller. requested_bytes must be nonzero. On CUDA failure,
// no pointer is retained and std::runtime_error reports requested, free, and
// total bytes plus allocation and memory-query error details. cudaMalloc keeps
// its CUDA Runtime-defined synchronization behavior.
inline void* Allocate(std::size_t requested_bytes) {
  void* pointer = nullptr;
  const cudaError_t allocation_result = cudaMalloc(&pointer, requested_bytes);
  if (allocation_result != cudaSuccess) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t memory_result = cudaMemGetInfo(&free_bytes, &total_bytes);
    std::ostringstream message;
    message << "CUDA allocation failed: requested_bytes=" << requested_bytes
            << " free_bytes=" << free_bytes << " total_bytes=" << total_bytes
            << " code=" << static_cast<int>(allocation_result)
            << " text=" << cudaGetErrorString(allocation_result);
    if (memory_result != cudaSuccess) {
      message << " memory_query_code=" << static_cast<int>(memory_result)
              << " memory_query_text=" << cudaGetErrorString(memory_result);
    }
    throw std::runtime_error(message.str());
  }
  RecordAllocation();
  return pointer;
}

// Releases a non-null uniquely owned device pointer and updates the internal
// live count. The pointer becomes unusable even if cudaFree reports an error;
// errors cannot escape this noexcept cleanup. No explicit synchronization is
// added around cudaFree, whose CUDA Runtime-defined behavior still applies.
inline void Free(void* pointer) noexcept {
  if (pointer == nullptr) {
    return;
  }
  static_cast<void>(cudaFree(pointer));
  RecordRelease();
}

}  // namespace device_buffer_detail

// Owns count contiguous device elements allocated with cudaMalloc. Ownership
// is unique and movable but not copyable; get() remains valid until move,
// destruction, or replacement. A zero count owns no allocation. T must be a
// complete object type, allocation may throw std::runtime_error with requested
// and device-memory details, and destruction never throws. cudaMalloc/cudaFree
// retain their CUDA Runtime-defined synchronization behavior.
template <typename T>
class DeviceBuffer {
 public:
  // Constructs an empty nonallocating buffer. No CUDA call is made.
  DeviceBuffer() noexcept;

  // Allocates count device elements, or remains empty when count is zero. The
  // caller owns this object; allocation errors throw std::runtime_error and no
  // device pointer is retained on failure.
  explicit DeviceBuffer(std::size_t count);

  // Releases owned device storage with cudaFree. CUDA free errors are ignored
  // because destruction is noexcept; no pointer remains usable afterward.
  ~DeviceBuffer() noexcept;

  // Copy construction is forbidden because device allocation ownership is
  // unique; callers must explicitly move the source object instead.
  DeviceBuffer(const DeviceBuffer&) = delete;

  // Copy assignment is forbidden because device allocation ownership is
  // unique; callers must explicitly move the source object instead.
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // Transfers pointer and count from other, leaving other empty. Existing
  // references to the allocation remain valid but are now owned by *this.
  DeviceBuffer(DeviceBuffer&& other) noexcept;

  // Releases current storage, then transfers ownership from other and leaves
  // other empty. Self-move is harmless and no CUDA errors escape.
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  // Returns the owned device pointer, or nullptr for an empty/moved-from
  // buffer. The caller receives no ownership and must not free it.
  T* get() noexcept;

  // Returns the owned device pointer through a const owner, or nullptr when
  // empty. The pointee remains device-mutable; the caller must not free it.
  const T* get() const noexcept;

  // Returns the number of T elements associated with the allocation; this is
  // zero for empty and moved-from buffers and cannot fail.
  std::size_t size() const noexcept;

#ifdef CUDA_LENET_ENABLE_TEST_HOOKS
  // Returns the process-local number of currently owned cudaMalloc
  // allocations made by DeviceBuffer in this translation unit. This test-only,
  // single-host-thread observer performs no CUDA call or ownership transfer.
  static std::size_t AllocationCountForTests() noexcept;
#endif

 private:
  T* data_;
  std::size_t count_;
};

template <typename T>
DeviceBuffer<T>::DeviceBuffer() noexcept : data_(nullptr), count_(0) {}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(std::size_t count) : data_(nullptr), count_(0) {
  if (count == 0) {
    return;
  }
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::overflow_error("DeviceBuffer byte count overflow");
  }
  data_ = static_cast<T*>(device_buffer_detail::Allocate(count * sizeof(T)));
  count_ = count;
}

template <typename T>
DeviceBuffer<T>::~DeviceBuffer() noexcept {
  device_buffer_detail::Free(data_);
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(other.data_), count_(other.count_) {
  other.data_ = nullptr;
  other.count_ = 0;
}

template <typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    device_buffer_detail::Free(data_);
    data_ = other.data_;
    count_ = other.count_;
    other.data_ = nullptr;
    other.count_ = 0;
  }
  return *this;
}

template <typename T>
T* DeviceBuffer<T>::get() noexcept {
  return data_;
}

template <typename T>
const T* DeviceBuffer<T>::get() const noexcept {
  return data_;
}

template <typename T>
std::size_t DeviceBuffer<T>::size() const noexcept {
  return count_;
}

#ifdef CUDA_LENET_ENABLE_TEST_HOOKS
template <typename T>
std::size_t DeviceBuffer<T>::AllocationCountForTests() noexcept {
  return device_buffer_detail::LiveAllocationCount();
}
#endif

#endif  // INCLUDE_TENSOR_H_
