#ifndef INCLUDE_TENSOR_H_
#define INCLUDE_TENSOR_H_

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace device_buffer_detail {

// Returns the internal atomic live-allocation counter shared by all
// DeviceBuffer specializations and translation units in this process. The
// reference is implementation-owned, retained for program lifetime, and not
// exposed by the public DeviceBuffer interface.
inline std::atomic<std::size_t>& LiveAllocationCount() noexcept {
  static std::atomic<std::size_t> count{0};
  return count;
}

// Returns the process-wide monotonic count of successful DeviceBuffer
// allocations. Unlike the live count, releases never decrement this diagnostic
// event counter, so a transient allocate/free pair remains observable.
inline std::atomic<std::size_t>& SuccessfulAllocationEventCount() noexcept {
  static std::atomic<std::size_t> count{0};
  return count;
}

// Returns the process-wide monotonic count of CUDA memory-information query
// attempts made through QueryMemoryInfo. It is independent of allocation
// ownership and exists to verify constructor query contracts.
inline std::atomic<std::size_t>& MemoryInfoQueryCount() noexcept {
  static std::atomic<std::size_t> count{0};
  return count;
}

// Supplies one process-wide total order for host diagnostic events. Sequentially
// consistent increments let tests compare query and allocation-attempt tokens
// even when unrelated threads consume intervening sequence values.
inline std::atomic<std::size_t>& DiagnosticEventSequence() noexcept {
  static std::atomic<std::size_t> sequence{0};
  return sequence;
}

// Stores the most recent memory-query invocation token on the calling host
// thread. Per-thread storage prevents unrelated concurrent threads from
// overwriting the constructor thread's evidence.
inline std::atomic<std::size_t>& LastMemoryInfoQuerySequence() noexcept {
  static thread_local std::atomic<std::size_t> sequence{0};
  return sequence;
}

// Stores the most recent DeviceBuffer allocation-attempt token on the calling
// host thread, including attempts for which cudaMalloc subsequently fails.
inline std::atomic<std::size_t>& LastAllocationAttemptSequence() noexcept {
  static thread_local std::atomic<std::size_t> sequence{0};
  return sequence;
}

inline std::size_t NextDiagnosticEventSequence() noexcept {
  return DiagnosticEventSequence().fetch_add(1, std::memory_order_seq_cst) + 1;
}

inline void RecordMemoryInfoQueryInvocation() noexcept {
  LastMemoryInfoQuerySequence().store(NextDiagnosticEventSequence(),
                                      std::memory_order_seq_cst);
}

inline void RecordAllocationAttempt() noexcept {
  LastAllocationAttemptSequence().store(NextDiagnosticEventSequence(),
                                        std::memory_order_seq_cst);
}

// Records one successful allocation in the implementation-owned accounting.
// Relaxed ordering is sufficient because the counter conveys no ownership or
// memory visibility; it only records an independent diagnostic total.
inline void RecordAllocation() noexcept {
  LiveAllocationCount().fetch_add(1, std::memory_order_relaxed);
  SuccessfulAllocationEventCount().fetch_add(1, std::memory_order_relaxed);
}

// Records one ownership release in the implementation-owned accounting. It
// uses relaxed ordering for the independent diagnostic total and owns no data.
inline void RecordRelease() noexcept {
  LiveAllocationCount().fetch_sub(1, std::memory_order_relaxed);
}

// Records and performs one cudaMemGetInfo query. Relaxed ordering is sufficient
// because the monotonic count is diagnostic only and carries no memory or
// ownership synchronization. Output pointers follow cudaMemGetInfo's contract.
inline cudaError_t QueryMemoryInfo(std::size_t* free_bytes,
                                   std::size_t* total_bytes) noexcept {
  MemoryInfoQueryCount().fetch_add(1, std::memory_order_relaxed);
  RecordMemoryInfoQueryInvocation();
  return cudaMemGetInfo(free_bytes, total_bytes);
}

// Allocates exactly requested_bytes of device storage and returns unique
// ownership to the caller. requested_bytes must be nonzero. On CUDA failure,
// no pointer is retained and std::runtime_error reports requested, free, and
// total bytes plus allocation and memory-query error details. cudaMalloc keeps
// its CUDA Runtime-defined synchronization behavior.
inline void* Allocate(std::size_t requested_bytes) {
  void* pointer = nullptr;
  RecordAllocationAttempt();
  const cudaError_t allocation_result = cudaMalloc(&pointer, requested_bytes);
  if (allocation_result != cudaSuccess) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t memory_result =
        QueryMemoryInfo(&free_bytes, &total_bytes);
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
// Returns the process-wide number of currently owned cudaMalloc allocations
// made by all DeviceBuffer specializations. The relaxed load is race-free with
// concurrent allocation/free but is only a point-in-time diagnostic snapshot;
// it performs no CUDA call or ownership transfer. Production translation units
// do not receive this test-only declaration.
inline std::size_t DeviceBufferAllocationCountForTests() noexcept {
  return device_buffer_detail::LiveAllocationCount().load(
      std::memory_order_relaxed);
}

// Returns the number of successful DeviceBuffer allocation events since process
// start. The counter is monotonic (subject only to size_t wraparound), uses a
// relaxed atomic load, and detects transient allocations that the live-count
// observer cannot. This standalone test hook changes no template definition.
inline std::size_t DeviceBufferSuccessfulAllocationEventCountForTests()
    noexcept {
  return device_buffer_detail::SuccessfulAllocationEventCount().load(
      std::memory_order_relaxed);
}

// Returns the number of cudaMemGetInfo attempts made through the shared wrapper
// since process start. The relaxed snapshot performs no CUDA call and changes
// no DeviceBuffer or LeNet class definition.
inline std::size_t CudaMemoryInfoQueryCountForTests() noexcept {
  return device_buffer_detail::MemoryInfoQueryCount().load(
      std::memory_order_relaxed);
}

// Returns the calling host thread's latest memory-query invocation token. The
// token is recorded immediately before cudaMemGetInfo, regardless of its return
// value. A query-before-allocation assertion is meaningful when snapshots and a
// successfully completed constructor run on this same thread, its existing
// count checks prove one query/allocation, and no nested diagnostic operation is
// performed on that thread. Concurrent other threads can create sequence gaps
// but cannot overwrite this thread-local token. Wraparound is outside test scope.
inline std::size_t CudaMemoryInfoQueryInvocationSequenceForTests() noexcept {
  return device_buffer_detail::LastMemoryInfoQuerySequence().load(
      std::memory_order_seq_cst);
}

// Returns the calling host thread's latest DeviceBuffer allocation-attempt
// token, recorded immediately before cudaMalloc whether that call succeeds or
// fails. It has the same-thread/successful-constructor scope as the query
// token. A failed allocation may issue a later diagnostic memory query, so its
// token pair must not be interpreted as the successful preflight contract.
inline std::size_t DeviceBufferAllocationAttemptSequenceForTests() noexcept {
  return device_buffer_detail::LastAllocationAttemptSequence().load(
      std::memory_order_seq_cst);
}
#endif

#endif  // INCLUDE_TENSOR_H_
