cmake_minimum_required(VERSION 3.18)

# Pick the deployment architecture only when the caller has not chosen one.
# This must run before project() enables CUDA so compiler initialization cannot
# silently establish a different implicit default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

project(course_lenet LANGUAGES CXX CUDA)

# C++14 and CUDA C++14 are both required by the course contract.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Release configuration also uses O2 rather than CMake's default O3; the
# unconditional target option below covers other build types.
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG")

# Deterministic O2 optimization for host and device code. Fast-math and other
# value-changing math options are deliberately never enabled.
add_compile_options(-O2)

# Shared header roots follow the existing source layout.
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Applies strict host warnings without passing unknown flags to nvcc.
function(course_enable_host_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
endfunction()

# Canonical parameter schema and checkpoint format used by the course host.
add_library(course_parameters STATIC src/parameters.cpp)
course_enable_host_warnings(course_parameters)

add_library(course_checkpoint STATIC src/checkpoint.cpp)
# The checkpoint implementation calls the canonical schema/parameter helpers,
# so every consumer of this static archive must link them, including tools and
# executables that never reference parameter symbols directly.
target_link_libraries(course_checkpoint PUBLIC course_parameters)
course_enable_host_warnings(course_checkpoint)

# Host-side course interface and model-bundle I/O. No training workflow or CUDA
# code is part of this library.
add_library(course_host STATIC
  course/io.cpp
  course/model_io.cpp)
target_link_libraries(course_host PUBLIC course_parameters course_checkpoint)
course_enable_host_warnings(course_host)

# Existing original forward kernels only. The course forward library may reuse
# them but must never depend on the reference training backend.
add_library(course_reference_forward STATIC
  src/kernels/activation.cu
  src/kernels/pooling.cu
  src/kernels/linear.cu
  src/kernels/convolution.cu)

# Course GPU input/argmax helpers and the forward-only inference model.
add_library(course_forward STATIC
  course/gpu_ops.cu
  course/inference.cu)
target_link_libraries(course_forward PUBLIC
  course_reference_forward
  course_parameters)

# Original trainable LeNet and its training-only kernels. Production course
# training uses this reviewed backend; course_forward/inference never depends on
# it. The old src/train.cu workflow, CLI, dataset, and reporting sources are
# deliberately not part of this library.
add_library(course_reference_training STATIC
  src/lenet.cu
  src/kernels/input.cu
  src/kernels/loss.cu
  src/kernels/metrics.cu
  src/kernels/adam.cu)
target_link_libraries(course_reference_training PUBLIC
  course_reference_forward
  course_parameters)

# Course-only training workflow over supplied FP32/int32 arrays, plus the
# original deterministic split/shuffle helper it consumes.
add_library(course_training STATIC
  course/training.cu
  src/random.cpp)
target_link_libraries(course_training PUBLIC
  course_host
  course_forward
  course_reference_training)
course_enable_host_warnings(course_training)

# Optional tests and development tools. None of these sources are needed by the
# default submission build.
option(COURSE_BUILD_TESTS "Build course host/GPU tests and model bundle tool" OFF)

if(COURSE_BUILD_TESTS)
  enable_testing()

  add_executable(course_io_tests course/tests/io_tests.cpp)
  target_link_libraries(course_io_tests PRIVATE course_host)
  course_enable_host_warnings(course_io_tests)

  add_executable(course_gpu_tests course/tests/gpu_tests.cu)
  target_link_libraries(course_gpu_tests PRIVATE
    course_forward
    course_reference_training)

  add_executable(course_training_tests course/tests/training_tests.cu)
  target_link_libraries(course_training_tests PRIVATE course_training)

  add_executable(course_model_bundle course/tools/model_bundle.cpp)
  target_link_libraries(course_model_bundle PRIVATE course_host)
  course_enable_host_warnings(course_model_bundle)

  add_test(NAME course_io_tests COMMAND course_io_tests)
  add_test(NAME course_gpu_tests COMMAND course_gpu_tests)
  add_test(NAME course_training COMMAND course_training_tests)
  set_tests_properties(course_training PROPERTIES
    PASS_REGULAR_EXPRESSION "executed=9 failures=0 status=pass")
endif()
