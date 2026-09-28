CXX ?= g++
NVCC ?= nvcc

CUDA_ARCH ?= sm_90
CUDA_COMPUTE := compute_$(patsubst sm_%,%,$(CUDA_ARCH))
CXXFLAGS := -std=c++14 -O2 -Wall -Wextra -Wpedantic
NVCCFLAGS := -std=c++14 -O2 -lineinfo \
  -gencode=arch=$(CUDA_COMPUTE),code=$(CUDA_ARCH) \
  -gencode=arch=$(CUDA_COMPUTE),code=$(CUDA_COMPUTE)
CPPFLAGS := -Iinclude -Isrc -Itests

BUILD_DIR := build
HOST_OBJECT_DIR := $(BUILD_DIR)/obj/host
CUDA_OBJECT_DIR := $(BUILD_DIR)/obj/cuda

ifeq ($(OS),Windows_NT)
EXEEXT := .exe
endif

HOST_TEST_SOURCES := $(wildcard tests/*_tests.cpp)
CUDA_TEST_SOURCES := $(wildcard tests/*_tests.cu)
HOST_TEST_NAMES := $(basename $(notdir $(HOST_TEST_SOURCES)))
CUDA_TEST_NAMES := $(basename $(notdir $(CUDA_TEST_SOURCES)))
COLLIDING_TEST_NAMES := $(filter $(HOST_TEST_NAMES),$(CUDA_TEST_NAMES))
HOST_UNIQUE_NAMES := $(filter-out $(COLLIDING_TEST_NAMES),$(HOST_TEST_NAMES))
CUDA_UNIQUE_NAMES := $(filter-out $(COLLIDING_TEST_NAMES),$(CUDA_TEST_NAMES))

HOST_TEST_OBJECTS := $(addprefix $(HOST_OBJECT_DIR)/,$(addsuffix .o,$(HOST_TEST_NAMES)))
CUDA_TEST_OBJECTS := $(addprefix $(CUDA_OBJECT_DIR)/,$(addsuffix .o,$(CUDA_TEST_NAMES)))
CUDA_KERNEL_SOURCES := src/kernels/input.cu src/kernels/activation.cu \
  src/kernels/pooling.cu src/kernels/linear.cu src/kernels/convolution.cu
CUDA_KERNEL_OBJECTS := $(patsubst src/kernels/%.cu,$(CUDA_OBJECT_DIR)/kernels/%.o,$(CUDA_KERNEL_SOURCES))
DATASET_PROBE_OBJECT := $(HOST_OBJECT_DIR)/dataset_probe.o
DATASET_OBJECT := $(HOST_OBJECT_DIR)/dataset.o
RANDOM_OBJECT := $(HOST_OBJECT_DIR)/random.o
PARAMETERS_OBJECT := $(HOST_OBJECT_DIR)/parameters.o
CHECKPOINT_OBJECT := $(HOST_OBJECT_DIR)/checkpoint.o
CPU_REFERENCE_OBJECT := $(HOST_OBJECT_DIR)/cpu_reference.o
HOST_TEST_PROGRAMS := $(strip \
  $(addprefix $(BUILD_DIR)/,$(addsuffix $(EXEEXT),$(HOST_UNIQUE_NAMES))) \
  $(addprefix $(BUILD_DIR)/host/,$(addsuffix $(EXEEXT),$(COLLIDING_TEST_NAMES))))
CUDA_TEST_PROGRAMS := $(strip \
  $(addprefix $(BUILD_DIR)/,$(addsuffix $(EXEEXT),$(CUDA_UNIQUE_NAMES))) \
  $(addprefix $(BUILD_DIR)/cuda/,$(addsuffix $(EXEEXT),$(COLLIDING_TEST_NAMES))))
DEPENDENCY_FILES := $(HOST_TEST_OBJECTS:.o=.d) $(CUDA_TEST_OBJECTS:.o=.d) \
  $(DATASET_PROBE_OBJECT:.o=.d) $(DATASET_OBJECT:.o=.d) \
  $(RANDOM_OBJECT:.o=.d) $(PARAMETERS_OBJECT:.o=.d) \
  $(CHECKPOINT_OBJECT:.o=.d) $(CPU_REFERENCE_OBJECT:.o=.d) \
  $(CUDA_KERNEL_OBJECTS:.o=.d)

ifeq ($(V),1)
Q :=
else
Q := @
endif

.PHONY: host-tests cuda-tests test makefile-tests check acceptance clean
.SECONDARY: $(HOST_TEST_OBJECTS) $(CUDA_TEST_OBJECTS)

host-tests: $(HOST_TEST_PROGRAMS)
	$(Q)set -e; for test in $(HOST_TEST_PROGRAMS); do "$$test"; done

cuda-tests: $(CUDA_TEST_PROGRAMS)
	$(Q)set -e; for test in $(CUDA_TEST_PROGRAMS); do "$$test"; done

test: host-tests cuda-tests

makefile-tests:
	$(Q)sh tests/makefile_tests.sh

check: test makefile-tests

acceptance: check

ifneq ($(strip $(HOST_UNIQUE_NAMES)),)
$(addprefix $(BUILD_DIR)/,$(addsuffix $(EXEEXT),$(HOST_UNIQUE_NAMES))): \
    $(BUILD_DIR)/%$(EXEEXT): $(HOST_OBJECT_DIR)/%.o | $(BUILD_DIR)
	$(Q)$(CXX) $(CXXFLAGS) $^ -o $@
endif

$(BUILD_DIR)/dataset_tests$(EXEEXT): $(DATASET_OBJECT)
$(BUILD_DIR)/random_tests$(EXEEXT): $(RANDOM_OBJECT)
$(BUILD_DIR)/parameters_tests$(EXEEXT): $(PARAMETERS_OBJECT) $(RANDOM_OBJECT)
$(BUILD_DIR)/checkpoint_tests$(EXEEXT): $(CHECKPOINT_OBJECT) $(PARAMETERS_OBJECT)
$(BUILD_DIR)/cpu_reference_tests$(EXEEXT): $(CPU_REFERENCE_OBJECT)
$(BUILD_DIR)/operator_tests$(EXEEXT): $(CPU_REFERENCE_OBJECT) $(CUDA_KERNEL_OBJECTS)

$(BUILD_DIR)/dataset_probe$(EXEEXT): $(DATASET_PROBE_OBJECT) $(DATASET_OBJECT) | $(BUILD_DIR)
	$(Q)$(CXX) $(CXXFLAGS) $^ -o $@

ifneq ($(EXEEXT),)
$(BUILD_DIR)/dataset_tests: $(BUILD_DIR)/dataset_tests$(EXEEXT)
$(BUILD_DIR)/dataset_probe: $(BUILD_DIR)/dataset_probe$(EXEEXT)
$(BUILD_DIR)/random_tests: $(BUILD_DIR)/random_tests$(EXEEXT)
$(BUILD_DIR)/parameters_tests: $(BUILD_DIR)/parameters_tests$(EXEEXT)
$(BUILD_DIR)/checkpoint_tests: $(BUILD_DIR)/checkpoint_tests$(EXEEXT)
$(BUILD_DIR)/cpu_reference_tests: $(BUILD_DIR)/cpu_reference_tests$(EXEEXT)
$(BUILD_DIR)/operator_tests: $(BUILD_DIR)/operator_tests$(EXEEXT)
endif

ifneq ($(strip $(CUDA_UNIQUE_NAMES)),)
$(addprefix $(BUILD_DIR)/,$(addsuffix $(EXEEXT),$(CUDA_UNIQUE_NAMES))): \
    $(BUILD_DIR)/%$(EXEEXT): $(CUDA_OBJECT_DIR)/%.o | $(BUILD_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) $^ -o $@
endif

$(BUILD_DIR)/host/%$(EXEEXT): $(HOST_OBJECT_DIR)/%.o | $(BUILD_DIR)/host
	$(Q)$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/cuda/%$(EXEEXT): $(CUDA_OBJECT_DIR)/%.o | $(BUILD_DIR)/cuda
	$(Q)$(NVCC) $(NVCCFLAGS) $^ -o $@

$(HOST_OBJECT_DIR)/%.o: tests/%.cpp | $(HOST_OBJECT_DIR)
	$(Q)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(DATASET_OBJECT): src/dataset.cpp | $(HOST_OBJECT_DIR)
	$(Q)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(RANDOM_OBJECT): src/random.cpp | $(HOST_OBJECT_DIR)
	$(Q)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(PARAMETERS_OBJECT): src/parameters.cpp | $(HOST_OBJECT_DIR)
	$(Q)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(CHECKPOINT_OBJECT): src/checkpoint.cpp | $(HOST_OBJECT_DIR)
	$(Q)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(CPU_REFERENCE_OBJECT): tests/cpu_reference.cpp | $(HOST_OBJECT_DIR)
	$(Q)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(CUDA_OBJECT_DIR)/%.o: tests/%.cu | $(CUDA_OBJECT_DIR)
	$(Q)$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -MMD -MP -c $< -o $@

$(CUDA_OBJECT_DIR)/kernels/%.o: src/kernels/%.cu | $(CUDA_OBJECT_DIR)/kernels
	$(Q)$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR) $(BUILD_DIR)/host $(BUILD_DIR)/cuda \
    $(HOST_OBJECT_DIR) $(CUDA_OBJECT_DIR) $(CUDA_OBJECT_DIR)/kernels:
	$(Q)mkdir -p $@

clean:
	$(Q)rm -rf $(BUILD_DIR)

-include $(DEPENDENCY_FILES)
