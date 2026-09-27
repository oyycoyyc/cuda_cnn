CXX ?= g++
NVCC ?= nvcc

CUDA_ARCH ?= sm_90
CUDA_COMPUTE := compute_$(patsubst sm_%,%,$(CUDA_ARCH))
CXXFLAGS := -std=c++14 -O2 -Wall -Wextra -Wpedantic
NVCCFLAGS := -std=c++14 -O2 -lineinfo \
  -gencode=arch=$(CUDA_COMPUTE),code=$(CUDA_ARCH) \
  -gencode=arch=$(CUDA_COMPUTE),code=$(CUDA_COMPUTE)
CPPFLAGS := -Iinclude -Itests

BUILD_DIR := build
HOST_OBJ_DIR := $(BUILD_DIR)/obj/host
CUDA_OBJ_DIR := $(BUILD_DIR)/obj/cuda

ifeq ($(OS),Windows_NT)
EXEEXT := .exe
endif

HOST_TEST_SOURCES := $(wildcard tests/*_tests.cpp)
CUDA_TEST_SOURCES := $(wildcard tests/*_tests.cu)
HOST_TEST_NAMES := $(basename $(notdir $(HOST_TEST_SOURCES)))
CUDA_TEST_NAMES := $(basename $(notdir $(CUDA_TEST_SOURCES)))
HOST_TEST_OBJECTS := $(addprefix $(HOST_OBJ_DIR)/,$(addsuffix .o,$(HOST_TEST_NAMES)))
CUDA_TEST_OBJECTS := $(addprefix $(CUDA_OBJ_DIR)/,$(addsuffix .o,$(CUDA_TEST_NAMES)))
HOST_TEST_BINS := $(addprefix $(BUILD_DIR)/,$(addsuffix $(EXEEXT),$(HOST_TEST_NAMES)))
CUDA_TEST_BINS := $(addprefix $(BUILD_DIR)/,$(addsuffix $(EXEEXT),$(CUDA_TEST_NAMES)))
DEPENDENCIES := $(HOST_TEST_OBJECTS:.o=.d) $(CUDA_TEST_OBJECTS:.o=.d)

ifeq ($(V),1)
Q :=
else
Q := @
endif

.PHONY: host-tests cuda-tests test check acceptance clean
.SECONDARY: $(HOST_TEST_OBJECTS) $(CUDA_TEST_OBJECTS)

host-tests: $(HOST_TEST_BINS)
	$(Q)set -e; for test in $(HOST_TEST_BINS); do "$$test"; done

cuda-tests: $(CUDA_TEST_BINS)
	$(Q)set -e; for test in $(CUDA_TEST_BINS); do "$$test"; done

test: host-tests cuda-tests

check: test

acceptance: check

$(BUILD_DIR)/%$(EXEEXT): $(HOST_OBJ_DIR)/%.o | $(BUILD_DIR)
	$(Q)$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/%$(EXEEXT): $(CUDA_OBJ_DIR)/%.o | $(BUILD_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) $< -o $@

$(HOST_OBJ_DIR)/%.o: tests/%.cpp | $(HOST_OBJ_DIR)
	$(Q)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(CUDA_OBJ_DIR)/%.o: tests/%.cu | $(CUDA_OBJ_DIR)
	$(Q)$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR) $(HOST_OBJ_DIR) $(CUDA_OBJ_DIR):
	$(Q)mkdir -p $@

clean:
	$(Q)rm -rf $(BUILD_DIR)

-include $(DEPENDENCIES)
