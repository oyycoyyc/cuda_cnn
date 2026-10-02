CXX ?= g++
NVCC ?= nvcc
PYTHON ?= python3.6
BASH ?= bash
.DEFAULT_GOAL := all

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
COMPLIANCE_TEST_SOURCES := $(HOST_TEST_SOURCES) $(CUDA_TEST_SOURCES) \
  tests/cpu_reference.cpp
HOST_TEST_NAMES := $(basename $(notdir $(HOST_TEST_SOURCES)))
CUDA_TEST_NAMES := $(basename $(notdir $(CUDA_TEST_SOURCES)))
COLLIDING_TEST_NAMES := $(filter $(HOST_TEST_NAMES),$(CUDA_TEST_NAMES))
HOST_UNIQUE_NAMES := $(filter-out $(COLLIDING_TEST_NAMES),$(HOST_TEST_NAMES))
CUDA_UNIQUE_NAMES := $(filter-out $(COLLIDING_TEST_NAMES),$(CUDA_TEST_NAMES))

HOST_TEST_OBJECTS := $(addprefix $(HOST_OBJECT_DIR)/,$(addsuffix .o,$(HOST_TEST_NAMES)))
CUDA_TEST_OBJECTS := $(addprefix $(CUDA_OBJECT_DIR)/,$(addsuffix .o,$(CUDA_TEST_NAMES)))
CUDA_KERNEL_SOURCES := src/kernels/input.cu src/kernels/activation.cu \
  src/kernels/pooling.cu src/kernels/linear.cu src/kernels/convolution.cu \
  src/kernels/loss.cu src/kernels/metrics.cu src/kernels/adam.cu
CUDA_KERNEL_OBJECTS := $(patsubst src/kernels/%.cu,$(CUDA_OBJECT_DIR)/kernels/%.o,$(CUDA_KERNEL_SOURCES))
LENET_OBJECT := $(CUDA_OBJECT_DIR)/lenet.o
TRAIN_OBJECT := $(CUDA_OBJECT_DIR)/train.o
MAIN_OBJECT := $(CUDA_OBJECT_DIR)/main.o
DATASET_PROBE_OBJECT := $(HOST_OBJECT_DIR)/dataset_probe.o
DATASET_OBJECT := $(HOST_OBJECT_DIR)/dataset.o
RANDOM_OBJECT := $(HOST_OBJECT_DIR)/random.o
PARAMETERS_OBJECT := $(HOST_OBJECT_DIR)/parameters.o
CHECKPOINT_OBJECT := $(HOST_OBJECT_DIR)/checkpoint.o
CPU_REFERENCE_OBJECT := $(HOST_OBJECT_DIR)/cpu_reference.o
CLI_OBJECT := $(HOST_OBJECT_DIR)/cli.o
TRAINING_DATA_OBJECT := $(HOST_OBJECT_DIR)/training_data.o
REPORTING_OBJECT := $(HOST_OBJECT_DIR)/reporting.o
HOST_TEST_PROGRAMS := $(strip \
  $(addprefix $(BUILD_DIR)/,$(addsuffix $(EXEEXT),$(HOST_UNIQUE_NAMES))) \
  $(addprefix $(BUILD_DIR)/host/,$(addsuffix $(EXEEXT),$(COLLIDING_TEST_NAMES))))
CUDA_TEST_PROGRAMS := $(strip \
  $(addprefix $(BUILD_DIR)/,$(addsuffix $(EXEEXT),$(CUDA_UNIQUE_NAMES))) \
  $(addprefix $(BUILD_DIR)/cuda/,$(addsuffix $(EXEEXT),$(COLLIDING_TEST_NAMES))))
WORKFLOW_TEST_PROGRAM := $(BUILD_DIR)/workflow_tests$(EXEEXT)
CUDA_TEST_PROGRAMS_WITHOUT_WORKFLOW := $(filter-out $(WORKFLOW_TEST_PROGRAM),$(CUDA_TEST_PROGRAMS))
PYTHON_TEST_MODULES := tests.test_prepare_mnist tests.test_data_interop
COMPLIANCE_TEST_MODULES := tests.test_check_prohibited \
  tests.test_check_comments tests.test_documentation \
  tests.output_format_tests tests.compliance_tests
DEPENDENCY_FILES := $(HOST_TEST_OBJECTS:.o=.d) $(CUDA_TEST_OBJECTS:.o=.d) \
  $(DATASET_PROBE_OBJECT:.o=.d) $(DATASET_OBJECT:.o=.d) \
  $(RANDOM_OBJECT:.o=.d) $(PARAMETERS_OBJECT:.o=.d) \
  $(CHECKPOINT_OBJECT:.o=.d) $(CPU_REFERENCE_OBJECT:.o=.d) \
  $(CLI_OBJECT:.o=.d) $(TRAINING_DATA_OBJECT:.o=.d) \
  $(REPORTING_OBJECT:.o=.d) \
  $(CUDA_KERNEL_OBJECTS:.o=.d) $(LENET_OBJECT:.o=.d) \
  $(TRAIN_OBJECT:.o=.d) $(MAIN_OBJECT:.o=.d)

ifeq ($(V),1)
Q :=
else
Q := @
endif

.PHONY: all host-tests cuda-tests python-tests prepare-data compliance test \
  makefile-tests check acceptance compliance-test-sources clean
.SECONDARY: $(HOST_TEST_OBJECTS) $(CUDA_TEST_OBJECTS)

all: $(BUILD_DIR)/lenet_cuda$(EXEEXT) $(HOST_TEST_PROGRAMS) \
  $(CUDA_TEST_PROGRAMS)
	@echo "event=build status=pass target=all"

compliance-test-sources:
	@for source in $(COMPLIANCE_TEST_SOURCES); do \
	  printf 'test-source=%s\n' "$$source"; \
	done

host-tests: $(HOST_TEST_PROGRAMS)
	$(Q)set -e; for test in $(HOST_TEST_PROGRAMS); do "$$test"; done

prepare-data:
	$(Q)$(PYTHON) scripts/prepare_mnist.py --output-dir data

cuda-tests: $(CUDA_TEST_PROGRAMS) prepare-data
	$(Q)set -e; for test in $(CUDA_TEST_PROGRAMS_WITHOUT_WORKFLOW); do "$$test"; done
	$(Q)$(WORKFLOW_TEST_PROGRAM) --mnist-train data/train.bin

python-tests:
	$(Q)$(PYTHON) -m unittest -v $(PYTHON_TEST_MODULES)

compliance:
	$(Q)$(PYTHON) -m unittest -v $(COMPLIANCE_TEST_MODULES)
	$(Q)$(PYTHON) scripts/check_comments.py --root . \
	  --checklist docs/comment-review-checklist.md
	$(Q)$(BASH) scripts/check_prohibited.sh source .

test: host-tests cuda-tests python-tests compliance

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
$(BUILD_DIR)/cli_tests$(EXEEXT): $(CLI_OBJECT)
$(BUILD_DIR)/training_data_tests$(EXEEXT): $(TRAINING_DATA_OBJECT) \
    $(DATASET_OBJECT) $(RANDOM_OBJECT)
$(BUILD_DIR)/reporting_tests$(EXEEXT): $(REPORTING_OBJECT)
$(BUILD_DIR)/operator_tests$(EXEEXT): $(CPU_REFERENCE_OBJECT) $(PARAMETERS_OBJECT) \
    $(RANDOM_OBJECT) $(LENET_OBJECT) $(CUDA_KERNEL_OBJECTS)
$(BUILD_DIR)/workflow_tests$(EXEEXT): $(CPU_REFERENCE_OBJECT) $(DATASET_OBJECT) \
    $(RANDOM_OBJECT) $(PARAMETERS_OBJECT) $(CHECKPOINT_OBJECT) \
    $(TRAINING_DATA_OBJECT) $(REPORTING_OBJECT) $(TRAIN_OBJECT) $(LENET_OBJECT) \
    $(CUDA_KERNEL_OBJECTS) | $(BUILD_DIR)/lenet_cuda$(EXEEXT)

$(BUILD_DIR)/lenet_cuda$(EXEEXT): $(MAIN_OBJECT) $(CLI_OBJECT) $(DATASET_OBJECT) \
    $(RANDOM_OBJECT) $(PARAMETERS_OBJECT) $(CHECKPOINT_OBJECT) \
    $(TRAINING_DATA_OBJECT) $(REPORTING_OBJECT) $(TRAIN_OBJECT) $(LENET_OBJECT) \
    $(CUDA_KERNEL_OBJECTS) | $(BUILD_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD_DIR)/dataset_probe$(EXEEXT): $(DATASET_PROBE_OBJECT) $(DATASET_OBJECT) | $(BUILD_DIR)
	$(Q)$(CXX) $(CXXFLAGS) $^ -o $@

ifneq ($(EXEEXT),)
$(BUILD_DIR)/dataset_tests: $(BUILD_DIR)/dataset_tests$(EXEEXT)
$(BUILD_DIR)/dataset_probe: $(BUILD_DIR)/dataset_probe$(EXEEXT)
$(BUILD_DIR)/random_tests: $(BUILD_DIR)/random_tests$(EXEEXT)
$(BUILD_DIR)/parameters_tests: $(BUILD_DIR)/parameters_tests$(EXEEXT)
$(BUILD_DIR)/checkpoint_tests: $(BUILD_DIR)/checkpoint_tests$(EXEEXT)
$(BUILD_DIR)/cpu_reference_tests: $(BUILD_DIR)/cpu_reference_tests$(EXEEXT)
$(BUILD_DIR)/cli_tests: $(BUILD_DIR)/cli_tests$(EXEEXT)
$(BUILD_DIR)/training_data_tests: $(BUILD_DIR)/training_data_tests$(EXEEXT)
$(BUILD_DIR)/reporting_tests: $(BUILD_DIR)/reporting_tests$(EXEEXT)
$(BUILD_DIR)/operator_tests: $(BUILD_DIR)/operator_tests$(EXEEXT)
$(BUILD_DIR)/workflow_tests: $(BUILD_DIR)/workflow_tests$(EXEEXT)
$(BUILD_DIR)/lenet_cuda: $(BUILD_DIR)/lenet_cuda$(EXEEXT)
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

$(CLI_OBJECT): src/cli.cpp | $(HOST_OBJECT_DIR)
	$(Q)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(TRAINING_DATA_OBJECT): src/training_data.cpp | $(HOST_OBJECT_DIR)
	$(Q)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(REPORTING_OBJECT): src/reporting.cpp | $(HOST_OBJECT_DIR)
	$(Q)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(CUDA_OBJECT_DIR)/%.o: tests/%.cu | $(CUDA_OBJECT_DIR)
	$(Q)$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -MMD -MP -c $< -o $@

$(CUDA_OBJECT_DIR)/kernels/%.o: src/kernels/%.cu | $(CUDA_OBJECT_DIR)/kernels
	$(Q)$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -MMD -MP -c $< -o $@

$(LENET_OBJECT): src/lenet.cu | $(CUDA_OBJECT_DIR)
	$(Q)$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -MMD -MP -c $< -o $@

$(TRAIN_OBJECT): src/train.cu | $(CUDA_OBJECT_DIR)
	$(Q)$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -MMD -MP -c $< -o $@

$(MAIN_OBJECT): src/main.cu | $(CUDA_OBJECT_DIR)
	$(Q)$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR) $(BUILD_DIR)/host $(BUILD_DIR)/cuda \
    $(HOST_OBJECT_DIR) $(CUDA_OBJECT_DIR) $(CUDA_OBJECT_DIR)/kernels:
	$(Q)mkdir -p $@

clean:
	$(Q)rm -rf $(BUILD_DIR)

-include $(DEPENDENCY_FILES)
