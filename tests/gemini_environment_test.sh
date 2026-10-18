#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

unset GEMINI_CODE GEMINI_DATA_IN1 GEMINI_DATA_OUT
PATH=/usr/bin
LD_LIBRARY_PATH=
source "$project_root/scripts/gemini_env.sh"

test "$CUDA_LENET_ROOT" = /gemini/code/cuda_cnn
test "$CUDA_LENET_TRAIN_DATA" = /gemini/data-1/train.bin
test "$CUDA_LENET_TEST_DATA" = /gemini/data-1/test.bin
test "$CUDA_LENET_WEIGHTS" = /gemini/code/cuda_cnn/weights/lenet.bin
test "$CUDA_LENET_OUTPUT_ROOT" = /gemini/code/cuda_cnn/runtime-output
test "$PATH" = /root/miniconda3/bin:/usr/local/cuda/bin:/usr/bin
test "$LD_LIBRARY_PATH" = /usr/local/cuda/lib64

first_path=$PATH
first_library_path=$LD_LIBRARY_PATH
source "$project_root/scripts/gemini_env.sh"
test "$PATH" = "$first_path"
test "$LD_LIBRARY_PATH" = "$first_library_path"
bash -c 'test "$CUDA_LENET_ROOT" = /gemini/code/cuda_cnn'
bash -c 'test "$CUDA_LENET_TEST_DATA" = /gemini/data-1/test.bin'

GEMINI_CODE=/custom/code
GEMINI_DATA_IN1=/custom/data
GEMINI_DATA_OUT=/custom/output
PATH=/usr/bin
LD_LIBRARY_PATH=/existing/lib
source "$project_root/scripts/gemini_env.sh"

test "$CUDA_LENET_ROOT" = /custom/code/cuda_cnn
test "$CUDA_LENET_TRAIN_DATA" = /custom/data/train.bin
test "$CUDA_LENET_TEST_DATA" = /custom/data/test.bin
test "$CUDA_LENET_WEIGHTS" = /custom/code/cuda_cnn/weights/lenet.bin
test "$CUDA_LENET_OUTPUT_ROOT" = /custom/output
test "$PATH" = /root/miniconda3/bin:/usr/local/cuda/bin:/usr/bin
test "$LD_LIBRARY_PATH" = /usr/local/cuda/lib64:/existing/lib

echo "event=gemini_environment_test status=pass"
