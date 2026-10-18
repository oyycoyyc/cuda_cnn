#!/usr/bin/env bash

# Persistent project paths, with documented Gemini mount points as fallbacks.
export CUDA_LENET_ROOT="${GEMINI_CODE:-/gemini/code}/cuda_cnn"
export CUDA_LENET_TRAIN_DATA="${GEMINI_DATA_IN1:-/gemini/data-1}/train.bin"
export CUDA_LENET_TEST_DATA="${GEMINI_DATA_IN1:-/gemini/data-1}/test.bin"
export CUDA_LENET_WEIGHTS="$CUDA_LENET_ROOT/weights/lenet.bin"
export CUDA_LENET_OUTPUT_ROOT="${GEMINI_DATA_OUT:-$CUDA_LENET_ROOT/runtime-output}"

case ":${PATH:-}:" in
  *:/usr/local/cuda/bin:*) ;;
  *) PATH="/usr/local/cuda/bin${PATH:+:$PATH}" ;;
esac
case ":${PATH:-}:" in
  *:/root/miniconda3/bin:*) ;;
  *) PATH="/root/miniconda3/bin${PATH:+:$PATH}" ;;
esac
export PATH

case ":${LD_LIBRARY_PATH:-}:" in
  *:/usr/local/cuda/lib64:*) ;;
  *) LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
export LD_LIBRARY_PATH
