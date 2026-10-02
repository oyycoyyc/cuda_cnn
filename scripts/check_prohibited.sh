#!/usr/bin/env bash

set -u
set -o pipefail

usage() {
  echo "usage: $0 source ROOT | build VERBOSE_BUILD_LOG | dry-run VERBOSE_BUILD_LOG" >&2
  exit 2
}

fail() {
  echo "$1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command is unavailable: $1"
}

find_policy_python() {
  local candidate
  if [[ -n ${PYTHON:-} ]]; then
    printf '%s\n' "$PYTHON"
    return 0
  fi
  for candidate in python3.6 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c 'import sys' >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

scan_pattern() {
  local label=$1
  local pattern=$2
  shift 2
  local output
  local status

  if output=$(grep -EinH -- "$pattern" "$@" 2>&1); then
    echo "$label detected:" >&2
    echo "$output" >&2
    return 1
  else
    status=$?
  fi
  if [[ $status -ne 1 ]]; then
    echo "grep failed while scanning for $label:" >&2
    echo "$output" >&2
    return 2
  fi
  return 0
}

find_project_root() {
  local directory=$1
  while [[ $directory != "/" && ! -f "$directory/Makefile" ]]; do
    directory=$(dirname "$directory")
  done
  [[ -f "$directory/Makefile" ]] || return 1
  printf '%s\n' "$directory"
}

PROHIBITED_FAMILIES='cudnn|cublas|nvinfer|nvonnxparser|nvparsers|createinfer(runtime|builder|refitter)|createparser|curand|cusolver|cusparse|cufft|cutensor|cutlass|cudss|nccl|nvshmem|npp([a-z0-9_]*)|torch|tensorflow|caffe|onnxruntime|openvino|opencv|thrust|cub([^a-z0-9_]|$)|cuda_fp16|(^|[/<"[:space:]])mma\.h'
PROHIBITED_LINKS='(^|[[:space:],=/\\])(-l(:lib)?|lib)?(cudnn|cublas|nvinfer|nvonnxparser|nvparsers|curand|cusolver|cusparse|cufft|cutensor|cutlass|cudss|nccl|nvshmem|npp[a-z0-9_]*|torch|tensorflow|caffe|onnxruntime|openvino|opencv)([^a-z0-9_]|$)'
CPU_FALLBACK='cpu[_:]*reference|cpu[a-z0-9_]*(forward|backward|infer|train)|runoncpu'
NONDETERMINISTIC_RANDOM='#[[:space:]]*include[[:space:]]*[<"][[:space:]]*random[[:space:]]*[>"]|std::(shuffle|random_device|mt19937(_64)?|default_random_engine|[a-z_]*distribution)'
BUILD_FAILURES='(^|[[:space:]])fatal[[:space:]]+error:|^error:|:[0-9]+(:[0-9]+)?:[[:space:]]+(fatal[[:space:]]+)?error:|(^|[[:space:]])(collect2|ld|nvcc|clang\+\+|g\+\+):[[:space:]]+error:|make(\[[0-9]+\])?:.*[[:space:]]Error[[:space:]]+[1-9][0-9]*([^0-9]|$)|make(\[[0-9]+\])?:[[:space:]]+\*\*\*[[:space:]]+(No rule|missing separator|recipe commences before first target)'

[[ $# -eq 2 ]] || usage
mode=$1
target=$2
script_directory=$(cd "$(dirname "$0")" 2>/dev/null && pwd) || \
  fail "cannot locate policy helper directory"
graph_analyzer="$script_directory/analyze_build_graph.py"
[[ -f $graph_analyzer ]] || fail "build graph analyzer is unavailable: $graph_analyzer"

case "$mode" in
  source)
    require_command find
    require_command grep
    require_command make
    policy_python=$(find_policy_python) || fail "required Python interpreter is unavailable"
    [[ -d "$target" ]] || fail "source root is not a directory: $target"
    root=$(cd "$target" 2>/dev/null && pwd) || fail "cannot read source root: $target"
    [[ -f "$root/Makefile" ]] || fail "source root has no Makefile: $root"

    temporary=$(mktemp) || fail "cannot create scanner temporary file"
    recipes=$(mktemp) || fail "cannot create recipe temporary file"
    manifest=$(mktemp) || fail "cannot create manifest temporary file"
    active_inputs=$(mktemp) || fail "cannot create active-input temporary file"
    trap 'rm -f "$temporary" "$recipes" "$manifest" "$active_inputs"' EXIT
    if ! find "$root/include" "$root/src" -type f \
        \( -name '*.h' -o -name '*.hpp' -o -name '*.c' -o -name '*.cc' \
           -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) \
        -print0 >"$temporary"; then
      fail "find failed while collecting production sources"
    fi

    files=("$root/Makefile")
    while IFS= read -r -d '' file; do
      files+=("$file")
    done <"$temporary"

    if ! scan_pattern "prohibited dependency/API" "$PROHIBITED_FAMILIES" \
        "${files[@]}"; then
      exit 1
    fi
    if ! scan_pattern "prohibited linker input" "$PROHIBITED_LINKS" "$root/Makefile"; then
      exit 1
    fi

    if ! build_commands=$(make -C "$root" -Bn all 2>&1); then
      fail "make dry-run failed while resolving compiled inputs and production link graph:\n$build_commands"
    fi

    if ! printf '%s\n' "$build_commands" >"$recipes"; then
      fail "cannot store Make recipe output"
    fi
    if ! make_manifest=$(make -s --no-print-directory -C "$root" \
        compliance-test-sources 2>&1); then
      fail "make failed while resolving the expected test-source manifest:\n$make_manifest"
    fi
    if ! printf '%s\n' "$make_manifest" >"$manifest"; then
      fail "cannot store Make test-source manifest"
    fi
    if ! "$policy_python" "$graph_analyzer" source \
        --root "$root" --recipes "$recipes" --manifest "$manifest" \
        >"$active_inputs"; then
      fail "Make recipe provenance analysis failed"
    fi
    while IFS= read -r file || [[ -n $file ]]; do
      file=${file%$'\r'}
      [[ -f "$root/$file" ]] || fail "analyzed active input is not a file: $file"
      files+=("$root/$file")
    done <"$active_inputs" || fail "cannot read analyzed active inputs"

    if ! scan_pattern "prohibited dependency/API" "$PROHIBITED_FAMILIES" "${files[@]}"; then
      exit 1
    fi

    production_files=()
    while IFS= read -r -d '' file; do
      production_files+=("$file")
    done <"$temporary"
    if ! scan_pattern "production CPU fallback" "$CPU_FALLBACK" "${production_files[@]}"; then
      exit 1
    fi
    if ! scan_pattern "implementation-dependent random API" \
        "$NONDETERMINISTIC_RANDOM" "${production_files[@]}"; then
      exit 1
    fi

    echo "compliance scan passed: mode=source scope=make-compiled-inputs"
    ;;
  build|dry-run)
    require_command find
    require_command grep
    policy_python=$(find_policy_python) || fail "required Python interpreter is unavailable"
    [[ -f "$target" ]] || fail "build log is not a readable file: $target"
    [[ -s "$target" ]] || fail "build log is empty: $target"

    if ! project_root=$(find_project_root "$(cd "$(dirname "$target")" && pwd)"); then
      fail "cannot locate project Makefile above build log: $target"
    fi
    newer=$(mktemp) || fail "cannot create scanner temporary file"
    trap 'rm -f "$newer"' EXIT
    if ! find "$project_root/include" "$project_root/src" "$project_root/tests" \
        "$project_root/Makefile" \
        -type f -newer "$target" -print >"$newer"; then
      fail "find failed while checking build-log freshness"
    fi
    if [[ -s "$newer" ]]; then
      echo "build log is stale; newer build inputs:" >&2
      cat "$newer" >&2
      exit 1
    fi

    if ! "$policy_python" "$graph_analyzer" build-log \
        --root "$project_root" --log "$target"; then
      fail "build-log recipe analysis failed"
    fi
    if ! scan_pattern "prohibited dependency/linker input" \
        "$PROHIBITED_FAMILIES|$PROHIBITED_LINKS" "$target"; then
      exit 1
    fi
    if [[ $mode == build ]]; then
      if ! scan_pattern "build failure record" "$BUILD_FAILURES" "$target"; then
        exit 1
      fi
      mapfile -t log_lines <"$target" || fail "cannot read build log: $target"
      final_nonempty=
      for line in "${log_lines[@]}"; do
        line=${line%$'\r'}
        [[ $line =~ ^[[:space:]]*$ ]] || final_nonempty=$line
      done
      [[ $final_nonempty == 'event=build status=pass target=all' ]] || \
        fail "successful-build completion marker is not the final nonempty line"
    fi
    if [[ $mode == dry-run ]]; then
      echo "compliance scan passed: mode=dry-run evidence=commands-not-executed"
    else
      echo "compliance scan passed: mode=build evidence=successful-build"
    fi
    ;;
  *)
    usage
    ;;
esac
