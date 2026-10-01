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

[[ $# -eq 2 ]] || usage
mode=$1
target=$2

case "$mode" in
  source)
    require_command find
    require_command grep
    require_command make
    [[ -d "$target" ]] || fail "source root is not a directory: $target"
    root=$(cd "$target" 2>/dev/null && pwd) || fail "cannot read source root: $target"
    [[ -f "$root/Makefile" ]] || fail "source root has no Makefile: $root"

    temporary=$(mktemp) || fail "cannot create scanner temporary file"
    trap 'rm -f "$temporary"' EXIT
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

    if ! make_database=$(make -C "$root" -nprR 2>&1); then
      fail "make failed while resolving compiled source inputs:\n$make_database"
    fi
    while IFS= read -r token; do
      token=${token#./}
      [[ -n $token && -f "$root/$token" ]] || continue
      case "$token" in
        tests/*.c|tests/*.cc|tests/*.cpp|tests/*.cu|tests/*.cuh|tests/*.h|tests/*.hpp)
          files+=("$root/$token")
          ;;
      esac
    done < <(printf '%s\n' "$make_database" | grep -Eo 'tests/[A-Za-z0-9_./-]+\.(c|cc|cpp|cu|cuh|h|hpp)' | sort -u)

    if ! scan_pattern "prohibited dependency/API" "$PROHIBITED_FAMILIES" "${files[@]}"; then
      exit 1
    fi
    if ! scan_pattern "prohibited linker input" "$PROHIBITED_LINKS" "$root/Makefile"; then
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

    if ! build_commands=$(make -C "$root" -Bn 2>&1); then
      fail "make dry-run failed while resolving the production link graph:\n$build_commands"
    fi
    found_application_link=0
    while IFS= read -r line; do
      if [[ $line =~ -o[[:space:]]+([^[:space:]]*/)?lenet_cuda(\.exe)?([[:space:]]|$) ]]; then
        found_application_link=1
        if [[ $line =~ [Cc][Pp][Uu][_-]?[Rr][Ee][Ff][Ee][Rr][Ee][Nn][Cc][Ee] ]]; then
          fail "production lenet_cuda link includes a CPU reference object:\n$line"
        fi
      fi
    done <<<"$build_commands"
    [[ $found_application_link -eq 1 ]] || \
      fail "make dry-run did not expose a lenet_cuda linker command"

    echo "compliance scan passed: mode=source scope=make-compiled-inputs"
    ;;
  build|dry-run)
    require_command find
    require_command grep
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

    if grep -Eq '(^|[ /\\])(nvcc|g\+\+|c\+\+)(\.exe)?[[:space:]]' "$target"; then
      :
    else
      status=$?
      [[ $status -eq 1 ]] || fail "grep failed while validating compiler commands"
      fail "build log contains no compiler invocation"
    fi
    if grep -Eq '(^|[ /\\])(nvcc|g\+\+|c\+\+)(\.exe)?.*[[:space:]]-c([[:space:]]|$)' "$target"; then
      :
    else
      status=$?
      [[ $status -eq 1 ]] || fail "grep failed while validating compile commands"
      fail "build log contains no compile command"
    fi
    if link_commands=$(grep -E -- '-o[[:space:]]+([^[:space:]]*/)?lenet_cuda(\.exe)?([[:space:]]|$)' "$target"); then
      :
    else
      status=$?
      [[ $status -eq 1 ]] || fail "grep failed while validating the lenet_cuda link"
      fail "build log contains no lenet_cuda linker command"
    fi
    for flag in \
      '-gencode=arch=compute_90,code=sm_90' \
      '-gencode=arch=compute_90,code=compute_90'; do
      if grep -Fq -- "$flag" "$target"; then
        :
      else
        status=$?
        [[ $status -eq 1 ]] || fail "grep failed while validating architecture flags"
        fail "build log is missing required architecture flag: $flag"
      fi
    done
    valid_link=0
    while IFS= read -r line; do
      if [[ $line == *'-gencode=arch=compute_90,code=sm_90'* &&
            $line == *'-gencode=arch=compute_90,code=compute_90'* ]]; then
        valid_link=1
      fi
    done <<<"$link_commands"
    [[ $valid_link -eq 1 ]] || \
      fail "lenet_cuda linker command is missing required architecture flags"
    if [[ $mode == build ]]; then
      if grep -Fxq 'event=build status=pass target=all' "$target"; then
        :
      else
        status=$?
        [[ $status -eq 1 ]] || fail "grep failed while validating build completion"
        fail "build log is missing the successful-build completion marker"
      fi
    fi
    if ! scan_pattern "prohibited dependency/linker input" \
        "$PROHIBITED_FAMILIES|$PROHIBITED_LINKS" "$target"; then
      exit 1
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
