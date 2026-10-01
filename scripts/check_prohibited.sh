#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 source ROOT | build VERBOSE_BUILD_LOG" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

for command_name in find grep; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "required command is unavailable: $command_name" >&2
    exit 2
  fi
done

mode=$1
target=$2
files=()

case "$mode" in
  source)
    if [[ ! -d "$target" ]]; then
      echo "source root does not exist: $target" >&2
      exit 2
    fi
    [[ -f "$target/Makefile" ]] && files+=("$target/Makefile")
    for directory in include src tests; do
      [[ -d "$target/$directory" ]] || continue
      while IFS= read -r -d '' file; do
        files+=("$file")
      done < <(find "$target/$directory" -type f \
        \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \
           -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
        -print0)
    done
    ;;
  build)
    if [[ ! -f "$target" ]]; then
      echo "verbose build log does not exist: $target" >&2
      exit 2
    fi
    files+=("$target")
    ;;
  *)
    usage
    exit 2
    ;;
esac

patterns=(
  'prohibited header|#[[:space:]]*include[[:space:]]*[<"][^>"]*(cudnn|cublas|nvinfer|nvonnxparser|thrust/|cub/|curand)'
  'prohibited namespace|(^|[^[:alnum:]_])(nvinfer1|thrust|cub)[[:space:]]*::'
  'prohibited API|(^|[^[:alnum:]_])(cudnn[A-Za-z0-9_]*|cublas[A-Za-z0-9_]*|curand[A-Za-z0-9_]*|createInfer(Runtime|Builder|Refitter))[[:space:]]*\('
  'prohibited linker input|(^|[^[:alnum:]_])-l(cudnn|cublas|nvinfer|nvonnxparser|curand)[A-Za-z0-9_.-]*([^[:alnum:]_]|$)'
  'prohibited shared library|(^|[/\\])lib(cudnn|cublas|nvinfer|nvonnxparser|curand)[A-Za-z0-9_.-]*\.(so|a|dylib|lib)([^[:alnum:]_]|$)'
)

status=0
for entry in "${patterns[@]}"; do
  label=${entry%%|*}
  expression=${entry#*|}
  if [[ ${#files[@]} -gt 0 ]]; then
    matches=$(grep -nHIEi "$expression" "${files[@]}" || true)
    if [[ -n "$matches" ]]; then
      echo "$label detected:" >&2
      echo "$matches" >&2
      status=1
    fi
  fi
done

if [[ "$mode" == source ]]; then
  production_files=()
  for directory in include src; do
    [[ -d "$target/$directory" ]] || continue
    while IFS= read -r -d '' file; do
      production_files+=("$file")
    done < <(find "$target/$directory" -type f \
      \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \
         -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
      -print0)
  done
  if [[ ${#production_files[@]} -gt 0 ]]; then
    fallback_matches=$(grep -nHIEi \
      'cpu_reference([.]h)?|cpu_reference[[:space:]]*::|std[[:space:]]*::[[:space:]]*(shuffle|normal_distribution)' \
      "${production_files[@]}" || true)
    if [[ -n "$fallback_matches" ]]; then
      echo "CPU fallback or nondeterministic random API detected:" >&2
      echo "$fallback_matches" >&2
      status=1
    fi
  fi
fi

if [[ $status -ne 0 ]]; then
  exit $status
fi

echo "compliance scan passed: mode=$mode"
