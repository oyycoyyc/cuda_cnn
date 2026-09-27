#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
host_fixture="$root/tests/make_collision_tests.cpp"
cuda_fixture="$root/tests/make_collision_tests.cu"

if [ -e "$host_fixture" ] || [ -e "$cuda_fixture" ]; then
  printf '%s\n' 'makefile test fixture path already exists' >&2
  exit 1
fi

temp_dir=$(mktemp -d)

cleanup() {
  rm -f "$host_fixture" "$cuda_fixture"
  rm -rf "$temp_dir"
}
trap cleanup EXIT HUP INT TERM

printf '%s\n' 'int main() { return 0; }' >"$host_fixture"
printf '%s\n' 'int main() { return 0; }' >"$cuda_fixture"

(
  cd "$root"
  make -n V=1 host-tests >"$temp_dir/host.out"
  make -n V=1 cuda-tests >"$temp_dir/cuda.out"
)

grep -F 'tests/make_collision_tests.cpp' "$temp_dir/host.out" >/dev/null
grep -F 'tests/make_collision_tests.cu' "$temp_dir/cuda.out" >/dev/null

host_run=$(grep '^set -e; for test in ' "$temp_dir/host.out")
cuda_run=$(grep '^set -e; for test in ' "$temp_dir/cuda.out")
host_binary=$(printf '%s\n' "$host_run" | tr ' ' '\n' |
  grep 'make_collision_tests' | sed -n '1{s/;.*$//;p;}')
cuda_binary=$(printf '%s\n' "$cuda_run" | tr ' ' '\n' |
  grep 'make_collision_tests' | sed -n '1{s/;.*$//;p;}')

if [ -z "$host_binary" ] || [ -z "$cuda_binary" ] ||
   [ "$host_binary" = "$cuda_binary" ]; then
  printf '%s\n' "host_binary=$host_binary" "cuda_binary=$cuda_binary" >&2
  exit 1
fi

printf '%s\n' "host_binary=$host_binary" "cuda_binary=$cuda_binary"
printf '%s\n' 'event=makefile_test name=distinct_suite_binaries status=pass'
