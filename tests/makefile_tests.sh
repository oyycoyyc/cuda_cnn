#!/bin/sh
set -eu

# Regression: host and CUDA test suites must build distinct collision binaries.
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# Same stem in both source trees to provoke a name collision.
host_fixture="$root/tests/make_collision_tests.cpp"
cuda_fixture="$root/tests/make_collision_tests.cu"

# Refuse to run if a previous fixture was left behind.
if [ -e "$host_fixture" ] || [ -e "$cuda_fixture" ]; then
  printf '%s\n' 'makefile test fixture path already exists' >&2
  exit 1
fi

# Scratch directory for captured dry-run output.
temp_dir=$(mktemp -d)

# Removes fixtures and scratch output on every exit path.
cleanup() {
  rm -f "$host_fixture" "$cuda_fixture"
  rm -rf "$temp_dir"
}
# Register cleanup so interruption cannot leave fixture files behind.
trap cleanup EXIT HUP INT TERM

# Minimal compilable fixtures so the dry run reaches the link command.
printf '%s\n' 'int main() { return 0; }' >"$host_fixture"
printf '%s\n' 'int main() { return 0; }' >"$cuda_fixture"

# Capture host and CUDA dry-run recipes without executing the toolchain.
(
  cd "$root"
  make -n V=1 host-tests >"$temp_dir/host.out"
  make -n V=1 cuda-tests >"$temp_dir/cuda.out"
)

# Each dry-run recipe must reference its own collision fixture.
grep -F 'tests/make_collision_tests.cpp' "$temp_dir/host.out" >/dev/null
grep -F 'tests/make_collision_tests.cu' "$temp_dir/cuda.out" >/dev/null

# Extract each suite runner command and the first binary it invokes.
host_run=$(grep '^set -e; for test in ' "$temp_dir/host.out")
cuda_run=$(grep '^set -e; for test in ' "$temp_dir/cuda.out")
host_binary=$(printf '%s\n' "$host_run" | tr ' ' '\n' |
  grep 'make_collision_tests' | sed -n '1{s/;.*$//;p;}')
cuda_binary=$(printf '%s\n' "$cuda_run" | tr ' ' '\n' |
  grep 'make_collision_tests' | sed -n '1{s/;.*$//;p;}')

# Both suites must resolve a binary, and the two paths must differ.
if [ -z "$host_binary" ] || [ -z "$cuda_binary" ] ||
   [ "$host_binary" = "$cuda_binary" ]; then
  printf '%s\n' "host_binary=$host_binary" "cuda_binary=$cuda_binary" >&2
  exit 1
fi

# Emit the resolved binaries and the regression pass record.
printf '%s\n' "host_binary=$host_binary" "cuda_binary=$cuda_binary"
printf '%s\n' 'event=makefile_test name=distinct_suite_binaries status=pass'
