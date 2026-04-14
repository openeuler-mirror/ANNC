#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TEST_BIN_DIR=""
TEST_TMP_DIR="$PROJECT_ROOT/tests/tmp"

BUILD_DIR=""
KERNEL_LIB=""
BUILTIN_LIB=""
THREADPOOL_LIB=""
for candidate in "$PROJECT_ROOT/build" "$PROJECT_ROOT/cmake-build-debug-wsl"; do
  if [[ -f "$candidate/lib/libANNCKernel.a" && -f "$candidate/lib/libANNCBuiltinKernels.a" && -f "$candidate/lib/libANNCThreadPool.a" ]]; then
    BUILD_DIR="$candidate"
    KERNEL_LIB="$candidate/lib/libANNCKernel.a"
    BUILTIN_LIB="$candidate/lib/libANNCBuiltinKernels.a"
    THREADPOOL_LIB="$candidate/lib/libANNCThreadPool.a"
    break
  fi

  if [[ -f "$candidate/annc/lib/Kernel/libANNCKernel.a" && -f "$candidate/annc/lib/Kernel/builtin_kernels/libANNCBuiltinKernels.a" && -f "$candidate/annc/lib/Kernel/threadpool/libANNCThreadPool.a" ]]; then
    BUILD_DIR="$candidate"
    KERNEL_LIB="$candidate/annc/lib/Kernel/libANNCKernel.a"
    BUILTIN_LIB="$candidate/annc/lib/Kernel/builtin_kernels/libANNCBuiltinKernels.a"
    THREADPOOL_LIB="$candidate/annc/lib/Kernel/threadpool/libANNCThreadPool.a"
    break
  fi
done

if [[ -z "$BUILD_DIR" ]]; then
  echo "Failed to locate a usable build directory. Checked:"
  echo "  - $PROJECT_ROOT/build/lib/libANNCKernel.a"
  echo "  - $PROJECT_ROOT/build/lib/libANNCBuiltinKernels.a"
  echo "  - $PROJECT_ROOT/build/lib/libANNCThreadPool.a"
  echo "  - $PROJECT_ROOT/build/annc/lib/Kernel/libANNCKernel.a"
  echo "  - $PROJECT_ROOT/build/annc/lib/Kernel/builtin_kernels/libANNCBuiltinKernels.a"
  echo "  - $PROJECT_ROOT/build/annc/lib/Kernel/threadpool/libANNCThreadPool.a"
  echo "  - $PROJECT_ROOT/cmake-build-debug-wsl/lib/libANNCKernel.a"
  echo "  - $PROJECT_ROOT/cmake-build-debug-wsl/lib/libANNCBuiltinKernels.a"
  echo "  - $PROJECT_ROOT/cmake-build-debug-wsl/lib/libANNCThreadPool.a"
  echo "  - $PROJECT_ROOT/cmake-build-debug-wsl/annc/lib/Kernel/libANNCKernel.a"
  echo "  - $PROJECT_ROOT/cmake-build-debug-wsl/annc/lib/Kernel/builtin_kernels/libANNCBuiltinKernels.a"
  echo "  - $PROJECT_ROOT/cmake-build-debug-wsl/annc/lib/Kernel/threadpool/libANNCThreadPool.a"
  exit 1
fi

TEST_BIN_DIR="$BUILD_DIR/bin"

mkdir -p "$TEST_BIN_DIR" "$TEST_TMP_DIR"

AUTO_WRAPPER_GEN="$TEST_TMP_DIR/auto_kernel_wrappers.gen.cpp"
AUTO_REGISTRY_GEN="$TEST_TMP_DIR/auto_kernel_registry.gen.cpp"

cat > "$AUTO_WRAPPER_GEN" <<'GENEOF'
#include "Kernel/MemRefTypes.h"
#define ANNC_SPEC_TOKEN_PREFIX auto_kernel_specs
#define ANNC_BUILTIN_KERNEL_SPECS_FILE "tests/kernels/auto_kernel_specs.inc"
#include "Kernel/BuiltinKernelSpecDeclare.h"
#include "tests/kernels/auto_kernel_impls.cpp"
#define ANNC_SPEC_TOKEN_PREFIX auto_kernel_specs
#define ANNC_BUILTIN_KERNEL_SPECS_FILE "tests/kernels/auto_kernel_specs.inc"
#include "Kernel/BuiltinKernelSpecDefine.h"
GENEOF

cat > "$AUTO_REGISTRY_GEN" <<'GENEOF'
#define ANNC_SPEC_TOKEN_PREFIX auto_kernel_specs
#define ANNC_BUILTIN_KERNEL_SPECS_FILE "tests/kernels/auto_kernel_specs.inc"
#include "Kernel/BuiltinKernelSpecRegister.h"
GENEOF

echo "=== Building kernel_registry_test ==="
echo "Project root: $PROJECT_ROOT"
echo "Build dir: $BUILD_DIR"
echo "Kernel lib: $KERNEL_LIB"
echo "Builtin lib: $BUILTIN_LIB"
echo "Threadpool lib: $THREADPOOL_LIB"

g++ -std=c++17 \
    -I"$PROJECT_ROOT" \
    -I"$PROJECT_ROOT/annc/include" \
    -I"$BUILD_DIR/annc/include" \
    -I"$PROJECT_ROOT/third_party/json/include" \
    -pthread \
    -rdynamic \
    -o "$TEST_BIN_DIR/kernel_registry_test" \
    "$SCRIPT_DIR/kernel_registry_test.cpp" \
    "$AUTO_WRAPPER_GEN" \
    "$AUTO_REGISTRY_GEN" \
    "$KERNEL_LIB" \
    -ldl

echo "=== Build successful ==="

echo "=== Building builtin_shared_spec_test ==="
g++ -std=c++17 \
    -I"$PROJECT_ROOT" \
    -I"$PROJECT_ROOT/annc/include" \
    -I"$BUILD_DIR/annc/include" \
    -I"$PROJECT_ROOT/third_party/json/include" \
    -pthread \
    -rdynamic \
    -o "$TEST_BIN_DIR/builtin_shared_spec_test" \
    "$SCRIPT_DIR/builtin_shared_spec_test.cpp" \
    -Wl,--start-group \
    "$KERNEL_LIB" \
    -Wl,--whole-archive "$BUILTIN_LIB" -Wl,--no-whole-archive \
    "$THREADPOOL_LIB" \
    -Wl,--end-group \
    -ldl

echo "=== Build successful ==="

echo "=== Building customcall_symbol_resolution_stub ==="
g++ -std=c++17 \
    -I"$PROJECT_ROOT" \
    -I"$PROJECT_ROOT/annc/include" \
    -I"$BUILD_DIR/annc/include" \
    -I"$PROJECT_ROOT/third_party/json/include" \
    -pthread \
    -o "$TEST_BIN_DIR/customcall_symbol_resolution_stub" \
    "$SCRIPT_DIR/customcall_symbol_resolution_stub.cpp" \
    "$KERNEL_LIB"

echo "=== Build successful ==="
echo ""
echo "=== Running kernel_registry_test ==="
"$TEST_BIN_DIR/kernel_registry_test"

echo ""
echo "=== Running builtin_shared_spec_test ==="
"$TEST_BIN_DIR/builtin_shared_spec_test"

echo ""
echo "=== Running customcall_symbol_resolution_stub ==="
"$TEST_BIN_DIR/customcall_symbol_resolution_stub"

echo ""
echo "=== Test completed ==="
