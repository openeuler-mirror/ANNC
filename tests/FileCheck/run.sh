#!/bin/bash
# FileCheck —— ANNC pass pipeline
#
# 用法:
#   bash tests/FileCheck/run.sh                  # 跑全部
#   bash tests/FileCheck/run.sh Dialect/Atir     # 跑子目录

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ANNCOPT="${ANNCOPT:-$REPO_ROOT/build/bin/annc-opt}"
ANNASM="${ANNASM:-$REPO_ROOT/build/bin/annc-asm}"
ANNCFUSIONMETADATA="${ANNCFUSIONMETADATA:-$REPO_ROOT/build/bin/annc-fusion-metadata}"
# v2-route tool: only exists in ANNC_ENABLE_TF2ATIR_V2=ON builds; tests
# referencing it are skipped (not failed) when the binary is absent.
ANNCTF2ATIRV2="${ANNCTF2ATIRV2:-$REPO_ROOT/build/bin/annc-stablehlo-sink}"
ANNCEPILOGUEKERNELCOMPILE="${ANNCEPILOGUEKERNELCOMPILE:-${ANNCGEMMKERNELCOMPILE:-$REPO_ROOT/build/bin/annc-gemm-epilogue-kernel-compile}}"
ANNCDRIVER="${ANNCDRIVER:-$REPO_ROOT/build/bin/annc}"

# LLVM 20 tools: ANNC_LLVM_BIN (if exported) wins; otherwise use the toolchain
# directory recorded by CMake at configure time. Never fall back to a bare
# mlir-opt/FileCheck on PATH, which may belong to a different LLVM version.
if [ -z "${ANNC_LLVM_BIN:-}" ] && [ -r "$REPO_ROOT/build/annc_llvm_bin.txt" ]; then
    ANNC_LLVM_BIN="$(cat "$REPO_ROOT/build/annc_llvm_bin.txt")"
fi
if [ -z "${ANNC_LLVM_BIN:-}" ]; then
    echo "ERROR: ANNC_LLVM_BIN is not set and $REPO_ROOT/build/annc_llvm_bin.txt is missing." >&2
    echo "       Configure the build (writes the file) or export ANNC_LLVM_BIN=<llvm-20-bin>." >&2
    exit 1
fi
MLIROPT="${MLIROPT:-$ANNC_LLVM_BIN/mlir-opt}"
FILECHECK="${FILECHECK:-$ANNC_LLVM_BIN/FileCheck}"
# Some RUN lines invoke LLVM's `not` and `split-file` by bare name.
export PATH="$ANNC_LLVM_BIN:$PATH"
# Matches both log prefixes in the tree: the tool-level "[ANNC ..." and the
# Support/Log.h "[annc:<subsystem>][<level>]" (only relevant at info/debug).
FILTER='grep -vE "^\[[Aa][Nn][Nn][Cc][ :]"'

# The AArch64 GEMM pipeline has no valid legacy route when either the V2
# frontend is compiled in or the context ABI switch is enabled. The V2 tool
# may be unavailable in partial builds, so use the build configuration rather
# than its presence as the route selector.
ANNC_V2_BUILD=0
if [ -r "$REPO_ROOT/build/CMakeCache.txt" ] &&
   grep -q '^ANNC_ENABLE_TF2ATIR_V2:BOOL=ON$' "$REPO_ROOT/build/CMakeCache.txt"; then
    ANNC_V2_BUILD=1
elif [ -x "$ANNCTF2ATIRV2" ]; then
    ANNC_V2_BUILD=1
fi
ANNC_CONTEXT_ABI=0
case "${ANNC_ENABLE_LOWERING_CONTEXT_ABI:-}" in
    1|ON|TRUE|YES|on|true|yes) ANNC_CONTEXT_ABI=1 ;;
esac
ANNC_LEGACY_AARCH64_PIPELINE=1
if [ "$ANNC_V2_BUILD" -eq 1 ] || [ "$ANNC_CONTEXT_ABI" -eq 1 ]; then
    ANNC_LEGACY_AARCH64_PIPELINE=0
fi
export ANNC_LEGACY_AARCH64_PIPELINE

for t in "$ANNCOPT" "$ANNASM" "$ANNCFUSIONMETADATA" "$MLIROPT" "$FILECHECK" \
         "$ANNC_LLVM_BIN/not" "$ANNC_LLVM_BIN/split-file"; do
    [ -x "$t" ] || { echo "ERROR: $t 不存在或不可执行" >&2; exit 1; }
done

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

if [ $# -gt 0 ]; then
    FILES=$(find "$SCRIPT_DIR/$1" -name '*.mlir' -not -path '*/Inputs/*' 2>/dev/null)
else
    FILES=$(find "$SCRIPT_DIR" -name '*.mlir' -not -path '*/Inputs/*' | sort)
fi

passed=0
failed=0
file_index=0

for f in $FILES; do
    name=$(realpath --relative-to="$REPO_ROOT" "$f")
    printf '%-60s ' "$name"

    mapfile -t runs < <(sed -n 's/^\/\/ RUN: //p' "$f")
    [ "${#runs[@]}" -eq 0 ] && { echo SKIP; continue; }

    # These fixtures specifically cover the legacy AArch64 route.
    if grep -q '^// REQUIRES: legacy-aarch64-pipeline$' "$f" &&
       [ "$ANNC_LEGACY_AARCH64_PIPELINE" -eq 0 ]; then
        echo SKIP
        continue
    fi

    temp_file="$TMP_DIR/test-$file_index.tmp"
    file_index=$((file_index + 1))
    file_failed=0

    for run in "${runs[@]}"; do
        run="${run//annc-opt/$ANNCOPT}"
        if [[ "$run" == *annc-stablehlo-sink* && ! -x "$ANNCTF2ATIRV2" ]]; then
            echo SKIP
            continue 2
        fi
        run="${run//annc-asm/$ANNASM}"
        run="${run//annc-fusion-metadata/$ANNCFUSIONMETADATA}"
        run="${run//annc-stablehlo-sink/$ANNCTF2ATIRV2}"
        run="${run//annc-gemm-epilogue-kernel-compile/$ANNCEPILOGUEKERNELCOMPILE}"
        run="${run// annc / $ANNCDRIVER }"
        run="${run//mlir-opt/$MLIROPT}"
        run="${run// | FileCheck / | $FILTER | FileCheck }"
        run="${run//FileCheck/$FILECHECK}"
        run="${run//%s/$f}"
        run="${run//%S/$(dirname "$f")}"
        run="${run//%t/$temp_file}"

        if ! eval "$run" > /dev/null 2>&1; then
            echo FAIL
            echo "RUN: $run"
            eval "$run" 2>&1 || true
            echo "---"
            file_failed=1
            break
        fi
    done

    if [ "$file_failed" -eq 0 ]; then
        echo PASS
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
    fi
done

echo "---"
echo "Passed: $passed  Failed: $failed"
[ "$failed" -eq 0 ] || exit 1
