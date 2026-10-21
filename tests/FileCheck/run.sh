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
FILTER='grep -vE "^\[ANNC "'

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

    temp_file="$TMP_DIR/test-$file_index.tmp"
    file_index=$((file_index + 1))
    file_failed=0

    for run in "${runs[@]}"; do
        run="${run//annc-opt/$ANNCOPT}"
        run="${run//annc-asm/$ANNASM}"
        run="${run//annc-fusion-metadata/$ANNCFUSIONMETADATA}"
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
