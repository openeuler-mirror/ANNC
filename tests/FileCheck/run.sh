#!/bin/bash
# FileCheck —— ANNC pass pipeline
#
# 用法:
#   bash tests/mlir/run.sh                  # 跑全部
#   bash tests/mlir/run.sh Dialect/Atir     # 跑子目录

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ANNCOPT="${ANNCOPT:-$REPO_ROOT/build/bin/annc-opt}"
ANNASM="${ANNASM:-$REPO_ROOT/build/bin/annc-asm}"
MLIROPT="${MLIROPT:-$REPO_ROOT/build/_deps/llvm-build/bin/mlir-opt}"
FILECHECK="${FILECHECK:-$REPO_ROOT/build/_deps/llvm-build/bin/FileCheck}"
FILTER='grep -vE "^\[ANNC |^this is |^Registered kernel|^ANNC: Registering|^ANNC: Populating|^outputValues"'

for t in "$ANNCOPT" "$ANNASM" "$MLIROPT" "$FILECHECK"; do
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
        run="${run//mlir-opt/$MLIROPT}"
        run="${run// | FileCheck / | $FILTER | FileCheck }"
        run="${run//FileCheck/$FILECHECK}"
        run="${run//%s/$f}"
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
