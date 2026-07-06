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
FILECHECK="${FILECHECK:-$REPO_ROOT/build/_deps/llvm-build/bin/FileCheck}"
FILTER='grep -vE "^\[ANNC |^this is |^Registered kernel|^ANNC: Registering|^ANNC: Populating|^outputValues"'

for t in "$ANNCOPT" "$ANNASM" "$FILECHECK"; do
    [ -x "$t" ] || { echo "ERROR: $t 不存在或不可执行" >&2; exit 1; }
done

if [ $# -gt 0 ]; then
    FILES=$(find "$SCRIPT_DIR/$1" -name '*.mlir' -not -path '*/Inputs/*' 2>/dev/null)
else
    FILES=$(find "$SCRIPT_DIR" -name '*.mlir' -not -path '*/Inputs/*' | sort)
fi

passed=0
failed=0

for f in $FILES; do
    name=$(realpath --relative-to="$REPO_ROOT" "$f")
    printf '%-60s ' "$name"

    run=$(grep -m1 '^// RUN:' "$f" 2>/dev/null | sed 's/^\/\/ RUN: //')
    [ -z "$run" ] && { echo SKIP; continue; }

    run="${run//%s/$f}"
    run="${run// annc-opt / $ANNCOPT }"
    run="${run// annc-asm / $ANNASM }"

    run="${run// | FileCheck / | $FILTER | FileCheck }"
    run="${run// FileCheck / $FILECHECK }"

    if eval "$run" > /dev/null 2>&1; then
        echo PASS
        passed=$((passed + 1))
    else
        echo FAIL
        eval "$run" 2>&1 || true
        echo "---"
        failed=$((failed + 1))
    fi
done

echo "---"
echo "Passed: $passed  Failed: $failed"
[ "$failed" -eq 0 ] || exit 1