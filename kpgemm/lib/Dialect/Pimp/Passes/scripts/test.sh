#!/bin/bash
PASSES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

patternName="$1"
iter="$2"

python3 ${PASSES_DIR}/scripts/test_${patternName}.py "${PASSES_DIR}" > ${PASSES_DIR}/outputs/test_outputs/${patternName}${iter}.txt 2>&1