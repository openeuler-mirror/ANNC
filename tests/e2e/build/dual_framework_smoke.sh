#!/usr/bin/env bash
# Dual-framework installation smoke test.
#
# Run after a build that enabled both ANNC_ENABLE_TENSORFLOW[_ADDON] and
# ANNC_ENABLE_ONNXRUNTIME_EP.  It verifies the artifact set and the forwarded
# build options without requiring a model or a running predictor.
#
#   ANNC_SRC=<source tree> ANNC_PREFIX=<install prefix> tests/e2e/build/dual_framework_smoke.sh
set -euo pipefail

: "${ANNC_SRC:?set ANNC_SRC to the ANNC source tree}"
: "${ANNC_PREFIX:?set ANNC_PREFIX to the ANNC install prefix}"

required=(
  bin/annc-tf-pipeline
  bin/annc-tf2atir
  bin/annc-converter
  bin/annc-onnx-pipeline
  lib/libannc_optimizer.so
  lib/libannc_fused_op.so
)
for rel in "${required[@]}"; do
  if [[ ! -e "${ANNC_PREFIX}/${rel}" ]]; then
    echo "FAIL: missing ${ANNC_PREFIX}/${rel}" >&2
    exit 1
  fi
done
if [[ ! -e "${ANNC_PREFIX}/lib/libannc_ep.so" &&
      ! -e "${ANNC_PREFIX}/lib64/libannc_ep.so" ]]; then
  echo "FAIL: missing libannc_ep.so in ${ANNC_PREFIX}/lib or lib64" >&2
  exit 1
fi

help_text=$("${ANNC_SRC}/build.sh" --help)
for var in ANNC_EP_ORT_BUILD_TREE ANNC_KERNEL_MLAS \
           ANNC_MLAS_INCLUDE_DIR ANNC_GEMM_ASM_SOURCE_DIR; do
  if ! grep -q "${var}" <<<"${help_text}"; then
    echo "FAIL: build.sh --help does not document ${var}" >&2
    exit 1
  fi
done

cache="${ANNC_SRC}/build/CMakeCache.txt"
if [[ ! -f "${cache}" ]]; then
  echo "FAIL: ${cache} not found (run build.sh first)" >&2
  exit 1
fi
for entry in \
  "ANNC_ENABLE_TENSORFLOW:BOOL=ON" \
  "ANNC_ENABLE_TENSORFLOW_ADDON:BOOL=ON" \
  "ANNC_ENABLE_ONNX:BOOL=ON" \
  "ANNC_ENABLE_ONNXRUNTIME_EP:BOOL=ON" \
  "ANNC_KERNEL_MLAS:BOOL=ON"; do
  if ! grep -q "^${entry}$" "${cache}"; then
    echo "FAIL: cache does not contain ${entry}" >&2
    exit 1
  fi
done

for key in ANNC_EP_ORT_BUILD_TREE ANNC_MLAS_INCLUDE_DIR \
           ANNC_GEMM_ASM_SOURCE_DIR; do
  value=$(grep "^${key}:" "${cache}" | tail -n1 | cut -d= -f2-)
  if [[ -z "${value}" ]]; then
    echo "FAIL: cache has no value for ${key}" >&2
    exit 1
  fi
done

echo "PASS: dual-framework install tree and build switches"
