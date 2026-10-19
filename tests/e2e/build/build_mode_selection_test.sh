#!/usr/bin/env bash
# Check the mutually exclusive framework-selection modes without configuring
# or building anything.
#
#   ANNC_SRC=<source tree> tests/e2e/build/build_mode_selection_test.sh
set -euo pipefail

SRC=${ANNC_SRC:-$(cd "$(dirname "$0")/../../.." && pwd)}
BUILD="${SRC}/build.sh"

expect() {
  local expected=$1
  shift
  local out
  out=$("${BUILD}" --dry-run "$@")
  if ! grep -qF "${expected}" <<<"${out}"; then
    echo "FAIL: build.sh --dry-run $* did not contain '${expected}'" >&2
    echo "${out}" >&2
    exit 1
  fi
}

expect_fail() {
  if "${BUILD}" --dry-run "$@" >/dev/null 2>&1; then
    echo "FAIL: build.sh --dry-run $* should have failed" >&2
    exit 1
  fi
}

expect "Framework selection: tf" --enable-tf
expect "ANNC_ENABLE_TENSORFLOW=ON" --enable-tf
expect "ANNC_ENABLE_TENSORFLOW_ADDON=ON" --enable-tf
expect "ANNC_ENABLE_ONNX=OFF" --enable-tf
expect "ANNC_ENABLE_ONNXRUNTIME_EP=OFF" --enable-tf

expect "Framework selection: ort" --enable-ort
expect "ANNC_ENABLE_TENSORFLOW=OFF" --enable-ort
expect "ANNC_ENABLE_TENSORFLOW_ADDON=OFF" --enable-ort
expect "ANNC_ENABLE_ONNX=ON" --enable-ort
expect "ANNC_ENABLE_ONNXRUNTIME_EP=ON" --enable-ort
# --enable-ort-ep is the compatibility alias.
expect "Framework selection: ort" --enable-ort-ep
expect "ANNC_ENABLE_TENSORFLOW=OFF" --enable-ort-ep
expect "ANNC_ENABLE_TENSORFLOW_ADDON=OFF" --enable-ort-ep
expect "ANNC_ENABLE_ONNX=ON" --enable-ort-ep
expect "ANNC_ENABLE_ONNXRUNTIME_EP=ON" --enable-ort-ep

expect "Framework selection: all" --all
expect "ANNC_ENABLE_TENSORFLOW=ON" --all
expect "ANNC_ENABLE_TENSORFLOW_ADDON=ON" --all
expect "ANNC_ENABLE_ONNX=ON" --all
expect "ANNC_ENABLE_ONNXRUNTIME_EP=ON" --all

# Legacy default: TF + ONNX frontend, ORT EP off.
expect "Framework selection: legacy"
expect "ANNC_ENABLE_TENSORFLOW=ON"
expect "ANNC_ENABLE_TENSORFLOW_ADDON=ON"
expect "ANNC_ENABLE_ONNX=ON"
expect "ANNC_ENABLE_ONNXRUNTIME_EP=OFF"

# Mutually exclusive selection and conflicting disable flags.
expect_fail --enable-tf --enable-ort
expect_fail --enable-tf --all
expect_fail --all --disable-onnx
expect_fail --all --disable-tensorflow
expect_fail --enable-tf --disable-tensorflow
expect_fail --enable-ort --disable-onnx

# Consistent fine-grained flags are accepted in a mode.
expect "Framework selection: tf" --enable-tf --disable-onnx
expect "Framework selection: ort" --enable-ort --disable-tensorflow

echo "PASS: framework selection modes"
