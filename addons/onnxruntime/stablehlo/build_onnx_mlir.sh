#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
ONNX_SOURCE="${REPO_ROOT}/third-party/onnx-mlir"
BUILD_ROOT="${ANNC_STABLEHLO_BUILD_ROOT:-${REPO_ROOT}/build/third-party}"
JOBS="${ANNC_ONNX_MLIR_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')}"
CMAKE_BIN="${CMAKE:-cmake}"
NINJA_BIN="${NINJA:-ninja}"

die() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $0 [options]

Build ONNX-MLIR and StableHLO tools using the LLVM/MLIR installed for ANNC.

Options:
  --build-root <path>  Root for the ONNX-MLIR build directory
  --jobs <count>       Parallel build jobs (default: detected CPU count)
  -h, --help           Show this help

Environment:
  ANNC_STABLEHLO_BUILD_ROOT   Default: build/third-party
  ANNC_ONNX_MLIR_BUILD_DIR    Default: <build-root>/onnx-mlir-build
  ANNC_ONNX_MLIR_BUILD_JOBS
  ANNC_LLVM_BIN, LLVM_DIR, MLIR_DIR
  CC, CXX, CMAKE, NINJA

Requires the LLVM/MLIR 20.1.x development packages (verified with 20.1.8),
git, CMake, Ninja, GNU tar, sha256sum, mktemp, sed, a C/C++ compiler, Python
development headers, and the system protobuf compiler/library.
EOF
}

while (($#)); do
  case "$1" in
    --build-root)
      [[ -n "${2:-}" ]] || die "$1 requires a value"
      BUILD_ROOT="$2"
      shift 2
      ;;
    --jobs)
      [[ -n "${2:-}" ]] || die "$1 requires a value"
      JOBS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help for usage)"
      ;;
  esac
done

[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
for command in git "${NINJA_BIN}" "${CMAKE_BIN}" tar sha256sum mktemp sed; do
  command -v "${command}" >/dev/null 2>&1 || die "required command not found: ${command}"
done
tar --sort=name --version >/dev/null 2>&1 || die "tar must support --sort=name (GNU tar)"
[[ -f "${ONNX_SOURCE}/third_party/stablehlo/CMakeLists.txt" && -f "${ONNX_SOURCE}/third_party/onnx/CMakeLists.txt" ]] ||
  die "initialize the dependencies with: git submodule update --init --recursive third-party/onnx-mlir"

if [[ -z "${LLVM_DIR:-}" ]]; then
  if [[ -z "${ANNC_LLVM_BIN:-}" && -x /opt/openEuler/llvm-toolset-20/root/usr/bin/llvm-config ]]; then
    ANNC_LLVM_BIN=/opt/openEuler/llvm-toolset-20/root/usr/bin
  fi
  LLVM_CONFIG="${ANNC_LLVM_BIN:+${ANNC_LLVM_BIN}/}llvm-config"
  LLVM_DIR="$("${LLVM_CONFIG}" --cmakedir)" ||
    die "set ANNC_LLVM_BIN or LLVM_DIR to the toolchain used to build ANNC"
fi
MLIR_DIR="${MLIR_DIR:-$(dirname "${LLVM_DIR}")/mlir}"
[[ -f "${LLVM_DIR}/LLVMConfig.cmake" && -f "${MLIR_DIR}/MLIRConfig.cmake" ]] ||
  die "LLVM_DIR and MLIR_DIR must point to the installed LLVM/MLIR CMake packages"
LLVM_DIR="$(cd "${LLVM_DIR}" && pwd -P)"
MLIR_DIR="$(cd "${MLIR_DIR}" && pwd -P)"
LLVM_VERSION="$(sed -n 's/^set(PACKAGE_VERSION "\([^"]*\)")$/\1/p' \
  "${LLVM_DIR}/LLVMConfigVersion.cmake" 2>/dev/null)" ||
  die "cannot read LLVMConfigVersion.cmake in ${LLVM_DIR}"
[[ "${LLVM_VERSION}" =~ ^20\.1\.[0-9]+$ ]] ||
  die "ONNX StableHLO requires LLVM/MLIR 20.1.x. Found LLVM ${LLVM_VERSION:-unknown} in ${LLVM_DIR}. \
Use openEuler llvm-toolset-20 or set ANNC_LLVM_BIN or LLVM_DIR to a compatible installation."

ONNX_BUILD="${ANNC_ONNX_MLIR_BUILD_DIR:-${BUILD_ROOT}/onnx-mlir-build}"
mkdir -p "${ONNX_BUILD}"
ONNX_BUILD="$(cd "${ONNX_BUILD}" && pwd -P)"
PREPARED_SOURCE="${ONNX_BUILD}/annc-source"
PATCH="${REPO_ROOT}/patches/onnx-mlir/llvm-20.1.patch"
TEMP="$(mktemp -d "${ONNX_BUILD}/.annc-source.XXXXXX")"
trap 'rm -rf -- "${TEMP}"' EXIT

# Keep the checked-out submodules intact and reuse an unchanged source copy.
tar --sort=name --exclude=.git --exclude=./build -cf "${TEMP}/source.tar" -C "${ONNX_SOURCE}" .
SOURCE_KEY="$({ sha256sum < "${TEMP}/source.tar"; sha256sum < "${PATCH}"; } | sha256sum)"
if [[ ! -f "${PREPARED_SOURCE}/.annc-source-key" ]] ||
   [[ "$(cat "${PREPARED_SOURCE}/.annc-source-key")" != "${SOURCE_KEY}" ]]; then
  mkdir "${TEMP}/source"
  tar -xf "${TEMP}/source.tar" -C "${TEMP}/source"
  GIT_CEILING_DIRECTORIES="${TEMP}" "${CMAKE_BIN}" \
    -DANNC_PATCH_FILE="${PATCH}" -DANNC_PATCH_WORKDIR="${TEMP}/source" \
    -P "${REPO_ROOT}/cmake/ApplyPatchIfNeeded.cmake"
  printf '%s\n' "${SOURCE_KEY}" > "${TEMP}/source/.annc-source-key"
  if [[ -e "${PREPARED_SOURCE}" ]]; then
    [[ -f "${PREPARED_SOURCE}/.annc-source-key" ]] ||
      die "source staging path already exists: ${PREPARED_SOURCE}"
    rm -rf -- "${PREPARED_SOURCE}"
  fi
  mv "${TEMP}/source" "${PREPARED_SOURCE}"
fi

printf 'Configuring ONNX-MLIR with LLVM_DIR=%s and MLIR_DIR=%s\n' "${LLVM_DIR}" "${MLIR_DIR}"
"${CMAKE_BIN}" -S "${PREPARED_SOURCE}" -B "${ONNX_BUILD}" \
  -G Ninja -DCMAKE_MAKE_PROGRAM="${NINJA_BIN}" \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR="${LLVM_DIR}" -DMLIR_DIR="${MLIR_DIR}" \
  -DONNX_MLIR_ENABLE_STABLEHLO=ON \
  -DONNX_MLIR_BUILD_TESTS=OFF -DONNX_MLIR_ENABLE_JAVA=OFF \
  -DONNX_MLIR_ENABLE_WERROR=OFF

"${CMAKE_BIN}" --build "${ONNX_BUILD}" --config Release \
  --parallel "${JOBS}" --target onnx-mlir onnx-mlir-opt stablehlo-translate

for tool in onnx-mlir onnx-mlir-opt stablehlo-translate; do
  [[ -x "${ONNX_BUILD}/Release/bin/${tool}" ]] ||
    die "expected tool was not built: ${tool}"
done
printf 'ONNX-MLIR tools are ready in %s/Release/bin\n' "${ONNX_BUILD}"
