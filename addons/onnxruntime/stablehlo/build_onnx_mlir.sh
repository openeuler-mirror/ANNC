#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
ONNX_SOURCE="${REPO_ROOT}/third-party/onnx-mlir"
BUILD_ROOT="${ANNC_STABLEHLO_BUILD_ROOT:-${REPO_ROOT}/build/third-party}"
LLVM_SOURCE="${ANNC_LLVM_SOURCE_DIR:-${BUILD_ROOT}/llvm-project-src}"
LLVM_BUILD="${ANNC_LLVM_BUILD_DIR:-${BUILD_ROOT}/llvm-build}"
ONNX_BUILD="${ANNC_ONNX_MLIR_BUILD_DIR:-${BUILD_ROOT}/onnx-mlir-build}"
JOBS="${ANNC_ONNX_MLIR_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')}"
GENERATOR=Ninja
CMAKE_BIN="${CMAKE:-cmake}"
LLVM_REPOSITORY="https://github.com/llvm/llvm-project.git"

die() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $0 [options]

Build the pinned LLVM/MLIR and ONNX-MLIR/StableHLO toolchains from source.
The source revisions are pinned by the ONNX-MLIR submodule.

Options:
  --build-root <path>  Root for cached LLVM source and build directories
  --llvm-source <path> Use an existing checkout of the required LLVM revision
  --jobs <count>       Parallel build jobs (default: detected CPU count)
  -h, --help           Show this help

Environment:
  ANNC_STABLEHLO_BUILD_ROOT  Default: build/third-party
  ANNC_LLVM_SOURCE_DIR       Default: <build-root>/llvm-project-src
  ANNC_LLVM_BUILD_DIR        Default: <build-root>/llvm-build
  ANNC_ONNX_MLIR_BUILD_DIR   Default: build/third-party/onnx-mlir-build
  ANNC_ONNX_MLIR_BUILD_JOBS
  CC, CXX, CMAKE

Requires git, CMake, Ninja, a C/C++ compiler, Python development headers,
and the system protobuf compiler/library.
EOF
}

need_value() {
  [[ -n "${2:-}" ]] || die "$1 requires a value"
}

while (($#)); do
  case "$1" in
    --build-root)
      need_value "$1" "${2:-}"
      BUILD_ROOT="$2"
      LLVM_SOURCE="${ANNC_LLVM_SOURCE_DIR:-${BUILD_ROOT}/llvm-project-src}"
      LLVM_BUILD="${ANNC_LLVM_BUILD_DIR:-${BUILD_ROOT}/llvm-build}"
      ONNX_BUILD="${ANNC_ONNX_MLIR_BUILD_DIR:-${BUILD_ROOT}/onnx-mlir-build}"
      shift 2
      ;;
    --llvm-source)
      need_value "$1" "${2:-}"
      LLVM_SOURCE="$2"
      shift 2
      ;;
    --jobs)
      need_value "$1" "${2:-}"
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
for command in git ninja rg "${CMAKE_BIN}"; do
  command -v "${command}" >/dev/null 2>&1 || die "required command not found: ${command}"
done
[[ -f "${ONNX_SOURCE}/CMakeLists.txt" ]] ||
  die "initialize ONNX-MLIR with: git submodule update --init --recursive third-party/onnx-mlir"
[[ -f "${ONNX_SOURCE}/third_party/stablehlo/CMakeLists.txt" ]] ||
  die "initialize ONNX-MLIR's nested submodules with: git submodule update --init --recursive third-party/onnx-mlir"

LLVM_REVISION="$(sed -nE 's/^.*git checkout ([0-9a-f]{40}).*$/\1/p' "${ONNX_SOURCE}/utils/clone-mlir.sh")"
[[ "${LLVM_REVISION}" =~ ^[0-9a-f]{40}$ ]] ||
  die "cannot read the pinned LLVM revision from ONNX-MLIR's utils/clone-mlir.sh"
STABLEHLO_LLVM_REVISION="$(tr -d '[:space:]' < "${ONNX_SOURCE}/third_party/stablehlo/build_tools/llvm_version.txt")"
[[ "${STABLEHLO_LLVM_REVISION}" == "${LLVM_REVISION}" ]] ||
  die "ONNX-MLIR and its StableHLO submodule pin different LLVM revisions"

if [[ ! -d "${LLVM_SOURCE}/.git" ]]; then
  [[ ! -e "${LLVM_SOURCE}" ]] || die "LLVM source path exists but is not a git checkout: ${LLVM_SOURCE}"
  mkdir -p "$(dirname "${LLVM_SOURCE}")"
  printf 'Cloning llvm-project source (revision %s)\n' "${LLVM_REVISION}"
  git clone --filter=blob:none --no-checkout "${LLVM_REPOSITORY}" "${LLVM_SOURCE}"
fi

LLVM_SOURCE="$(cd "${LLVM_SOURCE}" && pwd -P)"
LLVM_HEAD="$(git -C "${LLVM_SOURCE}" rev-parse HEAD)"
LLVM_DIRTY="$(git -C "${LLVM_SOURCE}" status --porcelain --untracked-files=no)"
[[ -z "${LLVM_DIRTY}" ]] || die "LLVM source checkout has tracked local changes: ${LLVM_SOURCE}"
if [[ "${LLVM_HEAD}" != "${LLVM_REVISION}" ]]; then
  printf 'Checking out pinned llvm-project revision %s\n' "${LLVM_REVISION}"
  git -C "${LLVM_SOURCE}" fetch --filter=blob:none origin "${LLVM_REVISION}"
  git -C "${LLVM_SOURCE}" checkout --detach "${LLVM_REVISION}"
fi

mkdir -p "${LLVM_BUILD}" "${ONNX_BUILD}"
LLVM_BUILD="$(cd "${LLVM_BUILD}" && pwd -P)"
ONNX_BUILD="$(cd "${ONNX_BUILD}" && pwd -P)"

printf 'Configuring LLVM/MLIR at %s\n' "${LLVM_REVISION}"
"${CMAKE_BIN}" -S "${LLVM_SOURCE}/llvm" -B "${LLVM_BUILD}" \
  -G "${GENERATOR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DMLIR_INCLUDE_EXAMPLES=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_ZLIB=OFF

printf 'Building MLIR configuration tools\n'
"${CMAKE_BIN}" --build "${LLVM_BUILD}" --parallel "${JOBS}" \
  --target MLIRIR mlir-tblgen llvm-config

printf 'Configuring ONNX-MLIR and its pinned StableHLO source\n'
"${CMAKE_BIN}" -S "${ONNX_SOURCE}" -B "${ONNX_BUILD}" \
  -G "${GENERATOR}" \
  -DMLIR_DIR="${LLVM_BUILD}/lib/cmake/mlir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNX_MLIR_ENABLE_STABLEHLO=ON \
  -DONNX_MLIR_BUILD_TESTS=OFF \
  -DONNX_MLIR_ENABLE_JAVA=OFF \
  -DONNX_MLIR_ENABLE_WERROR=OFF

printf 'Discovering LLVM/MLIR libraries required by ONNX-MLIR tools\n'
NINJA_BIN="${NINJA:-ninja}"
command -v "${NINJA_BIN}" >/dev/null 2>&1 || die "Ninja is required to discover ONNX-MLIR link dependencies"
mapfile -t LLVM_LIBRARIES < <(
  "${NINJA_BIN}" -C "${ONNX_BUILD}" -t commands onnx-mlir onnx-mlir-opt stablehlo-translate |
    rg -o --no-filename "${LLVM_BUILD}/lib/lib[A-Za-z0-9_]+\\.a" |
    sed -E 's#^.*/lib([^/]+)\.a$#\1#' |
    sort -u
)
((${#LLVM_LIBRARIES[@]} > 0)) || die "could not determine LLVM/MLIR libraries needed by ONNX-MLIR"
"${CMAKE_BIN}" --build "${LLVM_BUILD}" --parallel "${JOBS}" \
  --target "${LLVM_LIBRARIES[@]}" llc opt MLIROptLib

printf 'Building ONNX-MLIR and StableHLO command-line tools\n'
"${CMAKE_BIN}" --build "${ONNX_BUILD}" --config Release \
  --parallel "${JOBS}" \
  --target onnx-mlir onnx-mlir-opt stablehlo-translate

for tool in onnx-mlir onnx-mlir-opt stablehlo-translate; do
  [[ -x "${ONNX_BUILD}/Release/bin/${tool}" ]] ||
    die "expected tool was not built: ${ONNX_BUILD}/Release/bin/${tool}"
done

printf 'ONNX-MLIR tools are ready in %s/Release/bin\n' "${ONNX_BUILD}"
printf 'For the ANNC CMake addon, set ANNC_ONNX_MLIR_BUILD_DIR=%s\n' "${ONNX_BUILD}"
