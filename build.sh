#!/bin/bash

# Default configuration
BUILD_TYPE="Release"
INSTALL_PREFIX="${PWD}/install"
ENABLE_LIBCXX="OFF"
ENABLE_ASSERTIONS="ON"
ENABLE_CONSTANT_FOLDING="OFF"
ENABLE_KDNN_ADAPTOR="ON"
ENABLE_COVERAGE="OFF"
KDNN_SOURCE="LOCAL"
KDNN_DIR="${PWD}/third_party/KDNN"
KDNN_LIB_VARIANT="sve-threadpool"
C_COMPILER="${CC:-gcc}"
CXX_COMPILER="${CXX:-g++}"
PYTHON="${PYTHON:-python3}"
INSTALL_DEPS="YES"
REGEN_TF_PROTOS="NO"

# Internal flags used to detect whether the user explicitly passed certain
# options on the command line.  These are not user-tunable defaults.
_user_kdnn_dir_set=""
_user_kdnn_lib_variant_set=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    --install-prefix)
      INSTALL_PREFIX="$2"
      shift 2
      ;;
    --enable-libcxx)
      ENABLE_LIBCXX="ON"
      shift
      ;;
    --disable-assertions)
      ENABLE_ASSERTIONS="OFF"
      shift
      ;;
    --enable-constant-folding)
      ENABLE_CONSTANT_FOLDING="ON"
      shift
      ;;
    --enable-kdnn-adaptor)
      ENABLE_KDNN_ADAPTOR="ON"
      shift
      ;;
    --disable-kdnn-adaptor)
      ENABLE_KDNN_ADAPTOR="OFF"
      shift
      ;;
    --kdnn-source)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --kdnn-source requires LOCAL, REMOTE, or RELEASE" >&2
        exit 1
      fi
      KDNN_SOURCE="$2"
      shift 2
      ;;
    --kdnn-dir|--annc-kdnn-dir)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: $1 requires a path" >&2
        exit 1
      fi
      KDNN_DIR="$2"
      _user_kdnn_dir_set="1"
      shift 2
      ;;
    --kdnn-lib-variant|--annc-kdnn-lib-variant)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: $1 requires a variant" >&2
        exit 1
      fi
      KDNN_LIB_VARIANT="$2"
      _user_kdnn_lib_variant_set="1"
      shift 2
      ;;
    --clean)
      CLEAN_BUILD="YES"
      shift
      ;;
    --no-install-deps)
      INSTALL_DEPS="NO"
      shift
      ;;
    --regen-tf-protos)
      REGEN_TF_PROTOS="YES"
      shift
      ;;
    --coverage)
      ENABLE_COVERAGE="ON"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --build-type [Debug|Release|RelWithDebInfo]  Set build type (default: Release)"
      echo "  --install-prefix <path>       Set installation prefix (default: ./install)"
      echo "  --enable-libcxx               Enable libc++"
      echo "  --disable-assertions          Disable assertions"
      echo "  --enable-constant-folding     Enable constant folding and KDNN packed-B support (default: OFF)"
      echo "  --enable-kdnn-adaptor         Build builtin KDNN adaptor kernels (default: ON)"
      echo "  --disable-kdnn-adaptor        Disable builtin KDNN adaptor kernels"
      echo "  --kdnn-source [LOCAL|REMOTE|RELEASE]  KDNN source (default: LOCAL)"
      echo "  --kdnn-dir <path>             Local KDNN root, only valid with --kdnn-source LOCAL (default: ./third_party/KDNN)"
      echo "  --kdnn-lib-variant <variant>  KDNN library variant for RELEASE mode (default: ${KDNN_LIB_VARIANT})"
      echo "  --clean                       Clean build directory before build (forces full reconfigure)"
      echo "  --no-install-deps             Skip automatic pip install of missing Python deps"
      echo "  --regen-tf-protos             Regenerate minimal TensorFlow protobuf sources"
      echo "  --coverage                    Enable code coverage (gcovr; auto-installed if missing)"
      echo "  -h, --help                    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ "${KDNN_SOURCE}" != "LOCAL" && "${KDNN_SOURCE}" != "REMOTE" && "${KDNN_SOURCE}" != "RELEASE" ]]; then
  echo "ERROR: --kdnn-source must be LOCAL, REMOTE, or RELEASE, got '${KDNN_SOURCE}'" >&2
  exit 1
fi

if [[ "${KDNN_SOURCE}" == "RELEASE" ]]; then
  for cmd in unzip rpm2cpio cpio; do
    if ! command -v "${cmd}" &>/dev/null; then
      echo "ERROR: RELEASE mode requires '${cmd}' but it is not installed." >&2
      echo "       On openEuler, install it with: sudo yum install -y unzip rpm cpio" >&2
      exit 1
    fi
  done
fi

if [[ "${KDNN_SOURCE}" == "RELEASE" && -n "${_user_kdnn_dir_set}" ]]; then
  echo "ERROR: --kdnn-dir is only valid with --kdnn-source LOCAL." >&2
  exit 1
fi

if [[ "${KDNN_SOURCE}" != "RELEASE" && -n "${_user_kdnn_lib_variant_set}" ]]; then
  echo "ERROR: --kdnn-lib-variant is only valid with --kdnn-source RELEASE." >&2
  exit 1
fi

if [[ -n "${_user_kdnn_lib_variant_set}" && \
      "${KDNN_LIB_VARIANT}" != "sve-threadpool" && \
      "${KDNN_LIB_VARIANT}" != "sve-omp" && \
      "${KDNN_LIB_VARIANT}" != "sve2-threadpool" && \
      "${KDNN_LIB_VARIANT}" != "sve2-omp" ]]; then
  echo "ERROR: --kdnn-lib-variant must be one of: sve-threadpool, sve-omp, sve2-threadpool, sve2-omp, got '${KDNN_LIB_VARIANT}'" >&2
  exit 1
fi

# Coverage note: keep the existing build type (forcing Debug would recompile
# LLVM). Line coverage is accurate in Release; branch coverage is approximate.
# The gcovr dependency is handled below alongside the other Python deps.
if [ "${ENABLE_COVERAGE}" == "ON" ] && [ "${BUILD_TYPE}" == "Release" ]; then
  echo "NOTE: --coverage with Release build: line coverage accurate, branch coverage approximate."
  echo "      For precise branch coverage use a clean Debug build: rm -rf build && ./build.sh --coverage --build-type Debug"
fi

# Check for ninja early so we fail with a clear message before CMake runs.
if ! command -v ninja >/dev/null 2>&1; then
  echo "ERROR: ninja is required but not found in PATH." >&2
  echo "       Install it with one of the following commands:" >&2
  echo "         sudo yum install ninja-build" >&2
  echo "         sudo dnf install ninja-build" >&2
  echo "         sudo apt-get install ninja-build" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Python dependency helpers
# -----------------------------------------------------------------------------

# Check whether a Python module can be imported.
# On failure, prints the import error to stderr so callers can diagnose it.
python_module_available() {
  local output
  output=$("${PYTHON}" -c "import $1" 2>&1)
  if [ $? -ne 0 ]; then
    echo "${output}" >&2
    return 1
  fi
  return 0
}

# Get the CMake directory for pybind11.
get_pybind11_dir() {
  "${PYTHON}" -c "import pybind11; print(pybind11.get_cmake_dir())"
}

# Get the CMake directory for nanobind.
get_nanobind_dir() {
  "${PYTHON}" -c "import nanobind, os; print(os.path.join(os.path.dirname(nanobind.__file__), 'cmake'))"
}

# Detect whether the selected Python interpreter is inside a virtualenv.
# Checks sys.prefix vs sys.base_prefix and common environment indicators
# (VIRTUAL_ENV, CONDA_PREFIX) to avoid misclassifying conda/pyenv envs.
is_virtualenv() {
  "${PYTHON}" -c "import sys, os; venv = sys.prefix != sys.base_prefix; sys.exit(0 if (venv or os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_PREFIX')) else 1)" 2>/dev/null
}

# Install a Python package via pip. Use --user when not inside a virtualenv to
# avoid permission issues.
pip_install() {
  local pip_args=("$@")
  echo "Installing Python package(s): ${pip_args[*]}"
  echo "Use --no-install-deps to skip automatic installation."
  if is_virtualenv; then
    # Inside a virtualenv/conda env: install into the active environment.
    "${PYTHON}" -m pip install "${pip_args[@]}"
  else
    # System Python: install into the user site-packages to avoid permission errors.
    "${PYTHON}" -m pip install --user "${pip_args[@]}"
  fi
}

# -----------------------------------------------------------------------------
# Dependency checks
# -----------------------------------------------------------------------------

DEP_ERRORS=0

echo "Checking Python environment (${PYTHON})..."

# pybind11 and nanobind are lightweight build-time deps. Auto-install if missing.
for dep in pybind11 nanobind; do
  if ! python_module_available "${dep}"; then
    if [ "${INSTALL_DEPS}" == "YES" ]; then
      echo "Missing Python package '${dep}'. Attempting to install it automatically..."
      if ! pip_install "${dep}"; then
        echo "ERROR: Failed to install '${dep}' via pip." >&2
        echo "       Please install it manually, e.g.: ${PYTHON} -m pip install ${dep}" >&2
        DEP_ERRORS=$((DEP_ERRORS + 1))
      elif ! python_module_available "${dep}"; then
        echo "ERROR: '${dep}' was installed but cannot be imported by ${PYTHON}." >&2
        DEP_ERRORS=$((DEP_ERRORS + 1))
      fi
    else
      echo "ERROR: Required Python package '${dep}' is not installed." >&2
      echo "       Install it with: ${PYTHON} -m pip install ${dep}" >&2
      echo "       Or rerun without --no-install-deps to auto-install." >&2
      DEP_ERRORS=$((DEP_ERRORS + 1))
    fi
  fi
done

# gcovr is the coverage report tool, only needed with --coverage.
if [ "${ENABLE_COVERAGE}" == "ON" ]; then
  if ! python_module_available gcovr; then
    if [ "${INSTALL_DEPS}" == "YES" ]; then
      echo "Missing Python package 'gcovr'. Attempting to install it automatically..."
      if ! pip_install gcovr; then
        echo "ERROR: Failed to install 'gcovr' via pip." >&2
        echo "       Please install it manually, e.g.: ${PYTHON} -m pip install gcovr" >&2
        DEP_ERRORS=$((DEP_ERRORS + 1))
      elif ! python_module_available gcovr; then
        echo "ERROR: 'gcovr' was installed but cannot be imported by ${PYTHON}." >&2
        DEP_ERRORS=$((DEP_ERRORS + 1))
      fi
    else
      echo "ERROR: --coverage requires 'gcovr' which is not installed." >&2
      echo "       Install it with: ${PYTHON} -m pip install gcovr" >&2
      echo "       Or rerun without --no-install-deps to auto-install." >&2
      DEP_ERRORS=$((DEP_ERRORS + 1))
    fi
  fi
  # pip --user installs the gcovr CLI to the user bin dir which may not be on
  # PATH. CMake's find_program needs the CLI, so expose it for the cmake call.
  if ! command -v gcovr >/dev/null 2>&1; then
    USER_BIN="$("${PYTHON}" -c "import site; print(site.USER_BASE)")/bin"
    if [ -x "${USER_BIN}/gcovr" ]; then
      export PATH="${USER_BIN}:${PATH}"
    fi
  fi
  if ! command -v gcovr >/dev/null 2>&1; then
    echo "ERROR: gcovr is importable but its CLI could not be found on PATH." >&2
    echo "       Add the pip user bin to PATH or install gcovr system-wide." >&2
    DEP_ERRORS=$((DEP_ERRORS + 1))
  fi
fi

# TensorFlow is large and version-sensitive; only verify presence, do not auto-install.
if ! python_module_available "tensorflow"; then
  echo "ERROR: TensorFlow cannot be imported by ${PYTHON}." >&2
  echo "       ANNC requires TensorFlow at configure time. Install compatible deps with:" >&2
  echo "         ${PYTHON} -m pip install -r requirements.txt" >&2
  echo "       If the error mentions NumPy 2.x, downgrade with:" >&2
  echo "         ${PYTHON} -m pip install 'numpy<2'" >&2
  DEP_ERRORS=$((DEP_ERRORS + 1))
fi

if [ ${DEP_ERRORS} -ne 0 ]; then
  echo "ERROR: ${DEP_ERRORS} Python dependency issue(s) found. Aborting build." >&2
  exit 1
fi

# At this point the required modules must be importable.
PYBIND11_DIR=$(get_pybind11_dir) || {
  echo "ERROR: Unable to determine pybind11 CMake directory." >&2
  exit 1
}
NANOBIND_DIR=$(get_nanobind_dir) || {
  echo "ERROR: Unable to determine nanobind CMake directory." >&2
  exit 1
}

# -----------------------------------------------------------------------------
# TensorFlow proto generation
# -----------------------------------------------------------------------------

ensure_tf_protos() {
  local gen_dir="${PWD}/tf_protos_minimal/gen_code"
  local required_headers=(
    "${gen_dir}/tensorflow/core/framework/graph.pb.h"
    "${gen_dir}/tensorflow/core/framework/node_def.pb.h"
    "${gen_dir}/tensorflow/core/framework/tensor.pb.h"
    "${gen_dir}/tensorflow/core/protobuf/saved_model.pb.h"
  )

  local missing="NO"
  if [ "${REGEN_TF_PROTOS}" == "YES" ]; then
    missing="YES"
  else
    for header in "${required_headers[@]}"; do
      if [ ! -f "${header}" ]; then
        missing="YES"
        break
      fi
    done
  fi

  if [ "${missing}" == "NO" ]; then
    echo "TensorFlow protobuf sources already generated (${gen_dir})"
    return 0
  fi

  echo "Generating minimal TensorFlow protobuf sources..."
  bash "${PWD}/tf_protos_minimal.sh" -d "${PWD}"
}

ensure_tf_protos

# -----------------------------------------------------------------------------
# Create build directory (preserve LLVM build if exists)
# -----------------------------------------------------------------------------

if [ -d "build" ] && [ "$CLEAN_BUILD" != "YES" ]; then
  echo "Using existing build directory (use --clean to force full rebuild)"
  cd build || exit 1
else
  echo "Creating new build directory..."
  rm -rf build && mkdir build && cd build || exit 1
fi

# -----------------------------------------------------------------------------
# Run CMake
# -----------------------------------------------------------------------------

# Read a value from CMakeCache.txt (format: KEY:TYPE=VALUE).
get_cache_value() {
  grep "^$1:" CMakeCache.txt 2>/dev/null | tail -n1 | cut -d= -f2-
}

# Check that key build options still match an existing cache.  CMake does not
# automatically reconfigure when -D arguments change; silently reusing the old
# cache would ignore user-requested changes (e.g. --kdnn-source RELEASE on a
# tree previously configured with LOCAL).  If a mismatch is found, force an
# explicit --clean reconfigure.
check_cache_consistency() {
  [ -f "CMakeCache.txt" ] || return 0

  local key old_val new_val
  local -a mismatches=()
  local -a cache_checks=(
    "CMAKE_BUILD_TYPE:${BUILD_TYPE}"
    "CMAKE_INSTALL_PREFIX:${INSTALL_PREFIX}"
    "CMAKE_C_COMPILER:$(command -v ${C_COMPILER})"
    "CMAKE_CXX_COMPILER:$(command -v ${CXX_COMPILER})"
    "PYTHON_EXECUTABLE:$(command -v ${PYTHON})"
    "LLVM_ENABLE_LIBCXX:${ENABLE_LIBCXX}"
    "LLVM_ENABLE_ASSERTIONS:${ENABLE_ASSERTIONS}"
    "ANNC_ENABLE_CONSTANT_FOLDING:${ENABLE_CONSTANT_FOLDING}"
    "ANNC_ENABLE_KDNN_ADAPTOR:${ENABLE_KDNN_ADAPTOR}"
    "ANNC_ENABLE_COVERAGE:${ENABLE_COVERAGE}"
    "ANNC_KDNN_SOURCE:${KDNN_SOURCE}"
    "ANNC_KDNN_DIR:${KDNN_DIR}"
    "ANNC_KDNN_LIB_VARIANT:${KDNN_LIB_VARIANT}"
  )

  for entry in "${cache_checks[@]}"; do
    key="${entry%%:*}"
    new_val="${entry#*:}"
    old_val=$(get_cache_value "${key}")
    if [ -n "${old_val}" ] && [ "${old_val}" != "${new_val}" ]; then
      mismatches+=("  ${key}: cache='${old_val}' != requested='${new_val}'")
    fi
  done

  if [ ${#mismatches[@]} -ne 0 ]; then
    echo "ERROR: Build option(s) changed but an existing CMakeCache.txt was found." >&2
    printf '%s\n' "${mismatches[@]}" >&2
    echo "       Use --clean to reconfigure with the new options." >&2
    exit 1
  fi
}

# 增量构建: 若 build/CMakeCache.txt 已存在且参数一致, 跳过 cmake 重新配置,
# 直接复用既有 cache。避免对已存在 cache 重复传 -D 触发的 cache 变量时序问题
# (如 CMAKE_BUILD_TYPE / pybind11_DIR 在 LLVM add_subdirectory 作用域不可见)。
# 若参数不一致, 必须先使用 --clean 重新配置。
SKIP_CMAKE="NO"
if [ -f "CMakeCache.txt" ]; then
  check_cache_consistency
  SKIP_CMAKE="YES"
fi

if [ "${SKIP_CMAKE}" == "YES" ]; then
  echo "Skipping CMake configuration (reusing existing CMakeCache.txt)"
  echo "Use --clean to force a full reconfigure."
else
  echo "Running CMake with the following configuration:"
  echo "  Build Type: ${BUILD_TYPE}"
  echo "  Install Prefix: ${INSTALL_PREFIX}"
  echo "  C Compiler: ${C_COMPILER}"
  echo "  C++ Compiler: ${CXX_COMPILER}"
  echo "  Python: ${PYTHON}"
  echo "  Constant Folding: ${ENABLE_CONSTANT_FOLDING}"
  echo "  KDNN Adaptor: ${ENABLE_KDNN_ADAPTOR}"
  echo "  Coverage: ${ENABLE_COVERAGE}"
  echo "  KDNN Source: ${KDNN_SOURCE}"
  if [[ "${KDNN_SOURCE}" == "LOCAL" ]]; then
    echo "  KDNN Dir: ${KDNN_DIR}"
  elif [[ "${KDNN_SOURCE}" == "RELEASE" ]]; then
    echo "  KDNN Lib Variant: ${KDNN_LIB_VARIANT}"
  fi

  cmake .. \
    -G Ninja \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DLLVM_ENABLE_LIBCXX="${ENABLE_LIBCXX}" \
    -DLLVM_ENABLE_ASSERTIONS="${ENABLE_ASSERTIONS}" \
    -DANNC_ENABLE_CONSTANT_FOLDING="${ENABLE_CONSTANT_FOLDING}" \
    -DANNC_ENABLE_KDNN_ADAPTOR="${ENABLE_KDNN_ADAPTOR}" \
    -DANNC_ENABLE_COVERAGE="${ENABLE_COVERAGE}" \
    -DANNC_KDNN_SOURCE="${KDNN_SOURCE}" \
    -DANNC_KDNN_DIR="${KDNN_DIR}" \
    -DANNC_KDNN_LIB_VARIANT="${KDNN_LIB_VARIANT}" \
    -DKDNN_DIR="${KDNN_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_C_COMPILER="$(command -v ${C_COMPILER})" \
    -DCMAKE_CXX_COMPILER="$(command -v ${CXX_COMPILER})" \
    -DCMAKE_CXX_FLAGS="-fPIC" \
    -DPYTHON_EXECUTABLE="$(command -v ${PYTHON})" \
    -Dpybind11_DIR="${PYBIND11_DIR}" \
    -Dnanobind_DIR="${NANOBIND_DIR}"

  if [ $? -ne 0 ]; then
      echo "CMake configuration failed, aborting build"
      exit 1
  fi
fi

# Build and install
echo "Starting build with $(nproc) parallel jobs..."
ninja -j$(nproc)
if [ $? -ne 0 ]; then
    echo "Build failed, aborting installation"
    exit 1
fi

ninja install
echo "Build and installation completed successfully"
