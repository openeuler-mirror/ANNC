#!/bin/bash

# Default configuration
BUILD_TYPE="Release"
INSTALL_PREFIX="${PWD}/install"
ENABLE_LIBCXX="OFF"
ENABLE_ASSERTIONS="ON"
ENABLE_CONSTANT_FOLDING="OFF"
ENABLE_KDNN_ADAPTOR="ON"
KDNN_SOURCE="LOCAL"
KDNN_DIR="${PWD}/third_party/KDNN"
C_COMPILER="${CC:-gcc}"
CXX_COMPILER="${CXX:-g++}"
PYTHON="${PYTHON:-python3}"
INSTALL_DEPS="YES"
REGEN_TF_PROTOS="NO"

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
        echo "ERROR: --kdnn-source requires LOCAL or REMOTE" >&2
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
      echo "  --kdnn-source [LOCAL|REMOTE]  KDNN source (default: LOCAL)"
      echo "  --kdnn-dir <path>             Local KDNN root (default: ./third_party/KDNN)"
      echo "  --clean                       Clean build directory before build"
      echo "  --no-install-deps             Skip automatic pip install of missing Python deps"
      echo "  --regen-tf-protos             Regenerate minimal TensorFlow protobuf sources"
      echo "  -h, --help                    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ "${KDNN_SOURCE}" != "LOCAL" && "${KDNN_SOURCE}" != "REMOTE" ]]; then
  echo "ERROR: --kdnn-source must be LOCAL or REMOTE, got '${KDNN_SOURCE}'" >&2
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

echo "Running CMake with the following configuration:"
echo "  Build Type: ${BUILD_TYPE}"
echo "  Install Prefix: ${INSTALL_PREFIX}"
echo "  C Compiler: ${C_COMPILER}"
echo "  C++ Compiler: ${CXX_COMPILER}"
echo "  Python: ${PYTHON}"
echo "  Constant Folding: ${ENABLE_CONSTANT_FOLDING}"
echo "  KDNN Adaptor: ${ENABLE_KDNN_ADAPTOR}"
echo "  KDNN Source: ${KDNN_SOURCE}"
echo "  KDNN Dir: ${KDNN_DIR}"

cmake .. \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DLLVM_ENABLE_LIBCXX="${ENABLE_LIBCXX}" \
  -DLLVM_ENABLE_ASSERTIONS="${ENABLE_ASSERTIONS}" \
  -DANNC_ENABLE_CONSTANT_FOLDING="${ENABLE_CONSTANT_FOLDING}" \
  -DANNC_ENABLE_KDNN_ADAPTOR="${ENABLE_KDNN_ADAPTOR}" \
  -DANNC_KDNN_SOURCE="${KDNN_SOURCE}" \
  -DANNC_KDNN_DIR="${KDNN_DIR}" \
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

# Build and install
echo "Starting build with $(nproc) parallel jobs..."
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "Build failed, aborting installation"
    exit 1
fi

make install
echo "Build and installation completed successfully"
