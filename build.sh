#!/bin/bash

# Default configuration
BUILD_TYPE="Debug"
INSTALL_PREFIX="${PWD}/install"
LLVM_INSTALL_DIR="${LLVM_INSTALL:-/usr/local/llvm}"
ENABLE_LIBCXX="OFF"
ENABLE_ASSERTIONS="ON"
C_COMPILER="clang"
CXX_COMPILER="clang++"

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
    --llvm-install)
      LLVM_INSTALL_DIR="$2"
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
    --cc)
      C_COMPILER="$2"
      shift 2
      ;;
    --cxx)
      CXX_COMPILER="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --build-type [Debug|Release]  Set build type (default: Debug)"
      echo "  --install-prefix <path>       Set installation prefix (default: ./install)"
      echo "  --llvm-install <path>         Set LLVM/MLIR installation path (default: \$LLVM_INSTALL or /usr/local/llvm)"
      echo "  --enable-libcxx               Enable libc++"
      echo "  --disable-assertions          Disable assertions"
      echo "  --cc <compiler>               Set C compiler (default: clang)"
      echo "  --cxx <compiler>              Set C++ compiler (default: clang++)"
      echo "  -h, --help                    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

C_COMPILER="$LLVM_INSTALL_DIR/bin/$C_COMPILER"
CXX_COMPILER="$LLVM_INSTALL_DIR/bin/$CXX_COMPILER"

# Initialize and update submodules
git submodule init
git submodule update

# Create build directory
rm -rf build && mkdir build && cd build || exit 1

# Run CMake
echo "Running CMake with the following configuration:"
echo "  Build Type: ${BUILD_TYPE}"
echo "  Install Prefix: ${INSTALL_PREFIX}"
echo "  LLVM Install Dir: ${LLVM_INSTALL_DIR}"
echo "  C Compiler: $(which ${C_COMPILER})"
echo "  C++ Compiler: $(which ${CXX_COMPILER})"

cmake -G Ninja .. \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DLLVM_ENABLE_LIBCXX="${ENABLE_LIBCXX}" \
  -DMLIR_DIR="${LLVM_INSTALL_DIR}/lib/cmake/mlir" \
  -DLLVM_DIR="${LLVM_INSTALL_DIR}/lib/cmake/llvm" \
  -DLLVM_ENABLE_ASSERTIONS="${ENABLE_ASSERTIONS}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_C_COMPILER="$(which ${C_COMPILER})" \
  -DCMAKE_CXX_COMPILER="$(which ${CXX_COMPILER})" \
  -DCMAKE_CXX_FLAGS="-fPIC" \
  -DPYTHON_EXECUTABLE="$(which python3)" \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
  -Dnanobind_DIR=$(python -c "import nanobind, os; print(os.path.join(os.path.dirname(nanobind.__file__), 'cmake'))")

if [ $? -ne 0 ]; then
    echo "CMake configuration failed, aborting build"
    exit 1
fi

# Build and install
echo "Starting build..."
ninja
if [ $? -ne 0 ]; then
    echo "Build failed, aborting installation"
    exit 1
fi

ninja install
echo "Build and installation completed successfully"
