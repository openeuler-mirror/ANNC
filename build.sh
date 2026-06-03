#!/bin/bash

# Default configuration
BUILD_TYPE="Release"
INSTALL_PREFIX="${PWD}/install"
ENABLE_LIBCXX="OFF"
ENABLE_ASSERTIONS="ON"
C_COMPILER="${CC:-gcc}"
CXX_COMPILER="${CXX:-g++}"

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
    --clean)
      CLEAN_BUILD="YES"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --build-type [Debug|Release|RelWithDebInfo]  Set build type (default: Release)"
      echo "  --install-prefix <path>       Set installation prefix (default: ./install)"
      echo "  --enable-libcxx               Enable libc++"
      echo "  --disable-assertions          Disable assertions"
      echo "  --clean                       Clean build directory before build"
      echo "  -h, --help                    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create build directory (preserve LLVM build if exists)
if [ -d "build" ] && [ "$CLEAN_BUILD" != "YES" ]; then
  echo "Using existing build directory (use --clean to force full rebuild)"
  cd build || exit 1
else
  echo "Creating new build directory..."
  rm -rf build && mkdir build && cd build || exit 1
fi

# Run CMake
echo "Running CMake with the following configuration:"
echo "  Build Type: ${BUILD_TYPE}"
echo "  Install Prefix: ${INSTALL_PREFIX}"
echo "  C Compiler: ${C_COMPILER}"
echo "  C++ Compiler: ${CXX_COMPILER}"

cmake .. \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DLLVM_ENABLE_LIBCXX="${ENABLE_LIBCXX}" \
  -DLLVM_ENABLE_ASSERTIONS="${ENABLE_ASSERTIONS}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_C_COMPILER="$(which ${C_COMPILER})" \
  -DCMAKE_CXX_COMPILER="$(which ${CXX_COMPILER})" \
  -DCMAKE_CXX_FLAGS="-fPIC" \
  -DPYTHON_EXECUTABLE="$(which python3)" \
  -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") \
  -Dnanobind_DIR=$(python3 -c "import nanobind, os; print(os.path.join(os.path.dirname(nanobind.__file__), 'cmake'))")

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
