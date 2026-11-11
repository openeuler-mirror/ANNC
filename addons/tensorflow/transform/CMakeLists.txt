# pb2stablehlo — TensorFlow GraphDef -> StableHLO frontend (V2 route).
#
# Entered from addons/tensorflow/CMakeLists.txt under ANNC_ENABLE_TF2ATIR_V2
# (the one opt-in switch of the whole V2 route, same as the sink side) while
# the target TensorFlow is 2.20.x.
#
# Toolchain isolation: this target is compiled and linked against the headers
# and shared objects shipped by the pip TensorFlow wheel (its vendored MLIR,
# 21.0.0git), NOT against the project LLVM/MLIR toolchain. The pipeline calls
# TF's own C++ transform passes (odml::AddLegalizeTFToStablehloPasses, ...),
# whose symbols live in libtensorflow_cc.so.2; the MLIR C++ ABI must match
# that .so exactly, so mixing in the project toolchain is not an option.
#
# External dependency, resolved in this priority order (nothing is vendored
# into the repository):
#   1. third_party/llvm-include/ — a pre-placed local snapshot (gitignored;
#      the wheel does not ship llvm/ headers, this dir supplies them),
#   2. ANNC_TF_TRANSFORM_LLVM_INCLUDE_DIR — an existing llvm include root
#      (a full llvm-project checkout's llvm/include),
#   3. automatic sparse clone of llvm-project from
#      ANNC_TF_TRANSFORM_LLVM_REPO (default: the official GitHub repository;
#      e.g. https://gitee.com/mirrors/LLVM-Project.git is a reachable
#      mirror — its tags lag upstream, which is why the default ref is a
#      branch) at ANNC_TF_TRANSFORM_LLVM_REF (default release/21.x, the line
#      the TF 2.20 wheel vendors as 21.0.0git; pin an exact commit here when
#      a ref drift ever breaks the build), landing under the build tree.
# The generated llvm/Config/llvm-config.h is supplied from
# cmake/wheel-llvm-config/ (a build asset: it encodes the wheel's LLVM build
# configuration; upstream ships only the .cmake template). Version mismatch
# against the wheel's vendored MLIR fails at link time with relocation
# errors — keep ref/repo overrides on the 21.0.0 line only.
#
# Build: cmake -DANNC_ENABLE_TF2ATIR_V2=ON, then `ninja pb2_transform_objs`
# (the pb2stablehlo executable is assembled once every source file of the
# incremental delivery is in). Output lands in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
# (the project's build/bin).

find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import tensorflow as tf; print(tf.sysconfig.get_include())"
  OUTPUT_VARIABLE _tf_include
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _tf_include_rc
  ERROR_QUIET)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import tensorflow as tf; print(tf.sysconfig.get_lib())"
  OUTPUT_VARIABLE _tf_lib
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _tf_lib_rc
  ERROR_QUIET)

if(NOT _tf_include_rc EQUAL 0 OR NOT _tf_lib_rc EQUAL 0)
  message(FATAL_ERROR
    "ANNC_ENABLE_TF2ATIR_V2=ON with the TF-2.20 addons requires TensorFlow "
    "importable by ${Python3_EXECUTABLE}; `python3 -c 'import tensorflow as "
    "tf'` failed. Install the TensorFlow wheel pinned in requirements.txt, "
    "or configure with -DANNC_ENABLE_TF2ATIR_V2=OFF.")
endif()

# The vendored llvm headers and the wheel's vendored MLIR must be exactly the
# same LLVM 21.0.0git build; any other TF version passes the file checks below
# and only breaks later at compile/link time with far less legible errors.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(tf.__version__)"
  OUTPUT_VARIABLE _tf_version
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _tf_version_rc
  ERROR_QUIET)
if(NOT _tf_version_rc EQUAL 0 OR NOT _tf_version MATCHES "^2\\.20(\\.|$)")
  message(FATAL_ERROR
    "pb2stablehlo requires TensorFlow 2.20.x (found: ${_tf_version}); the "
    "llvm header snapshot is ABI-locked to the 2.20 wheel's vendored MLIR.")
endif()

# mlir headers are vendored inside the wheel, next to the .so that consumes them.
set(_tf_mlir_include "${_tf_include}/external/llvm-project/mlir/include")
if(NOT EXISTS "${_tf_mlir_include}/mlir/IR/Builders.h")
  message(FATAL_ERROR
    "ANNC_ENABLE_TF2ATIR_V2=ON: missing ${_tf_mlir_include}/mlir/IR/Builders.h "
    "(mlir headers come from the pip TensorFlow wheel).")
endif()
if(NOT EXISTS "${_tf_lib}/libtensorflow_cc.so.2")
  message(FATAL_ERROR
    "ANNC_ENABLE_TF2ATIR_V2=ON: missing ${_tf_lib}/libtensorflow_cc.so.2.")
endif()

# Pin the wheel's libstdc++ ABI: a mismatch fails at link/load time with
# relocation errors far from the cause.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import tensorflow as tf; print([f for f in tf.sysconfig.get_compile_flags() if '_GLIBCXX_USE_CXX11_ABI=' in f][0].split('=')[1])"
  OUTPUT_VARIABLE _tf_cxx11_abi
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _tf_abi_rc
  ERROR_QUIET)
if(NOT _tf_abi_rc EQUAL 0 OR NOT _tf_cxx11_abi MATCHES "^[01]$")
  message(FATAL_ERROR
    "pb2stablehlo: the TF wheel does not report a valid _GLIBCXX_USE_CXX11_ABI.")
endif()
message(STATUS "pb2stablehlo CXX11 ABI: ${_tf_cxx11_abi} (wheel-pinned)")

# ---- llvm header root (the three-level chain above) ----
set(_llvm_snapshot "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-include")
set(_llvm_source "local snapshot")
if(NOT EXISTS "${_llvm_snapshot}/llvm/Config/llvm-config.h")
  if(ANNC_TF_TRANSFORM_LLVM_INCLUDE_DIR)
    set(_llvm_snapshot "${ANNC_TF_TRANSFORM_LLVM_INCLUDE_DIR}")
    set(_llvm_source "ANNC_TF_TRANSFORM_LLVM_INCLUDE_DIR")
  else()
    set(_llvm_repo "${ANNC_TF_TRANSFORM_LLVM_REPO}")
    if(NOT _llvm_repo)
      set(_llvm_repo "https://github.com/llvm/llvm-project.git")
    endif()
    set(_llvm_ref "${ANNC_TF_TRANSFORM_LLVM_REF}")
    if(NOT _llvm_ref)
      set(_llvm_ref "release/21.x")
    endif()
    set(_llvm_snapshot "${CMAKE_BINARY_DIR}/transform-llvm-include")
    if(NOT EXISTS "${_llvm_snapshot}/llvm/ADT/StringRef.h")
      message(STATUS "pb2stablehlo: sparse-cloning llvm/include from "
                     "${_llvm_repo} @ ${_llvm_ref}")
      set(_clone_dir "${CMAKE_BINARY_DIR}/transform-llvm-clone")
      file(REMOVE_RECURSE "${_clone_dir}")
      execute_process(
        COMMAND git clone --quiet --depth 1 --filter=blob:none --sparse
                --branch "${_llvm_ref}" "${_llvm_repo}" "${_clone_dir}"
        RESULT_VARIABLE _clone_rc ERROR_VARIABLE _clone_err)
      if(_clone_rc EQUAL 0)
        execute_process(
          COMMAND git -C "${_clone_dir}" sparse-checkout set llvm/include
          RESULT_VARIABLE _sparse_rc ERROR_VARIABLE _sparse_err)
      endif()
      if(NOT _clone_rc EQUAL 0 OR NOT _sparse_rc EQUAL 0)
        message(FATAL_ERROR
          "Fetching the llvm headers failed (${_clone_err}${_sparse_err}). "
          "Fix the network/repo, or place a snapshot under "
          "third_party/llvm-include/, or point ANNC_TF_TRANSFORM_LLVM_INCLUDE_DIR "
          "at an existing llvm/include root.")
      endif()
      file(COPY "${_clone_dir}/llvm/include/llvm" DESTINATION "${_llvm_snapshot}")
      file(COPY "${_clone_dir}/llvm/include/llvm-c" DESTINATION "${_llvm_snapshot}")
    endif()
    set(_llvm_source "sparse clone @ ${_llvm_ref}")
  endif()
  # The generated config header is not part of any source tree; supply the
  # wheel's build configuration from the in-repo build asset.
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/wheel-llvm-config/llvm-config.h"
       DESTINATION "${_llvm_snapshot}/llvm/Config/")
endif()
message(STATUS "pb2stablehlo llvm headers: ${_llvm_snapshot} (${_llvm_source})")

# NOTE (experiment 2026-11-05): the wheel's external/stablehlo headers can be
# included, but the non-template convenience constructors (e.g.
# GatherDimensionNumbersAttr::get) are NOT in libtensorflow_cc.so.2's export
# table in either the mhlo or stablehlo namespace — typed construction is
# not linkable under wheel isolation. Emit ops via OperationState +
# mlir::parseAttribute (mlir/AsmParser/AsmParser.h); anchor attribute
# spellings on odml-produced artifact forms.

# Incremental delivery: the object library accumulates translation units as
# the migration lands them; the pb2stablehlo executable is added by the
# final batch and links every object with the TF wheel .so files.
add_library(pb2_transform_objs OBJECT
  common/pb2_common.cpp
  common/pb2_attrs.cpp
  graph/pb2_graph_pass.cpp
  graph/pb2_infer.cpp
  graph/pb2_official_infer.cpp
  passes/pb2_tf_rank_infer.cpp
  passes/pb2_mlir_passes.cpp
  passes/pb2_restore_pass.cpp
  pipeline/pb2_emit.cpp
  wrap/pb2_unrank_wrap.cpp
  wrap/pb2_strided_slice.cpp
  wrap/pb2_string_transport.cpp
  wrap/pb2_string_unwrap.cpp)

# The import and run-pipelines TUs reference shlo_pipeline:: (the legalize
# orchestration with its wrap-layer passes, landing with T8); OBJECT
# libraries have no link check, but folding them into pb2_transform_objs
# would break the test binaries. T8 completes the closure and folds back.
add_library(pb2_pipeline_objs OBJECT
  graph/pb2_import.cpp
  pipeline/pb2_run_pipelines.cpp)

target_include_directories(pb2_pipeline_objs BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${_tf_include}
  ${_tf_mlir_include}
  ${_llvm_snapshot})
target_compile_features(pb2_pipeline_objs PRIVATE cxx_std_17)
target_compile_options(pb2_pipeline_objs PRIVATE -O1 -g -pthread)
target_compile_definitions(pb2_pipeline_objs PRIVATE
  NDEBUG "_GLIBCXX_USE_CXX11_ABI=${_tf_cxx11_abi}")

# BEFORE: the project-level global include_directories() (third_party/llvm,
# build-llvm20, stablehlo, ...) would otherwise precede these paths and the
# llvm/ headers would silently resolve to the project LLVM 20 toolchain
# instead of the wheel-synchronized snapshot (measured via ninja -t deps).
target_include_directories(pb2_transform_objs BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${_tf_include}
  ${_tf_mlir_include}
  ${_llvm_snapshot})

# -O1 -g keeps compile time sane; NDEBUG is required (absl templates would
# otherwise split ODR across the .so boundary).
target_compile_features(pb2_transform_objs PRIVATE cxx_std_17)
target_compile_options(pb2_transform_objs PRIVATE -O1 -g -pthread)
target_compile_definitions(pb2_transform_objs PRIVATE
  NDEBUG "_GLIBCXX_USE_CXX11_ABI=${_tf_cxx11_abi}")

# Shared wheel link environment for executables consuming the object library
# (the final tool binary and the tests/unit/transform suite): the TF wheel
# .so files plus the export/rpath link options. Include paths and compile
# flags are borrowed per-consumer via $<TARGET_PROPERTY:...> so the crucial
# BEFORE ordering against the project-global include_directories() is kept.
add_library(pb2_wheel_env INTERFACE)
target_link_libraries(pb2_wheel_env INTERFACE
  "${_tf_lib}/libtensorflow_cc.so.2"
  "${_tf_lib}/libtensorflow_framework.so.2")
target_link_options(pb2_wheel_env INTERFACE
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.script"
  "-Wl,-rpath,${_tf_lib}")
