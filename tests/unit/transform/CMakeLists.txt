# Unit tests for the pb2stablehlo transform, compiled in its wheel-isolated
# domain. Entered only from tests/CMakeLists.txt while the pb2_transform_objs
# target exists (ANNC_ENABLE_TF_TRANSFORM=ON).
#
# No GTest here: the project GTest infrastructure belongs to the project
# LLVM/MLIR toolchain domain, and mixing those headers with the wheel's
# vendored MLIR/LLVM ABI is exactly what the transform build avoids. The
# suites use plain CHECK macros and a nonzero exit code.
#
# Include paths and compile flags are borrowed from the object library via
# $<TARGET_PROPERTY:...>; the BEFORE on the include line is load-bearing —
# the tests tree is entered after the project-global include_directories(),
# and without BEFORE the llvm/ headers would resolve to the project LLVM 20
# toolchain instead of the wheel-synchronized snapshot.

add_executable(pb2_common_test
  pb2_common_test.cpp
  $<TARGET_OBJECTS:pb2_transform_objs>)

target_include_directories(pb2_common_test BEFORE PRIVATE
  $<TARGET_PROPERTY:pb2_transform_objs,INCLUDE_DIRECTORIES>)
target_compile_options(pb2_common_test PRIVATE
  $<TARGET_PROPERTY:pb2_transform_objs,COMPILE_OPTIONS>)
target_compile_definitions(pb2_common_test PRIVATE
  $<TARGET_PROPERTY:pb2_transform_objs,COMPILE_DEFINITIONS>)
target_link_libraries(pb2_common_test PRIVATE pb2_wheel_env)

add_test(NAME Pb2CommonTest COMMAND pb2_common_test)

# T2 suites: the pb-domain inference chain and the tf-rank pass. Same
# borrowing pattern as above (include/flags from the object library, link
# env from pb2_wheel_env).
foreach(_suite pb2_infer pb2_tf_rank_infer pb2_mlir_passes pb2_t5 pb2_wrap pb2_string)
  add_executable(${_suite}_test
    ${_suite}_test.cpp
    $<TARGET_OBJECTS:pb2_transform_objs>)
  target_include_directories(${_suite}_test BEFORE PRIVATE
    $<TARGET_PROPERTY:pb2_transform_objs,INCLUDE_DIRECTORIES>)
  target_compile_options(${_suite}_test PRIVATE
    $<TARGET_PROPERTY:pb2_transform_objs,COMPILE_OPTIONS>)
  target_compile_definitions(${_suite}_test PRIVATE
    $<TARGET_PROPERTY:pb2_transform_objs,COMPILE_DEFINITIONS>)
  target_link_libraries(${_suite}_test PRIVATE pb2_wheel_env)
  add_test(NAME ${_suite}_test COMMAND ${_suite}_test)
endforeach()
