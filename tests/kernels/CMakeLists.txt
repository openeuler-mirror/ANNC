find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
include(GoogleTest)

set(ANNC_KERNEL_TEST_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${ANNC_KERNEL_TEST_GENERATED_DIR}")

set(ANNC_AUTO_KERNEL_WRAPPERS_GEN "${ANNC_KERNEL_TEST_GENERATED_DIR}/auto_kernel_wrappers.gen.cpp")
file(WRITE "${ANNC_AUTO_KERNEL_WRAPPERS_GEN}" [=[
#include "Kernel/MemRefTypes.h"
#define ANNC_SPEC_TOKEN_PREFIX auto_kernel_specs
#define ANNC_BUILTIN_KERNEL_SPECS_FILE "tests/kernels/auto_kernel_specs.inc"
#include "Kernel/BuiltinKernelSpecDeclare.h"
#include "tests/kernels/auto_kernel_impls.cpp"
#define ANNC_SPEC_TOKEN_PREFIX auto_kernel_specs
#define ANNC_BUILTIN_KERNEL_SPECS_FILE "tests/kernels/auto_kernel_specs.inc"
#include "Kernel/BuiltinKernelSpecDefine.h"
]=])

set(ANNC_AUTO_KERNEL_REGISTRY_GEN "${ANNC_KERNEL_TEST_GENERATED_DIR}/auto_kernel_registry.gen.cpp")
file(WRITE "${ANNC_AUTO_KERNEL_REGISTRY_GEN}" [=[
#include <utility>

#include "Kernel/KernelAPIMacros.h"
#include "Kernel/KernelRegistry.h"

namespace annc::kernels::test {

void registerAutoKernelSpecsForTest() {
    using ::annc::kernels::Name;
#undef ANNC_SPEC_TOKEN_PREFIX
#define ANNC_SPEC_TOKEN_PREFIX auto_kernel_specs
#define ANNC_BUILTIN_KERNEL_SPECS_FILE "tests/kernels/auto_kernel_specs.inc"
#define ANNC_KERNEL_SPEC(spec_token, builder_expr, args_decl, body)     ::annc::kernels::KernelRegistry::instance().registerKernel(         (builder_expr).Build(ANNC_AUTO_KERNEL_SYMBOL_STRING(spec_token), __FILE__, __LINE__));
#include ANNC_BUILTIN_KERNEL_SPECS_FILE
#undef ANNC_KERNEL_SPEC
#undef ANNC_BUILTIN_KERNEL_SPECS_FILE
#undef ANNC_SPEC_TOKEN_PREFIX
}

} // namespace annc::kernels::test
]=])

set(ANNC_BUILTIN_KERNEL_WRAPPERS_GEN "${ANNC_KERNEL_TEST_GENERATED_DIR}/builtin_kernel_wrappers.gen.cpp")
set(ANNC_BUILTIN_KERNEL_WRAPPER_SOURCE [=[
#include "Kernel/MemRefTypes.h"
#define ANNC_SPEC_TOKEN_PREFIX matmul_kernel_specs_inc
#define ANNC_BUILTIN_KERNEL_SPECS_FILE "matmul_kernel_specs.inc"
#include "Kernel/BuiltinKernelSpecDeclare.h"
#include "matmul_aarch64.cpp"
#define ANNC_SPEC_TOKEN_PREFIX matmul_kernel_specs_inc
#define ANNC_BUILTIN_KERNEL_SPECS_FILE "matmul_kernel_specs.inc"
#include "Kernel/BuiltinKernelSpecDefine.h"
#undef ANNC_SPEC_TOKEN_PREFIX
]=])
if(ANNC_ENABLE_KDNN_ADAPTOR)
    string(APPEND ANNC_BUILTIN_KERNEL_WRAPPER_SOURCE [=[

#define ANNC_SPEC_TOKEN_PREFIX kdnn_adaptor_matmul_kernel_specs_inc
#define ANNC_BUILTIN_KERNEL_SPECS_FILE "kdnn_adaptor/matmul_kernel_specs.inc"
#include "Kernel/BuiltinKernelSpecDeclare.h"
#include "kdnn_adaptor/matmul_aarch64.cpp"
#define ANNC_SPEC_TOKEN_PREFIX kdnn_adaptor_matmul_kernel_specs_inc
#define ANNC_BUILTIN_KERNEL_SPECS_FILE "kdnn_adaptor/matmul_kernel_specs.inc"
#include "Kernel/BuiltinKernelSpecDefine.h"
#undef ANNC_SPEC_TOKEN_PREFIX
]=])
endif()
file(WRITE "${ANNC_BUILTIN_KERNEL_WRAPPERS_GEN}" "${ANNC_BUILTIN_KERNEL_WRAPPER_SOURCE}")

function(annc_add_kernel_gtest target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target}
        PRIVATE
        "${PROJECT_SOURCE_DIR}"
        "${PROJECT_SOURCE_DIR}/annc/lib/Kernel/builtin_kernels"
        "${PROJECT_BINARY_DIR}/annc/include"
    )
    target_link_libraries(${target}
        PRIVATE
        GTest::gtest_main
        Threads::Threads
        ${CMAKE_DL_LIBS}
    )
    if(UNIX AND NOT APPLE)
        target_link_options(${target} PRIVATE -rdynamic)
    endif()
    gtest_discover_tests(${target})
endfunction()

annc_add_kernel_gtest(kernel_priority_resolver_test
    kernel_priority_resolver_test.cpp
)
target_link_libraries(kernel_priority_resolver_test PRIVATE ANNCKernel)

annc_add_kernel_gtest(kernel_registry_test
    kernel_registry_test.cpp
    "${ANNC_AUTO_KERNEL_WRAPPERS_GEN}"
    "${ANNC_AUTO_KERNEL_REGISTRY_GEN}"
)
target_link_libraries(kernel_registry_test PRIVATE ANNCKernel)

annc_add_kernel_gtest(customcall_symbol_resolution_test
    customcall_symbol_resolution_stub.cpp
)
target_link_libraries(customcall_symbol_resolution_test PRIVATE ANNCKernel)

annc_add_kernel_gtest(builtin_shared_spec_test
    builtin_shared_spec_test.cpp
    "${ANNC_BUILTIN_KERNEL_WRAPPERS_GEN}"
)
target_link_libraries(builtin_shared_spec_test
    PRIVATE
    ANNCKernel
    ANNCThreadPool
)
if(ANNC_ENABLE_KDNN_ADAPTOR)
    target_compile_definitions(builtin_shared_spec_test PRIVATE ANNC_TEST_ENABLE_KDNN=1)
    if(TARGET KDNN::kdnn)
        target_link_libraries(builtin_shared_spec_test PRIVATE KDNN::kdnn)
    else()
        target_include_directories(builtin_shared_spec_test PRIVATE "${KDNN_INCLUDE_DIR}")
        target_link_libraries(builtin_shared_spec_test PRIVATE "${KDNN_LIBRARY}")
    endif()
else()
    target_compile_definitions(builtin_shared_spec_test PRIVATE ANNC_TEST_ENABLE_KDNN=0)
endif()
