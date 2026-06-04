add_library(ANNCThreadPool STATIC
    threadpool/ThreadPool.cpp
    threadpool/Parallel.cpp
)

add_library(ANNCKernel
    KernelRegistry.cpp
    KernelPriorityResolver.cpp
)

install(TARGETS ANNCThreadPool ANNCKernel
        EXPORT MLIRTargets
        ARCHIVE DESTINATION lib)

# TDD test for kernel priority resolution (hasAnyAvailableKernel,
# resolveBestKernel). No MLIR dependency — kdnn config is a plain bool.
add_executable(kernel_priority_resolver_test
    ${CMAKE_SOURCE_DIR}/tests/kernels/kernel_priority_resolver_test.cpp
)
target_link_libraries(kernel_priority_resolver_test
    PRIVATE
    ANNCKernel
)

add_subdirectory(builtin_kernels)
