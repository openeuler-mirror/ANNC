add_library(ANNCThreadPool STATIC
    threadpool/ThreadPool.cpp
    threadpool/Parallel.cpp
)

add_library(ANNCKernel
    KernelRegistry.cpp
    KernelSymbolResolver.cpp
)

install(TARGETS ANNCThreadPool ANNCKernel
        EXPORT MLIRTargets
        ARCHIVE DESTINATION lib)

add_subdirectory(builtin_kernels)
