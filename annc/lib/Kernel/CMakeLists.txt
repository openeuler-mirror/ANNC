add_library(ANNCThreadPool STATIC
    threadpool/ThreadPoolContext.cpp
    threadpool/Parallel.cpp
)

add_library(ANNCKernel
    KernelRegistry.cpp
    KernelSymbolResolver.cpp
)

install(TARGETS ANNCThreadPool ANNCKernel
        ARCHIVE DESTINATION lib)

add_subdirectory(builtin_kernels)
