add_library(ANNCThreadPool STATIC
    threadpool/ThreadPool.cpp
    threadpool/Parallel.cpp
)

add_library(ANNCKernel
    KernelRegistry.cpp
    KernelPriorityResolver.cpp
)

install(TARGETS ANNCThreadPool ANNCKernel
        ARCHIVE DESTINATION lib)

add_subdirectory(builtin_kernels)
