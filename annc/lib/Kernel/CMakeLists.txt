add_library(ANNCKernel
    KernelRegistry.cpp
    KernelPriorityResolver.cpp
)

install(TARGETS ANNCKernel
        EXPORT MLIRTargets
        ARCHIVE DESTINATION lib)

add_subdirectory(builtin_kernels)
