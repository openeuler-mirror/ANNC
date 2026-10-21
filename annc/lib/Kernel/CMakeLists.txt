add_library(ANNCKernel
    KernelRegistry.cpp
    KernelPriorityResolver.cpp
)

install(TARGETS ANNCKernel
        EXPORT MLIRTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_subdirectory(builtin_kernels)
