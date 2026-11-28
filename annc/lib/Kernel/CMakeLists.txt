add_library(ANNCKernel
    KernelRegistry.cpp
    KernelPriorityResolver.cpp
)
target_link_libraries(ANNCKernel PUBLIC ANNCContext)

install(TARGETS ANNCKernel
        EXPORT MLIRTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_subdirectory(builtin_kernels)
