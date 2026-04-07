add_library(ANNCKernel
    KernelRegistry.cpp
    KernelSymbolResolver.cpp
)

install(TARGETS ANNCKernel
        ARCHIVE DESTINATION lib)

add_subdirectory(builtin_kernels)
