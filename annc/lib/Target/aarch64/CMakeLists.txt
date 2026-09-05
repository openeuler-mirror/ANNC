file(GLOB _srcs *.cpp)

set(_gemm_srcs
        Gemm/GemmPlan.cpp
        Gemm/ConstantFolding/GemmPrepackRhs.cpp
        Gemm/ConstantFolding/CheckpointReader.cpp
        Gemm/ResolveGemmPlan.cpp
        Gemm/SelectGemmStrategy.cpp
        Gemm/GemmAutotune.cpp
        Gemm/FinalizeGemmPlan.cpp
        Gemm/GemmTilingUtils.cpp
        Gemm/GemmCacheBlocking.cpp
        Gemm/GemmKernelTiling.cpp
        Gemm/GemmLeafMaterialization.cpp
        Gemm/GemmMicrokernelLowering.cpp
        Gemm/GemmABILowering.cpp
        Gemm/GemmEpilogueLowering.cpp
        Gemm/VerifyGemmSchedule.cpp
)

add_mlir_library(AArch64Codegen
        ${_srcs}
        ${_gemm_srcs}

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/annc/include
        ${CMAKE_SOURCE_DIR}/build/annc/include

        DEPENDS
        AArch64CodeGenPassIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRSupport
        MLIRParser
        MLIRRewrite
        MLIRTransforms
        MLIRAffineDialect
        MLIRArithDialect
        MLIRMemRefDialect
        MLIRFuncDialect
        MLIRBufferizationTransforms
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRLLVMDialect
)

if(ANNC_AARCH64_GEMM_KERNELS_AVAILABLE)
  target_link_libraries(AArch64Codegen PUBLIC ANNC::AArch64GemmMicrokernels)
endif()
