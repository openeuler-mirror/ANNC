file(GLOB _srcs *.cpp)

set(_gemm_srcs
        Gemm/GemmPlan.cpp
        Gemm/ConstantFolding/GemmPrepackRhs.cpp
        Gemm/ConstantFolding/JitPrepack.cpp
        Gemm/ResolveGemmPlan.cpp
        Gemm/SelectGemmStrategy.cpp
        Gemm/GemmThreadPlanning.cpp
        Gemm/FinalizeGemmPlan.cpp
        Gemm/GemmThreadTiling.cpp
        Gemm/GemmTilingUtils.cpp
        Gemm/GemmCacheBlocking.cpp
        Gemm/GemmKernelTiling.cpp
        Gemm/GemmLeafMaterialization.cpp
        Gemm/GemmMicrokernelSelection.cpp
        Gemm/GemmMicrokernelLowering.cpp
        Gemm/GemmABILowering.cpp
        Gemm/GemmEpilogueLowering.cpp
        Gemm/Epilogue/AtirGemmEpilogueCommit.cpp
        Gemm/VerifyGemmSchedule.cpp
        Gemm/Support/GemmPlanner.cpp
)

# Only this adapter sees the binary package C ABI. The object exporter can
# reuse it without linking MLIR passes or machine-specific generation code.
add_library(ANNCGemmCodegenAdapter STATIC Gemm/GemmCodegenAdapter.cpp)
target_compile_features(ANNCGemmCodegenAdapter PUBLIC cxx_std_17)
target_compile_definitions(ANNCGemmCodegenAdapter PRIVATE
  ANNC_GEMM_PACKAGE_AVAILABLE=$<BOOL:${ANNC_GEMM_PACKAGE_AVAILABLE}>)
target_include_directories(ANNCGemmCodegenAdapter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/annc/include>
  $<INSTALL_INTERFACE:include>)
if(ANNC_GEMM_PACKAGE_AVAILABLE)
  target_link_libraries(ANNCGemmCodegenAdapter PRIVATE ANNC::GemmCodegen)
endif()
install(TARGETS ANNCGemmCodegenAdapter EXPORT MLIRTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Dispatcher entry points are referenced by generated MLIR, so keep them in
# an installed runtime archive instead of the compiler-only pass library.
add_library(ANNCGemmSupport STATIC
        Gemm/Support/GemmDispatch.cpp
)
target_link_libraries(ANNCGemmSupport PUBLIC ANNCThreadPool ANNCContext)
target_include_directories(ANNCGemmSupport
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/annc/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Gemm
)
install(TARGETS ANNCGemmSupport
        EXPORT MLIRTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_mlir_library(AArch64Codegen
        ${_srcs}
        ${_gemm_srcs}

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/annc/include
        ${CMAKE_SOURCE_DIR}/build/annc/include

        DEPENDS
        AArch64CodeGenPassIncGen

        LINK_LIBS PUBLIC
        ANNCCheckpoint
        ANNCGemmCodegenAdapter
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
        MLIRSCFDialect
        AtirDialect
        AtirPasses
        ContextConversion
)

# This is a dependency capability, not a user-selectable optimization mode.
# Apply it to the source so it also reaches MLIR's obj.AArch64Codegen target.
set_property(SOURCE Gemm/ConstantFolding/GemmPrepackRhs.cpp APPEND PROPERTY
  COMPILE_DEFINITIONS
  ANNC_GEMM_PACKAGE_AVAILABLE=$<BOOL:${ANNC_GEMM_PACKAGE_AVAILABLE}>)

if(ANNC_GEMM_PACKAGE_AVAILABLE)
  target_link_libraries(AArch64Codegen PUBLIC ANNC::AArch64GemmMicrokernels)
endif()
