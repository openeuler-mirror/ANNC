file(GLOB _srcs *.cpp Patterns/*.cpp Patterns/CustomPatterns/*.cpp Fusion/*.cpp)
# FusionV2 sources (patterns, matchers, the pass shell) live in the FusionV2/
# subdirectory and join exclusively through the gated list below - the
# top-level glob never picks them up, so a pre-commit worktree file cannot
# slip into an unrelated build. They depend on stablehlo headers and; both the
# include path and the link dependency exist only under the V2 gate. The
# OFF build keeps the historical source set: gate the lib append AND strip
# the V2 second-level rewrite out of the glob (file() returns abs paths).
if(ANNC_ENABLE_TF2ATIR_V2)
  list(APPEND _srcs
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/OpFusionV2.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/CarrierMatchers.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/FusionV2Outline.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/KpGatherV2.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/DotGemmV2.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/ActionIdGatherV2.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/DnnEmbeddingHashBucketV2.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/EmbeddingPaddingV2.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/EmbeddingPaddingFastV2.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/SparseDynamicStitchV2.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/SparseSegmentReduceV2.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/SparseReshapeV2.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/SparseSelectV2.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/TargetBehaviorInteractionV2.cpp)
else()
  list(REMOVE_ITEM _srcs
       ${CMAKE_CURRENT_SOURCE_DIR}/Patterns/CustomPatterns/KpEmbeddingGatherV2Rewrite.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/Patterns/CustomPatterns/KpEmbeddingActionIdGatherV2Rewrite.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/Patterns/CustomPatterns/KpDnnEmbeddingHashBucketV2Rewrite.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/Patterns/CustomPatterns/KpEmbeddingPaddingV2Rewrite.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/Patterns/CustomPatterns/KpEmbeddingSparseDynamicStitchV2Rewrite.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/Patterns/CustomPatterns/KpEmbeddingSparseSegmentReduceV2Rewrite.cpp)
endif()
list(APPEND _srcs Patterns/CustomPatterns/KPFusedGatherMatch.cpp)
list(APPEND _srcs Patterns/CustomPatterns/KPFusedGatherRewrite.cpp)
list(APPEND _srcs Patterns/CustomPatterns/KpEmbeddingActionIdGatherRewrite.cpp)
list(APPEND _srcs Patterns/CustomPatterns/KpEmbeddingPaddingRewrite.cpp)
list(APPEND _srcs Patterns/CustomPatterns/KpEmbeddingSparseDynamicStitchRewrite.cpp)
list(APPEND _srcs Patterns/CustomPatterns/KpEmbeddingSparseReshapeRewrite.cpp)
list(APPEND _srcs Patterns/CustomPatterns/KpEmbeddingSparseSegmentReduceRewrite.cpp)
list(APPEND _srcs Patterns/CustomPatterns/KpEmbeddingSparseSelectRewrite.cpp)
list(APPEND _srcs Patterns/CustomPatterns/KpEmbeddingTargetBehaviorInteractionRewrite.cpp)
list(APPEND _srcs Patterns/FusionBoundaryUtils.cpp)
if(ANNC_ENABLE_TF2ATIR_V2)
  list(APPEND _srcs V2/SpecializeShapes.cpp V2/AnncGemmEpilogueCommit.cpp)
endif()
list(REMOVE_DUPLICATES _srcs)

add_mlir_library(AtirPasses
    ${_srcs}

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_SOURCE_DIR}/annc/include
    ${CMAKE_SOURCE_DIR}/build/annc/include

    DEPENDS
    AtirTableGen
    MLIRAtirPassIncGen
  
    LINK_LIBS PUBLIC
    AtirDialect
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRParser
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRRewrite
    MLIRTransforms
    MLIRDialect
    ANNCKernel
    nlohmann_json::nlohmann_json
    
    AtirShapeInferIntfc
    AtirInterpretIntfc
)

if(ANNC_ENABLE_TF2ATIR_V2)
  # Compile only the upstream StableHLO pass sources used by this route. The
  # aggregate target widens the build to VHLO and unrelated conversions; this
  # source selection avoids those dependencies but does not make the selected
  # passes independent of ANNC's LLVM/MLIR API version. If a required upstream
  # pass is incompatible, use a compatible toolchain or provide an ANNC-local
  # implementation instead of expecting the aggregate or a prebuilt library
  # to resolve the API mismatch.
  add_mlir_library(ANNCStablehloShapePasses
    ${CMAKE_SOURCE_DIR}/third_party/stablehlo/stablehlo/transforms/ShapeLegalizeToStablehlo.cpp
    ${CMAKE_SOURCE_DIR}/third_party/stablehlo/stablehlo/transforms/StablehloRefineArguments.cpp
    ${CMAKE_SOURCE_DIR}/third_party/stablehlo/stablehlo/transforms/StablehloRefineShapes.cpp
    ${CMAKE_SOURCE_DIR}/third_party/stablehlo/stablehlo/transforms/optimization/StablehloAggressiveFolder.cpp
    DEPENDS PassesIncGen OptimizationPassesIncGen
    LINK_LIBS PUBLIC
    ChloOps StablehloOps MLIRArithDialect MLIRAsmParser MLIRDialectUtils
    MLIRFuncDialect MLIRInferTypeOpInterface MLIRReconcileUnrealizedCasts
    MLIRTensorDialect
    MLIRRewrite MLIRTransformUtils MLIRTransforms
  )
  target_link_libraries(AtirPasses PUBLIC ANNCStablehloShapePasses)
  target_link_libraries(AtirPasses PUBLIC StablehloRegister)
  # Directory scope: the obj.AtirPasses sub-target created by add_mlir_library
  # does not inherit target_compile_definitions, and Initialize.cpp needs the
  # macro for the gated registration hook.
  add_compile_definitions(ANNC_ENABLE_TF2ATIR_V2=1)
  target_compile_definitions(obj.AtirPasses PRIVATE ANNC_ENABLE_TF2ATIR_V2)
endif()
