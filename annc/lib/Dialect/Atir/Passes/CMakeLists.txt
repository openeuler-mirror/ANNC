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
       ${CMAKE_CURRENT_SOURCE_DIR}/FusionV2/KpGatherV2.cpp)
else()
  list(REMOVE_ITEM _srcs
       ${CMAKE_CURRENT_SOURCE_DIR}/Patterns/CustomPatterns/KpEmbeddingGatherV2Rewrite.cpp)
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

# The register bundle carries the stablehlo dialect symbols into every
# consumer of AtirPasses (annc-opt / annc-asm grow accordingly: L1 change).
if(ANNC_ENABLE_TF2ATIR_V2)
  target_link_libraries(AtirPasses PUBLIC StablehloRegister)
  # Directory scope: the obj.AtirPasses sub-target created by add_mlir_library
  # does not inherit target_compile_definitions, and Initialize.cpp needs the
  # macro for the gated registration hook.
  add_compile_definitions(ANNC_ENABLE_TF2ATIR_V2=1)
endif()
