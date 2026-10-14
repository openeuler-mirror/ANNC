file(GLOB _srcs *.cpp Patterns/*.cpp Patterns/CustomPatterns/*.cpp Fusion/*.cpp)
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
