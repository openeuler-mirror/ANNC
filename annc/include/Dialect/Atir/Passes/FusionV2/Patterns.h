#ifndef ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_PATTERNS_H
#define ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_PATTERNS_H

#include "mlir/IR/PatternMatch.h"
#include "Dialect/Atir/Passes/FusionV2/FusionV2Outline.h"

namespace atir {

void addActionIdGatherV2Pattern(mlir::RewritePatternSet &patterns,
                                 fusionv2::CallPortTable *portTable);

// dnn_embedding_hash_bucket: string_to_hash_bucket_fast_v2 anchor; the
// string domain behind the hash stays opaque.
void addDnnEmbeddingHashBucketV2Pattern(
    mlir::RewritePatternSet &patterns,
    fusionv2::CallPortTable *portTable);

// kp_gather (KPFusedGather): Unique residue anchor, promoted mode
// (data/keys/begin).
void addKpGatherV2Pattern(mlir::RewritePatternSet &patterns,
                          fusionv2::CallPortTable *portTable);

void populateAtirOpFusionV2Patterns(mlir::RewritePatternSet &patterns,
                                    fusionv2::CallPortTable *portTable);
void addSparseDynamicStitchV2Pattern(mlir::RewritePatternSet &patterns);
void addSparseSegmentReduceV2Pattern(mlir::RewritePatternSet &patterns);
void addSparseReshapeV2Pattern(mlir::RewritePatternSet &patterns);
// kp_embedding_padding: Reshape-residue anchor with a runtime target over
// the string-transport carrier chain (concat-with-zeros tail).
void addEmbeddingPaddingV2Pattern(mlir::RewritePatternSet &patterns,
                                  fusionv2::CallPortTable *portTable);
void addSparseSelectV2Pattern(mlir::RewritePatternSet &patterns);
void addTargetBehaviorInteractionV2Pattern(mlir::RewritePatternSet &patterns);
// kp_embedding_padding_fast: StridedSlice anchor (or the scalar reshape
// folded at import); shares the padding kernel family (dual rank-0
// outputs). The slow form requires the unranked atir origin carrier, the
// fast form a plain rank1 i64 vector - the two FE shapes differ by design.
void addEmbeddingPaddingFastV2Pattern(mlir::RewritePatternSet &patterns,
                                      fusionv2::CallPortTable *portTable);

// dot GEMM: stablehlo.dot anchor (TF MatMul legalized shape), matching and
// outlining only (pre-lowering, native carrier ops moved verbatim); the
// fusion.metadata follows the context v3 contract (outlineContextBuiltin
// line); gated by annc::gemmConfig (env ANNC_GEMM).
void addDotGemmV2Pattern(mlir::RewritePatternSet &patterns);

}  // namespace atir

#endif  // ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_PATTERNS_H
