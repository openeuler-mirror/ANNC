#ifndef ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_PATTERNS_H
#define ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_PATTERNS_H

#include "mlir/IR/PatternMatch.h"
#include "Dialect/Atir/Passes/FusionV2/FusionV2Outline.h"

namespace atir {

void addActionIdGatherV2Pattern(mlir::RewritePatternSet &patterns);

// kp_gather (KPFusedGather): Unique residue anchor, promoted mode
// (data/keys/begin).
void addKpGatherV2Pattern(mlir::RewritePatternSet &patterns,
                          fusionv2::CallPortTable *portTable);

void populateAtirOpFusionV2Patterns(mlir::RewritePatternSet &patterns,
                                    fusionv2::CallPortTable *portTable);
void addSparseDynamicStitchV2Pattern(mlir::RewritePatternSet &patterns);
void addSparseSegmentReduceV2Pattern(mlir::RewritePatternSet &patterns);
void addSparseReshapeV2Pattern(mlir::RewritePatternSet &patterns);
void addEmbeddingPaddingV2Pattern(mlir::RewritePatternSet &patterns);
void addSparseSelectV2Pattern(mlir::RewritePatternSet &patterns);
void addTargetBehaviorInteractionV2Pattern(mlir::RewritePatternSet &patterns);
// kp_embedding_padding_fast: StridedSlice anchor (or the scalar reshape
// folded at import); shares the padding kernel family (dual rank-0 outputs).
void addEmbeddingPaddingFastV2Pattern(mlir::RewritePatternSet &patterns);
// dnn_embedding_hash_bucket: string_to_hash_bucket_fast_v2 anchor, ABI
// (input:string, embedding_weight) -> output; num_buckets via schema attr.
void addDnnEmbeddingHashBucketV2Pattern(mlir::RewritePatternSet &patterns);


}  // namespace atir

#endif  // ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_PATTERNS_H
