from .sparse_embedding import (KPSparseSelectPatternRewriter,
                               KPFusedGatherPatternRewriter,
                               KPSparseReshapePatternRewriter,
                               KPEmbeddingActionIdGatherPatternRewriter)

from .rec_embedding import (DnnSparseEmbeddingPatternRewriter,
                            DnnEmbeddingWithHashBucketPatternRewriter,
                            EmbeddingPatternRewriter,
                            EmbeddingWithHashBucketPatternRewriter,
                            LinearSparseEmbeddingPatternRewriter,
                            KPSparseSegmentReducePatternRewriter,
                            KPSparsePaddingFastPatternRewriter,
                            KPSparsePaddingPatternRewriter,
                            KPSparseDynamicStitchPatternRewriter)
