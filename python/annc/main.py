import os
import argparse
from annc.optimize import (DnnSparseEmbeddingPatternRewriter,
                           DnnEmbeddingWithHashBucketPatternRewriter,
                           EmbeddingPatternRewriter,
                           EmbeddingWithHashBucketPatternRewriter,
                           LinearSparseEmbeddingPatternRewriter,
                           KPSparseSegmentReducePatternRewriter,
                           KPSparsePaddingFastPatternRewriter,
                           KPSparsePaddingPatternRewriter,
                           KPSparseDynamicStitchPatternRewriter,
                           KPSparseSelectPatternRewriter,
                           KPFusedGatherPatternRewriter,
                           KPSparseReshapePatternRewriter,
                           KPEmbeddingActionIdGatherPatternRewriter,
                           KPFusedSparseSegmentReduceNonzeroPatternRewriter)
from annc.optimize.graph import MetaGraph


def parse_args():
    parser = argparse.ArgumentParser('ANNC-Model-Optimizer')
    parser.add_argument('input', help='Input model path')
    parser.add_argument('-o', '--output', required=True, help='Output model path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Output debug file')
    parser.add_argument('passes', nargs='+',
        help='opt: \'dnn_sparse\', \'dnn_hash_bucket\', \'embed\','
             ' \'embed_hash_bucket\', \'linear_sparse\', '
             '\'sparse_segment_reduce\', \'sparse_concat\','
             '\'sparse_select\', \'fused_gather\', \'sparse_reshape\','
             '\'action_id_gather\', \'sparse_segment_reduce_nonzero\'')
    return parser.parse_args()


OPT_PASSES = {
    'dnn_sparse': DnnSparseEmbeddingPatternRewriter,
    'dnn_hash_bucket': DnnEmbeddingWithHashBucketPatternRewriter,
    'embed': EmbeddingPatternRewriter,
    'embed_hash_bucket': EmbeddingWithHashBucketPatternRewriter,
    'linear_sparse': LinearSparseEmbeddingPatternRewriter,
    'sparse_segment_reduce': KPSparseSegmentReducePatternRewriter,
    'sparse_padding_fast': KPSparsePaddingFastPatternRewriter,
    'sparse_padding': KPSparsePaddingPatternRewriter,
    'sparse_dynamic_stitch': KPSparseDynamicStitchPatternRewriter,
    'sparse_select': KPSparseSelectPatternRewriter,
    'fused_gather': KPFusedGatherPatternRewriter,
    'sparse_reshape': KPSparseReshapePatternRewriter,
    'action_id_gather': KPEmbeddingActionIdGatherPatternRewriter,
    'sparse_segment_reduce_nonzero': KPFusedSparseSegmentReduceNonzeroPatternRewriter,
}


def opt():
    args = parse_args()
    meta_graph = MetaGraph(args.input)

    for pass_name in args.passes:
        if pass_name not in OPT_PASSES:
            raise TypeError(f'Pass \'{pass_name}\' not found')
        OPT_PASSES[pass_name](meta_graph.graph)

    os.makedirs(args.output, exist_ok=True)
    meta_graph.save(args.output, to_text=args.verbose)


if __name__ == '__main__':
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    opt()
