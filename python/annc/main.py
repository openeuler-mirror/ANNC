import os
import argparse
from annc.optimize.rec_embedding import (DnnSparseEmbeddingPatternRewriter,
                                         DnnEmbeddingWithHashBucketPatternRewriter,
                                         EmbeddingPatternRewriter,
                                         EmbeddingWithHashBucketPatternRewriter,
                                         LinearSparseEmbeddingPatternRewriter,
                                         KPSparseSegmentReducePatternRewriter,
                                         KPSparseConcatPatternRewriter,
                                         KPSparseDynamicStitchPatternRewriter,)
from annc.optimize.graph import MetaGraph
from annc.optimize.graph import convert_pbtxt_to_saved_model


def parse_args():
    parser = argparse.ArgumentParser('ANNC-Model-Optimizer')
    parser.add_argument('-I',
                        '--input',
                        required=True,
                        help='Input model path')
    parser.add_argument('-O',
                        '--output',
                        required=True,
                        help='Output model path')
    parser.add_argument(
        'passes',
        nargs='+',
        help='opt: \'dnn_sparse\', \'dnn_hash_bucket\', \'embed\','
             ' \'embed_hash_bucket\', \'linear_sparse\', '
             '\'sparse_segment_reduce\', \'sparse_concat\'')
    return parser.parse_args()


OPT_PASSES = {
    'dnn_sparse': DnnSparseEmbeddingPatternRewriter,
    'dnn_hash_bucket': DnnEmbeddingWithHashBucketPatternRewriter,
    'embed': EmbeddingPatternRewriter,
    'embed_hash_bucket': EmbeddingWithHashBucketPatternRewriter,
    'linear_sparse': LinearSparseEmbeddingPatternRewriter,
    'sparse_segment_reduce': KPSparseSegmentReducePatternRewriter,
    'sparse_concat': KPSparseConcatPatternRewriter,
    'sparse_dynamic_stitch': KPSparseDynamicStitchPatternRewriter,
}


def opt():
    args = parse_args()
    meta_graph = MetaGraph(args.input)

    for pass_name in args.passes:
        if pass_name not in OPT_PASSES:
            raise TypeError(f'Pass \'{pass_name}\' not found')
        OPT_PASSES[pass_name](meta_graph.graph)

    os.makedirs(args.output, exist_ok=True)
    meta_graph.save(args.output)
    
    convert_pbtxt_to_saved_model(args.output)


if __name__ == '__main__':
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    opt()
