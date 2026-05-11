import os
import argparse
from annc.optimize.rec_embedding import DnnSparseEmbeddingPatternRewriter
from annc.optimize.rec_embedding import DnnEmbeddingWithHashBucketPatternRewriter
from annc.optimize.rec_embedding import EmbeddingPatternRewriter
from annc.optimize.rec_embedding import EmbeddingWithHashBucketPatternRewriter
from annc.optimize.rec_embedding import LinearSparseEmbeddingPatternRewriter
from annc.optimize.rec_embedding import KPSoftmaxRewriter
from annc.optimize.rec_embedding import LookupEmbeddingWithHashPatternRewriter
from annc.optimize.constant_folding import LayoutMatmulRewriter
from annc.optimize.graph import MetaGraph


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
        help='opt: \'lookup_embedding_hash\', \'dnn_sparse\', \'dnn_hash_bucket\', \'embed\','
        ' \'embed_hash_bucket\', \'linear_sparse\', \'softmax\', \'layout_matmul\'')
    return parser.parse_args()


OPT_PASSES = {
    'dnn_sparse': DnnSparseEmbeddingPatternRewriter,
    'dnn_hash_bucket': DnnEmbeddingWithHashBucketPatternRewriter,
    'embed': EmbeddingPatternRewriter,
    'embed_hash_bucket': EmbeddingWithHashBucketPatternRewriter,
    'lookup_embedding_hash': LookupEmbeddingWithHashPatternRewriter,
    'linear_sparse': LinearSparseEmbeddingPatternRewriter,
    'softmax': KPSoftmaxRewriter,
}


def opt():
    args = parse_args()
    meta_graph = MetaGraph(args.input)

    passes = list(args.passes)
    if 'layout_matmul' in passes:
        # layout_matmul must run last because it modifies TF variable weights
        # and should operate on the graph after all structural passes have run.
        passes.remove('layout_matmul')
        passes.append('layout_matmul')

    layout_matmul_rewriter = None
    for pass_name in passes:
        if pass_name == 'layout_matmul':
            layout_matmul_rewriter = LayoutMatmulRewriter(args.input)
            layout_matmul_rewriter(meta_graph.graph)
        else:
            if pass_name not in OPT_PASSES:
                raise TypeError(f'Pass \'{pass_name}\' not found')
            OPT_PASSES[pass_name](meta_graph.graph)

    os.makedirs(args.output, exist_ok=True)
    meta_graph.save(args.output)

    if layout_matmul_rewriter:
        layout_matmul_rewriter.save(args.output)


if __name__ == '__main__':
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    opt()
