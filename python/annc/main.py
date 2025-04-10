import os
import argparse
from optimize.rec_embedding import EmbeddingV1PatternRewriter
from optimize.rec_embedding import EmbeddingV2PatternRewriter
from optimize.rec_embedding import EmbeddingV3PatternRewriter
from optimize.rec_embedding import LinearEmbeddingV1PatternRewriter
from optimize.graph import MetaGraph


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
        '--passes',
        nargs='+',
        help=
        'opt: \'--sparse-v1\', \'--sparse-v2\', \'--sparse-v3\','
        ' \'--linear-v1\''
    )
    return parser.parse_args()


OPT_PASSES = {
    '--sparse-v1': EmbeddingV1PatternRewriter,
    '--sparse-v2': EmbeddingV2PatternRewriter,
    '--sparse-v3': EmbeddingV3PatternRewriter,
    '--linear-v1': LinearEmbeddingV1PatternRewriter,
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


if __name__ == '__main__':
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    opt()
