"""sparse_segment_reduce through TensorFlow fusion, compilation and kernel execution."""

import itertools

import pytest

from tf_graph import build_graph, kernel_spec, scalar_slice, column
from tf_pipeline import run_pipeline


CASES = {
    f"segment_i{bits}_{combiner}": kernel_spec(
        "kp_sparse_segment_reduce",
        f"KPFusedSparseSegmentReduceI{bits}{combiner.title()}",
        f"float_int{bits}_t",
        bits=bits,
        combiner=combiner,
    )
    for bits, combiner in itertools.product((32, 64), ("sum", "mean"))
}


def _build(case, spec, tf, np, input_, feeds, variants):
    keys = input_("keys", tf.int64, [None, 2], [[0, 0], [0, 0], [0, 2]])
    begin = input_("begin", tf.int32, [2], [0, 1])
    data = input_("data", tf.float32, [None, 2], [[1, 2], [3, 4], [5, 6]])
    dtype = getattr(tf, f'int{spec["bits"]}')
    indices = input_("indices", dtype, [None], [0, 1, 2])
    begin1 = input_("begin_1", tf.int32, [1], [0])
    segs = tf.cast(column(tf, keys, begin, "segments"), tf.int32)
    op = (
        tf.raw_ops.SparseSegmentMean
        if spec["combiner"] == "mean"
        else tf.raw_ops.SparseSegmentSum
    )
    reduced = op(data=data, indices=indices, segment_ids=segs, name="reduced")
    size = scalar_slice(tf, tf.raw_ops.Shape(input=reduced), begin1, "dimension")
    outputs = [(reduced, "output"), (size, "slice_output")]
    variants.append(("embedding_width", {"begin_1:0": np.array([1], np.int32)}))
    frontend = [tensor.name for tensor, _ in outputs]
    return outputs, frontend


def make_graph(case):
    return build_graph(case, CASES[case], _build)


@pytest.mark.parametrize("case", sorted(CASES))
def test_production_kernel_full_pipeline(case, tmp_path):
    run_pipeline(CASES[case], make_graph, case, tmp_path)
