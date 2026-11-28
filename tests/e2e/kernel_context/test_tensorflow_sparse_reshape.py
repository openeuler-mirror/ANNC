"""sparse_reshape through TensorFlow fusion, compilation and kernel execution."""

import pytest

from tf_graph import build_graph, kernel_spec, scalar_slice, column
from tf_pipeline import run_pipeline


CASES = {
    "sparse_reshape_i64": kernel_spec(
        "kp_sparse_reshape", "KPFusedSparseReshapeI64", "int64_t",
    )
}


def _build(case, spec, tf, np, input_, feeds, variants):
    dtype = tf.int64
    keys = input_("keys", tf.int64, [None, 2], [[0, 1], [1, 2]])
    begin = input_("begin", tf.int32, [2], [0, 1])
    pack = input_("pack_const", dtype, [], 3)
    new_shape = input_("new_shape", tf.int64, [2], [-1, 2])
    count = scalar_slice(tf, tf.raw_ops.Shape(input=keys, out_type=dtype), [0], "count")
    limit = tf.constant(2, dtype=tf.int32)
    row_ids = tf.raw_ops.Range(start=0, limit=limit, delta=1)
    row_ids = tf.raw_ops.Cast(x=tf.reshape(row_ids, [-1, 1]), DstT=tf.int64)
    col = tf.reshape(column(tf, keys, begin, "column"), [-1, 1])
    indices = tf.concat([row_ids, col], -1)
    dense_shape = tf.raw_ops.Cast(x=tf.stack([count, pack]), DstT=tf.int64)
    out_i, out_s = tf.raw_ops.SparseReshape(
        input_indices=indices,
        input_shape=dense_shape,
        new_shape=new_shape,
        name="sparse",
    )
    outputs = [(out_i, "out_indices"), (out_s, "out_shape")]
    variants.append(
        ("inferred_other_axis", {"new_shape:0": np.array([2, -1], np.int64)})
    )
    frontend = [tensor.name for tensor, _ in outputs]
    return outputs, frontend


def make_graph(case):
    return build_graph(case, CASES[case], _build)


@pytest.mark.parametrize("case", sorted(CASES))
def test_production_kernel_full_pipeline(case, tmp_path):
    run_pipeline(CASES[case], make_graph, case, tmp_path)
