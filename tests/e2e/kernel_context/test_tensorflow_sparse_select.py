"""sparse_select through TensorFlow fusion, compilation and kernel execution."""

import pytest

from tf_graph import build_graph, kernel_spec
from tf_pipeline import run_pipeline


CASES = {
    "sparse_select": kernel_spec("kp_sparse_select", "KPFusedSparseSelect", "int32_t"),
}


def _build(case, spec, tf, np, input_, feeds, variants):
    a = input_("a", tf.int32, [None], [0, 0, -1, 2])
    b = input_("b", tf.int32, [None], [1, 7, 9, 0])
    c = input_("c", tf.int32, [None], [1, 2, 3, 4])
    x = tf.reshape(a, [-1, 1], name="x")
    rb = tf.reshape(b, [-1, 1])
    rc = tf.reshape(c, [-1, 1])
    fill = lambda value: tf.fill(tf.raw_ops.Shape(input=value), 1.0)
    y1 = tf.where(tf.equal(rb, 7), fill(rb), tf.cast(tf.greater(x, 1), tf.float32))
    y = tf.where(tf.equal(rb, 9), fill(rb), y1, name="y")
    right = tf.where(tf.equal(rc, 10), fill(rc), fill(rc))
    empty = tf.fill([4, 0], 0.0)
    w = tf.concat([y, right, empty], axis=1, name="w")
    outputs = [(x, "out_x"), (y, "out_y"), (w, "out_w")]
    frontend = [tensor.name for tensor, _ in outputs]
    frontend.remove("y:0")
    return outputs, frontend


def make_graph(case):
    return build_graph(case, CASES[case], _build)


@pytest.mark.parametrize("case", sorted(CASES))
def test_production_kernel_full_pipeline(case, tmp_path):
    run_pipeline(CASES[case], make_graph, case, tmp_path)
