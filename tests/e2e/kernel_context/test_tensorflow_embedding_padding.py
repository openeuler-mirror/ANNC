"""embedding_padding through TensorFlow fusion, compilation and kernel execution."""

import pytest

from tf_graph import build_graph, kernel_spec, scalar_slice
from tf_pipeline import run_pipeline


CASES = {
    "padding_fast" if fast else "padding": kernel_spec(
        "kp_embedding_padding_fast" if fast else "kp_embedding_padding",
        "KPFusedEmbeddingPadding" + ("Fast" if fast else ""),
        fast=fast,
    )
    for fast in (False, True)
}


def _build(case, spec, tf, np, input_, feeds, variants):
    origin = input_("origin_shape", tf.int64, [2], [3, 2])
    rows = input_("input_rows", tf.int32, [], 2)
    data = input_("data", tf.float32, [None, 2], [[1, 2], [3, 4]])
    width = tf.constant(2, name="pack")
    shape = input_("reshape_sizes", tf.int32, [2], [-1, 2])
    origin_rows = scalar_slice(tf, tf.cast(origin, tf.int32), [0], "origin_rows")
    padding = tf.subtract(origin_rows, rows, name="padding_rows")
    padded = tf.concat([data, tf.fill(tf.stack([padding, width]), 0.0)], 0)
    reshaped = tf.reshape(padded, shape, name="padded")
    if spec["fast"]:
        result = scalar_slice(tf, tf.raw_ops.Shape(input=reshaped), [0], "rows")
        outputs = [(padding, "out_padding_rows"), (result, "out_reshape_rows")]
    else:
        # The production matcher requires a ConcatV2 consumer.
        tf.concat([reshaped, tf.zeros([3, 1])], 1, name="consumer")
        outputs = [(padding, "out_padding_rows"), (reshaped, "out_data")]
    variants.append(("no_padding", {"origin_shape:0": np.array([2, 2], np.int64)}))
    frontend = [tensor.name for tensor, _ in outputs]
    if case == "padding":
        frontend.append("consumer:0")
    return outputs, frontend


def make_graph(case):
    return build_graph(case, CASES[case], _build)


@pytest.mark.parametrize("case", sorted(CASES))
def test_production_kernel_full_pipeline(case, tmp_path):
    run_pipeline(CASES[case], make_graph, case, tmp_path)
