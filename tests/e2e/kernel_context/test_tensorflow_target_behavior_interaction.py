"""target_behavior_interaction through TensorFlow fusion, compilation and kernel execution."""

import pytest

from tf_graph import build_graph, kernel_spec
from tf_pipeline import run_pipeline


CASES = {
    "target_behavior": kernel_spec(
        "kp_target_behavior_interaction", "KPFusedTargetBehaviorInteraction"
    ),
}


def _build(case, spec, tf, np, input_, feeds, variants):
    batch = input_("batch_input", tf.float32, [None, 1, 2], [[[1, 2]], [[-1, 1]]])
    weight = input_("weight", tf.float32, [2, 2], [[1, -1], [2, -2]])
    bias = input_("bias", tf.float32, [2], [1, 1])
    tile_input = input_("tile_input", tf.float32, [None, 1, 2], [[[3, 4]], [[-2, 1]]])
    product = tf.raw_ops.BatchMatMulV2(x=batch, y=weight)
    biased = tf.add(product, bias)
    relu = tf.multiply(tf.add(tf.abs(biased), biased), 0.5)
    tiled = tf.tile(tile_input, [1, 1, 1])
    result = tf.concat(
        [relu, tiled, tf.subtract(relu, tiled), tf.multiply(relu, tiled)],
        axis=-1,
        name="output",
    )
    outputs = [(result, "output")]
    frontend = [tensor.name for tensor, _ in outputs]
    return outputs, frontend


def make_graph(case):
    return build_graph(case, CASES[case], _build)


@pytest.mark.parametrize("case", sorted(CASES))
def test_production_kernel_full_pipeline(case, tmp_path):
    run_pipeline(CASES[case], make_graph, case, tmp_path)
