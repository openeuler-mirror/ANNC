"""Context V3 MatMul+Add+Relu multi-output fusion through TensorFlow."""

import pytest

from tf_graph import build_graph, kernel_spec
from tf_pipeline import run_pipeline


CASES = {
    "matmul_add_relu_v3": kernel_spec(
        "matmul_add_relu",
        "MatMulAddReluWithAddOutput",
        mode="jit",
        env={"ANNC_GEMM": "2"},
    ),
}


def _build(case, spec, tf, np, input_, feeds, variants):
    lhs = input_("lhs", tf.float32, [None, 3], [[1, -2, 3], [0.5, 4, -1]])
    rhs = input_(
        "rhs",
        tf.float32,
        [3, 4],
        [[2, -1, 0.5, 1], [1.5, 3, -2, 0.25], [0.5, 1, 2, -1]],
    )
    bias = input_("bias", tf.float32, [4], [0.5, -1, 0.25, 0.75])
    product = tf.matmul(lhs, rhs, name="product")
    added = tf.add(product, bias, name="added")
    relu = tf.nn.relu(added, name="relu")
    outputs = [(added, "add"), (relu, "relu")]
    variants.append(("one_row", {"lhs:0": feeds["lhs:0"][:1]}))
    return outputs, [tensor.name for tensor, _ in outputs]


def make_graph(case):
    return build_graph(case, CASES[case], _build)


@pytest.mark.parametrize("case", sorted(CASES))
def test_context_v3_matmul_add_relu_full_pipeline(case, tmp_path):
    run_pipeline(CASES[case], make_graph, case, tmp_path)
