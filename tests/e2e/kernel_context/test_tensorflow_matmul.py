"""Context V3 MatMul through TensorFlow fusion and kernel execution."""

import pytest

from tf_graph import build_graph, kernel_spec
from tf_pipeline import run_pipeline


CASES = {
    "matmul_v3": kernel_spec("matmul", "MatMul", mode="jit"),
}


def _build(case, spec, tf, np, input_, feeds, variants):
    lhs = input_("lhs", tf.float32, [None, 3], [[1, -2, 3], [0.5, 4, -1]])
    rhs = input_(
        "rhs",
        tf.float32,
        [3, 4],
        [[2, -1, 0.5, 1], [1.5, 3, -2, 0.25], [0.5, 1, 2, -1]],
    )
    product = tf.matmul(lhs, rhs, name="product")
    outputs = [(product, "output")]
    variants.append(("one_row", {"lhs:0": feeds["lhs:0"][:1]}))
    return outputs, [tensor.name for tensor, _ in outputs]


def make_graph(case):
    return build_graph(case, CASES[case], _build)


@pytest.mark.parametrize("case", sorted(CASES))
def test_context_v3_matmul_full_pipeline(case, tmp_path):
    run_pipeline(CASES[case], make_graph, case, tmp_path)
