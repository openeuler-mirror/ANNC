"""KDNN dense kernels through TensorFlow fusion and AOT execution."""

import pytest

from tf_graph import build_graph, kernel_spec
from tf_pipeline import run_pipeline


CASES = {
    "kdnn_matmul": kernel_spec(
        "matmul",
        "MatMul",
        suffix="kdnn_float",
        mode="aot",
        env={"ANNC_GEMM": "1"},
        fast_codegen="enable-kdnn=true",
        form="matmul",
    ),
    "kdnn_matmul_add": kernel_spec(
        "matmul_add",
        "MatMulAdd",
        suffix="kdnn_float",
        mode="aot",
        env={"ANNC_GEMM": "1"},
        fast_codegen="enable-kdnn=true",
        form="matmul_add",
    ),
    "kdnn_matmul_add_relu": kernel_spec(
        "matmul_add_relu",
        "MatMulAddRelu",
        suffix="kdnn_float",
        mode="aot",
        env={"ANNC_GEMM": "1"},
        fast_codegen="enable-kdnn=true",
        form="matmul_add_relu",
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
    result = tf.matmul(lhs, rhs, name="product")
    if spec["form"] in ("matmul_add", "matmul_add_relu"):
        bias = input_("bias", tf.float32, [4], [0.5, -1, 0.25, 0.75])
        result = tf.add(result, bias, name="added")
    if spec["form"] == "matmul_add_relu":
        result = tf.nn.relu(result, name="relu")
    outputs = [(result, "output")]
    variants.append(("one_row", {"lhs:0": feeds["lhs:0"][:1]}))
    return outputs, [tensor.name for tensor, _ in outputs]


def make_graph(case):
    return build_graph(case, CASES[case], _build)


@pytest.mark.parametrize("case", sorted(CASES))
def test_kdnn_dense_full_pipeline(case, tmp_path):
    run_pipeline(CASES[case], make_graph, case, tmp_path)
