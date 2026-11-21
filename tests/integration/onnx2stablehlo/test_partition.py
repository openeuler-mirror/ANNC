"""Check fusion boundaries using StableHLO converted from real ONNX graphs."""

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
READER = Path(
    os.environ.get("ANNC_STABLEHLO_READER", ROOT / "build/bin/annc-stablehlo-opt")
)
pytestmark = pytest.mark.skipif(
    not READER.is_file() or os.environ.get("ANNC_SOURCE_MAP_REAL_E2E") != "1",
    reason="requires paired reader and ANNC_SOURCE_MAP_REAL_E2E=1 producer tools",
)

sys.path.insert(0, str(ROOT / "tests/e2e/kernel_context"))


def convert_model(model):
    hlo = model.with_name("stablehlo.mlir")
    result = subprocess.run(
        [
            os.environ["ANNC_SOURCE_MAP_PRODUCER"], str(model),
            "-o", str(hlo), "--metadata", str(hlo.with_suffix(".json")),
            "--framework-opt", os.environ["ANNC_SOURCE_MAP_FRAMEWORK_OPT"],
            "--onnx-mlir-opt", os.environ["ANNC_ONNX_MLIR_OPT"],
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return hlo


def read_hlo(source, *options, patterns=("BATCHED_MATMUL_ADD",)):
    env = dict(os.environ)
    for name in ("BATCHED_MATMUL_ADD", "LAYER_NORM", "MASKED_MUL", "TENSORDOT_MATMUL"):
        env[f"ANNC_EP_FUSION_{name}"] = "1" if name in patterns else "0"
    return subprocess.run(
        [str(READER), str(source), *options],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def make_bmm(model, *, boundary=None, consecutive_scales=False):
    onnx = pytest.importorskip("onnx")
    h, t = onnx.helper, onnx.TensorProto
    prefix = [2, 1] if consecutive_scales else [2]
    rows = "M" if consecutive_scales else 5
    data_shape = prefix + [rows, 12]
    weight_shape = prefix + [12, 16]
    bias_shape = prefix + [1, 16]
    output_shape = prefix + [rows, 16]
    inputs = [h.make_tensor_value_info("x", t.FLOAT, data_shape)]
    outputs = [h.make_tensor_value_info("y", t.FLOAT, output_shape)]
    constants = [
        h.make_tensor("w", t.FLOAT, weight_shape, [0.25] * (2 * 12 * 16)),
        h.make_tensor("b", t.FLOAT, bias_shape, [0.5] * (2 * 16)),
    ]
    nodes = [
        h.make_node("MatMul", ["x", "w"], ["mm"], name="n_mm"),
        h.make_node("Add", ["mm", "b"], ["y"], name="n_add"),
    ]
    if boundary == "overridable-weight":
        inputs.append(h.make_tensor_value_info("w", t.FLOAT, weight_shape))
    if boundary == "side-consumer":
        nodes.append(h.make_node("Neg", ["mm"], ["neg"], name="n_neg"))
        outputs.append(h.make_tensor_value_info("neg", t.FLOAT, output_shape))
    value_info = []
    if consecutive_scales:
        # Only the second scale may be folded into the weights.
        nodes[0].input[0] = "scaled"
        nodes[:0] = [
            h.make_node("Mul", ["x", "outer_scale"], ["before_mul"], name="n_outer_mul"),
            h.make_node("Mul", ["before_mul", "scale"], ["scaled"], name="n_mul"),
        ]
        constants.extend([
            h.make_tensor("outer_scale", t.FLOAT, [], [2.0]),
            h.make_tensor("scale", t.FLOAT, [], [0.5]),
        ])
        value_info.append(h.make_tensor_value_info("before_mul", t.FLOAT, data_shape))
    graph = h.make_graph(nodes, "bmm", inputs, outputs, constants, value_info=value_info)
    onnx_model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=7)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, model)


@pytest.fixture(scope="module")
def bmm_hlo(tmp_path_factory):
    model = tmp_path_factory.mktemp("bmm") / "model.onnx"
    make_bmm(model)
    return convert_model(model)


def test_bmmadd_binds_without_original_files(tmp_path):
    model = tmp_path / "model.onnx"
    make_bmm(model)
    hlo = convert_model(model)
    partitioned = tmp_path / "partitioned.mlir"
    matched = read_hlo(hlo, "--annc-stablehlo-partition", "-o", str(partitioned))
    assert matched.returncode == 0, matched.stderr
    model.unlink()
    hlo.unlink()

    bound = tmp_path / "bound.mlir"
    result = read_hlo(partitioned, "--annc-stablehlo-bind-onnx", "-o", str(bound))
    assert result.returncode == 0, result.stderr
    assert "annc.onnx_bound" in bound.read_text()
    extracted = read_hlo(bound, "--annc-stablehlo-kernels")
    assert extracted.returncode == 0, extracted.stderr
    assert 'member_node_names = ["n_add", "n_mm"]' in extracted.stdout
    assert 'abi = "annc_context_v3"' in extracted.stdout
    assert extracted.stdout.count('role = "fixed"') == 2
    assert 'tf_name = "x"' in extracted.stdout and 'tf_name = "y"' in extracted.stdout
    assert "call @__annc_builtin_" in extracted.stdout
    for attribute in ("annc.onnx_model", "annc.hlo_region", "annc.onnx_bound"):
        assert attribute not in extracted.stdout
    assert "stablehlo." not in extracted.stdout


@pytest.mark.parametrize("boundary", ["side-consumer", "overridable-weight"])
def test_bmmadd_rejects_unsafe_boundaries(tmp_path, boundary):
    model = tmp_path / "model.onnx"
    make_bmm(model, boundary=boundary)
    result = read_hlo(
        convert_model(model), "--annc-stablehlo-partition", "--annc-stablehlo-bind-onnx"
    )
    assert result.returncode == 0, result.stderr
    assert "annc.onnx_bound" not in result.stdout
    assert "stablehlo.dot_general" in result.stdout


@pytest.mark.parametrize(
    "pattern,replacement,reason",
    [
        (
            r'annc.result_tensors = \[\["y"\]\]',
            "annc.result_tensors = [[]]",
            "ambiguous or missing original output",
        ),
        (
            r'source_ids = \["n_add"\]',
            'source_ids = ["unproven"]',
            "unknown original source node in kernel region",
        ),
    ],
    ids=["missing-output-alias", "unknown-source-node"],
)
def test_binding_rejects_incomplete_provenance(bmm_hlo, tmp_path, pattern, replacement, reason):
    damaged = tmp_path / "damaged.mlir"
    text, count = re.subn(pattern, replacement, bmm_hlo.read_text())
    assert count > 0
    damaged.write_text(text)
    result = read_hlo(damaged, "--annc-stablehlo-partition", "--annc-stablehlo-bind-onnx")
    assert result.returncode == 0, result.stderr
    assert "annc.onnx_bound" not in result.stdout
    assert reason in result.stdout


@pytest.mark.parametrize(
    "pattern,replacement",
    [
        (r'annc\.onnx_model = "(?:[^"\\]|\\.)*",?\s*', ""),
        (r'annc\.onnx_model = "(?:[^"\\]|\\.)*"', 'annc.onnx_model = "invalid JSON"'),
        ("annc-model-index/2", "invalid-schema"),
    ],
    ids=["missing", "invalid-json", "schema"],
)
def test_binding_rejects_invalid_imported_index(bmm_hlo, tmp_path, pattern, replacement):
    source = tmp_path / "bad.mlir"
    text, count = re.subn(pattern, replacement, bmm_hlo.read_text())
    assert count == 1
    source.write_text(text)
    output = tmp_path / "bound.mlir"
    result = read_hlo(source, "--annc-stablehlo-bind-onnx", "-o", str(output))
    assert result.returncode != 0
    assert "StableHLO ONNX binding:" in result.stderr
    assert not output.exists()


def test_bmmadd_consecutive_scales_keep_outer_boundary(tmp_path):
    model = tmp_path / "model.onnx"
    make_bmm(model, consecutive_scales=True)
    result = read_hlo(
        convert_model(model), "--annc-stablehlo-partition", "--annc-stablehlo-bind-onnx"
    )
    assert result.returncode == 0, result.stderr
    assert "annc.onnx_bound" in result.stdout
    assert "annc.effective_weight" in result.stdout
    assert 'member_node_names = ["n_add", "n_mm", "n_mul"]' in result.stdout
    assert 'tf_name = "before_mul"' in result.stdout


def test_layer_norm_binding_ignores_unrelated_gemm(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from layer_norm_context_e2e import build_model

    model = tmp_path / "model.onnx"
    build_model(model, 4, 16)
    graph = onnx.load(model)
    h, t = onnx.helper, onnx.TensorProto
    del graph.graph.input[:]
    for name in ("raw", "other"):
        graph.graph.input.append(h.make_tensor_value_info(name, t.FLOAT, [4, 8]))
    graph.graph.node.insert(
        0, h.make_node("Concat", ["raw", "other"], ["x"], name="input_concat", axis=1)
    )
    graph.graph.initializer.append(h.make_tensor("w", t.FLOAT, [8, 1], [0.5] * 8))
    graph.graph.node.append(h.make_node("Gemm", ["raw", "w"], ["unused"], name="unrelated"))
    onnx.checker.check_model(graph)
    onnx.save(graph, model)
    result = read_hlo(
        convert_model(model), "--annc-stablehlo-partition", "--annc-stablehlo-bind-onnx",
        patterns=("LAYER_NORM",),
    )
    assert result.returncode == 0, result.stderr
    assert 'pattern = "layer_norm"' in result.stdout
    assert "annc.onnx_bound" in result.stdout


def test_dynamic_tensordot_requires_shape_product(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from ftdmm_context_e2e import make_wide_model

    model = tmp_path / "model.onnx"
    make_wide_model(model, 16)
    graph = onnx.load(model)
    del graph.graph.value_info[:]
    for value in list(graph.graph.input) + list(graph.graph.output):
        for axis in (0, 1):
            value.type.tensor_type.shape.dim[axis].dim_param = f"d{axis}"
    onnx.save(graph, model)
    hlo = convert_model(model)
    result = read_hlo(hlo, "--annc-stablehlo-partition", patterns=("TENSORDOT_MATMUL",))
    assert result.returncode == 0, result.stderr
    assert 'pattern = "tensordot_matmul"' in result.stdout

    text, count = re.subn(r'"stablehlo.multiply"', '"stablehlo.add"', hlo.read_text(), count=1)
    assert count == 1
    hlo.write_text(text)
    rejected = read_hlo(hlo, "--annc-stablehlo-partition", patterns=("TENSORDOT_MATMUL",))
    assert rejected.returncode == 0, rejected.stderr
    assert "annc.kernel" not in rejected.stdout


def test_tensordot_keeps_post_reshape_nonzero_bias(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from ftdmm_context_e2e import make_wide_model

    model = tmp_path / "model.onnx"
    make_wide_model(model, 1)
    graph = onnx.load(model)
    output = graph.graph.output[0]
    h, t = onnx.helper, onnx.TensorProto
    graph.graph.initializer.append(h.make_tensor("post_bias", t.FLOAT, [1], [-0.28962675]))
    graph.graph.node.append(
        h.make_node("Add", [output.name, "post_bias"], ["biased"], name="post_add")
    )
    output.name = "biased"
    onnx.save(graph, model)
    result = read_hlo(
        convert_model(model), "--annc-stablehlo-partition", "--annc-stablehlo-bind-onnx",
        patterns=("TENSORDOT_MATMUL",),
    )
    assert result.returncode == 0, result.stderr
    assert 'pattern = "tensordot_matmul"' in result.stdout and "annc.onnx_bound" in result.stdout
    members = re.findall(r"member_node_names = \[(.*?)\]", result.stdout)
    assert members and all('"post_add"' not in group for group in members)
    assert "stablehlo.add" in result.stdout


def test_private_tensordot_stays_outside_dependent_bmmadd(tmp_path):
    onnx = pytest.importorskip("onnx")
    h, t = onnx.helper, onnx.TensorProto
    nodes = [
        h.make_node(
            "FusedTensordotMatMul", ["x", "projection"], ["hidden"],
            name="framework_projection", domain="com.microsoft",
            contract_axes=[2], free_axes=[0, 1],
        ),
        h.make_node("MatMul", ["hidden", "w"], ["mm"], name="matmul"),
        h.make_node("Add", ["mm", "b"], ["y"], name="add"),
    ]
    graph = h.make_graph(
        nodes,
        "mixed",
        [h.make_tensor_value_info("x", t.FLOAT, [2, "M", 8])],
        [h.make_tensor_value_info("y", t.FLOAT, [2, "M", 16])],
        [
            h.make_tensor(name, t.FLOAT, shape, [0.25] * count)
            for name, shape, count in (
                ("projection", [8, 12], 8 * 12),
                ("w", [2, 12, 16], 2 * 12 * 16),
                ("b", [2, 1, 16], 2 * 16),
            )
        ],
        value_info=[h.make_tensor_value_info("hidden", t.FLOAT, [2, "M", 12])],
    )
    model = tmp_path / "model.onnx"
    onnx.save(h.make_model(
        graph, opset_imports=[h.make_opsetid("", 13), h.make_opsetid("com.microsoft", 1)],
        ir_version=7,
    ), model)
    bound = tmp_path / "bound.mlir"
    result = read_hlo(
        convert_model(model), "--annc-stablehlo-partition", "--annc-stablehlo-bind-onnx",
        "-o", str(bound), patterns=("BATCHED_MATMUL_ADD", "TENSORDOT_MATMUL"),
    )
    assert result.returncode == 0, result.stderr
    assert "atir.opaque" in bound.read_text()
    extracted = read_hlo(bound, "--annc-stablehlo-kernels")
    assert extracted.returncode == 0, extracted.stderr
    assert 'pattern = "batched_matmul_add"' in extracted.stdout
    assert 'member_node_names = ["add", "matmul"]' in extracted.stdout
    assert "atir.opaque" not in extracted.stdout
