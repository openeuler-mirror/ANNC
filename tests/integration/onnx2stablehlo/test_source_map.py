"""ONNX conversion preserves public outputs and proven tensor aliases."""

import json
import os
import re
import subprocess

import pytest


def test_real_producer_preserves_public_results_and_proven_aliases(tmp_path):
    if os.environ.get("ANNC_SOURCE_MAP_REAL_E2E") != "1":
        pytest.skip("set ANNC_SOURCE_MAP_REAL_E2E=1 to run the external producer")
    onnx = pytest.importorskip("onnx")
    h = onnx.helper
    dtype = onnx.TensorProto.FLOAT
    source = tmp_path / "model.onnx"
    onnx.save(
        h.make_model(
            h.make_graph(
                [
                    h.make_node("Split", ["x", "splits"], ["a", "b"], name="split"),
                    h.make_node("Identity", ["a"], ["alias"], name="identity"),
                    h.make_node("Add", ["alias", "b"], ["y"], name="sum"),
                ],
                "main",
                [h.make_tensor_value_info("x", dtype, [4])],
                [h.make_tensor_value_info(name, dtype, [2]) for name in ("alias", "b", "y")],
                [h.make_tensor("splits", onnx.TensorProto.INT64, [2], [2, 2])],
            ),
            opset_imports=[h.make_opsetid("", 13)],
            ir_version=9,
        ),
        source,
    )
    original = source.read_bytes()
    output = tmp_path / "converted"
    output.mkdir()
    hlo, metadata = output / "stablehlo.mlir", output / "boundary.json"
    result = subprocess.run(
        [
            os.environ["ANNC_SOURCE_MAP_PRODUCER"],
            str(source),
            "-o", str(hlo),
            "--metadata", str(metadata),
            "--framework-opt", os.environ["ANNC_SOURCE_MAP_FRAMEWORK_OPT"],
            "--onnx-mlir-opt", os.environ["ANNC_ONNX_MLIR_OPT"],
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert source.read_bytes() == original
    assert {file.name for file in output.iterdir()} == {hlo.name, metadata.name}

    ir = hlo.read_text()
    aliases = {}
    for result_op, match in enumerate(
        re.finditer(r"annc\.result_tensors = (\[(?:\[[^\[\]]*\](?:,\s*)?)*\])", ir)
    ):
        for slot, names in enumerate(json.loads(match[1])):
            for name in names:
                assert name not in aliases, f"duplicate tensor alias: {name}"
                aliases[name] = (result_op, slot)
    assert aliases["a"] == aliases["alias"]
    assert aliases["a"] != aliases["b"]
    assert (
        "function_type = (tensor<4xf32>) -> "
        "(tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)"
    ) in ir
    assert (
        'res_attrs = [{onnx.name = "alias"}, {onnx.name = "b"}, {onnx.name = "y"}]'
    ) in ir
    assert 'source_ids = ["split"]' in ir
    assert 'source_ids = ["sum"]' in ir
