"""StableHLO compilation publishes kernels and sites for the ORT EP."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
PIPELINE = os.environ.get("ANNC_STABLEHLO_PIPELINE")
pytestmark = pytest.mark.skipif(
    not PIPELINE, reason="requires the StableHLO pipeline"
)


def bmm_model(directory):
    h, t = onnx.helper, onnx.TensorProto
    model = h.make_model(
        h.make_graph(
            [h.make_node("MatMul", ["x", "w"], ["mm"], name="matmul"),
             h.make_node("Add", ["mm", "b"], ["y"], name="add")],
            "bmm",
            [h.make_tensor_value_info("x", t.FLOAT, [2, 5, 12])],
            [h.make_tensor_value_info("y", t.FLOAT, [2, 5, 16])],
            [onnx.numpy_helper.from_array(np.ones(shape, np.float32), name)
             for name, shape in [("w", [2, 12, 16]), ("b", [2, 1, 16])]],
        ),
        opset_imports=[h.make_opsetid("", 18)], ir_version=9,
    )
    path = directory / "input.onnx"
    onnx.save(model, path)
    return path


def independent_pipeline(directory):
    """Stage the driver with only the StableHLO frontend tools."""
    directory.mkdir()
    original = Path(PIPELINE).resolve()
    entry = directory / original.name
    shutil.copy2(original, entry)
    # Keep compiler entries within the isolated tool prefix.
    for name in ["annc-onnx2stablehlo", "annc-stablehlo-opt"]:
        shutil.copy2(original.with_name(name), directory / name)
    for name in [
        "annc-onnx-framework-opt",
        "annc-asm", "annc", "annc-gemm-epilogue-kernel-compile",
    ]:
        (directory / name).symlink_to(original.with_name(name))
    return entry


def test_toolchain_info_distinguishes_frontends():
    def info(value):
        env = dict(os.environ)
        if value is None:
            env.pop("ANNC_ONNX_USE_STABLEHLO", None)
        else:
            env["ANNC_ONNX_USE_STABLEHLO"] = value
        return json.loads(subprocess.check_output(
            [PIPELINE, "--toolchain-info"], env=env, text=True,
        ))

    atir = info(None)
    stablehlo = info("1")
    assert atir == info("0")
    assert atir["frontend"] == "atir"
    assert stablehlo["frontend"] == "stablehlo"
    assert "annc-onnx2stablehlo" not in {tool["name"] for tool in atir["tools"]}
    assert {"annc-onnx2stablehlo", "annc-stablehlo-opt"} <= {
        tool["name"] for tool in stablehlo["tools"]
    }


def test_pipeline_returns_failed_stage_status_without_compiling(tmp_path):
    pipeline = independent_pipeline(tmp_path / "bin")
    reader = pipeline.with_name("annc-stablehlo-opt")
    reader.unlink()
    reader.write_text("#!/bin/sh\nexit 17\n")
    reader.chmod(0o755)
    model = bmm_model(tmp_path)
    compiled = tmp_path / "compiled"
    result = subprocess.run(
        [str(pipeline), "--model", str(model), "--out-dir", str(compiled)],
        env=dict(os.environ, ANNC_ONNX_USE_STABLEHLO="1",
                 ANNC_EP_FUSION_BATCHED_MATMUL_ADD="1"),
        capture_output=True, text=True,
    )
    assert result.returncode == 17, result.stderr
    assert "StableHLO fusion failed (exit 17)" in result.stderr
    assert (compiled / "model.stablehlo.mlir").is_file()
    report = json.loads((compiled / "model.stablehlo.json").read_text())
    assert report["schema"] == "annc-stablehlo-boundary/1"
    assert not (compiled / "model.fused.mlirbc").exists()
    assert not (compiled / "model.low.mlir").exists()
    assert not (compiled / "sites.json").exists()
    assert not (compiled / "libannc_kernels.so").exists()
