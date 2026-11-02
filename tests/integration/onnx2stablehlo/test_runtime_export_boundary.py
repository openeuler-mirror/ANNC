"""Full ORT GetCapability exports; never substitute generated ONNX fixtures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

ROOT = Path(__file__).resolve().parents[3]
BOUNDARY = Path(os.environ.get(
    "ANNC_ONNX2STABLEHLO", ROOT / "build/bin/annc-onnx2stablehlo"))


@pytest.mark.parametrize("model_index", range(1, 5))
def test_real_runtime_export_model_boundary(tmp_path, model_index):
    model_dir = os.environ.get("ANNC_ORT_EXPORTED_MODEL_DIR")
    model_name = os.environ.get("ANNC_ORT_EXPORTED_MODEL")
    if model_dir:
        model = Path(model_dir) / f"runtime_model{model_index}.onnx"
    elif model_name and model_index == 1:
        model = Path(model_name)
    else:
        pytest.skip("set ANNC_ORT_EXPORTED_MODEL_DIR to the real model1–4 exports")
    assert model.is_file(), f"ORT exported model is missing: {model}"
    assert BOUNDARY.is_file(), "build the StableHLO producer first"
    stablehlo = tmp_path / "stablehlo.mlir"
    metadata = tmp_path / "boundary.json"
    command = [str(BOUNDARY), str(model), "-o", str(stablehlo),
               "--metadata", str(metadata)]
    command.extend(["--onnx-mlir", os.environ["ANNC_ONNX_MLIR"],
                    "--onnx-mlir-opt", os.environ["ANNC_ONNX_MLIR_OPT"],
                    "--framework-opt", os.environ["ANNC_FRAMEWORK_OPT"]])
    stages = []
    portable = tmp_path / "model.stablehlo.portable"
    translate = os.environ.get("ANNC_STABLEHLO_TRANSLATE")
    assert translate, "set ANNC_STABLEHLO_TRANSLATE to the upstream serialization tool"
    for label, argv in (
        ("framework_boundary", command),
        ("portable_serialization", [translate, "--serialize", "--target=1.8.0",
                                    str(stablehlo), "-o", str(portable)]),
        ("portable_deserialization", [translate, "--deserialize", str(portable),
                                      "-o", str(tmp_path / "roundtrip.mlir")]),
    ):
        start = time.monotonic()
        result = subprocess.run(argv, text=True, capture_output=True, timeout=600)
        (tmp_path / f"{label}.stdout").write_text(result.stdout)
        (tmp_path / f"{label}.stderr").write_text(result.stderr)
        stages.append({"stage": label, "command": argv,
                       "exit_code": result.returncode,
                       "elapsed_seconds": time.monotonic() - start})
        (tmp_path / "run.json").write_text(json.dumps({
            "source_model": str(model.resolve()),
            "source_bytes": model.stat().st_size,
            "source_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "stages": stages,
        }, indent=2) + "\n")
        assert result.returncode == 0, result.stderr
    boundary = json.loads(metadata.read_text())
    assert set(boundary["dialect_counts"]) <= {"stablehlo", "builtin", "func"}
    assert 'atir.' not in stablehlo.read_text()
    assert portable.stat().st_size > 0
    assert stablehlo.stat().st_size > 0
