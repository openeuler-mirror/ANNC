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
    command.extend(["--onnx-mlir-opt", os.environ["ANNC_ONNX_MLIR_OPT"],
                    "--framework-opt", os.environ["ANNC_FRAMEWORK_OPT"]])
    stages = []
    source_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()

    def invoke(label, argv):
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
            "source_sha256": source_sha256,
            "stages": stages,
        }, indent=2) + "\n")
        assert result.returncode == 0, result.stderr

    invoke("framework_boundary", command)
    boundary = json.loads(metadata.read_text())
    assert set(boundary["dialect_counts"]) <= {
        "stablehlo", "atir", "builtin", "func"
    }
    onnx = pytest.importorskip("onnx")
    original = onnx.load(str(model), load_external_data=False)
    expected = sum(
        n.domain == "com.microsoft" and n.op_type == "FusedTensordotMatMul"
        for n in original.graph.node
    )
    assert boundary["op_counts"].get("atir.opaque", 0) == expected
    roundtrip = tmp_path / "roundtrip.mlir"
    # Mixed IR and ANNC source attributes require the paired pass runner.
    invoke(
        "provenance_ir_roundtrip",
        [
            os.environ["ANNC_FRAMEWORK_OPT"], str(stablehlo),
            "--verify-framework-stablehlo", "--mlir-print-op-generic",
            "-o", str(roundtrip),
        ],
    )
    assert roundtrip.read_text().count('"atir.opaque"') == expected
    assert 'annc.provenance = "onnx-node-name/2"' in roundtrip.read_text()
    assert hashlib.sha256(model.read_bytes()).hexdigest() == source_sha256
    assert stablehlo.stat().st_size > 0
    assert roundtrip.stat().st_size > 0


def test_pure_hlo_portable_roundtrip(tmp_path):
    translate = os.environ.get("ANNC_STABLEHLO_TRANSLATE")
    if not translate:
        pytest.skip("requires the upstream StableHLO serialization tool")
    source = tmp_path / "pure.mlir"
    portable = tmp_path / "pure.bc"
    restored = tmp_path / "restored.mlir"
    source.write_text("""module {
  func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.add %x, %x : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
""")
    for command in (
        [
            translate, "--serialize", "--target=1.8.0", str(source),
            "-o", str(portable),
        ],
        [translate, "--deserialize", str(portable), "-o", str(restored)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
    assert portable.stat().st_size > 0
    assert "stablehlo.add" in restored.read_text()
