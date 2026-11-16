"""Process-boundary contract for the standalone ONNX to StableHLO driver."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
DRIVER = Path(os.environ.get(
    "ANNC_ONNX2STABLEHLO", ROOT / "build/bin/annc-onnx2stablehlo"))
pytestmark = pytest.mark.skipif(
    not DRIVER.is_file(), reason="standalone StableHLO boundary driver is not built")


def _tools(tmp_path: Path, *, fail_optimizer: bool = False):
    tools = tmp_path / "tools"
    tools.mkdir()
    shutil.copy2(DRIVER, tools / DRIVER.name)
    log = tmp_path / "argv.log"
    stage = f"""#!/bin/sh
set -eu
printf '%s\\n' "$0 $*" >> "{log}"
if [ "{str(fail_optimizer).lower()}" = true ] && [ "$0" = "{tools / 'onnx-mlir-opt'}" ]; then
  exit 17
else
  output=''
  previous=''
  for arg in "$@"; do
    if [ "$previous" = '-o' ]; then output="$arg"; fi
    previous="$arg"
  done
  printf 'module {{\\n  func.func @main(%%arg0: tensor<2xf32>) -> tensor<2xf32> {{\\n    %%0 = "stablehlo.add"(%%arg0, %%arg0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>\\n    return %%0 : tensor<2xf32>\\n  }}\\n}}\\n' > "$output"
fi
"""
    for name in ("onnx-mlir-opt", "annc-onnx-framework-opt"):
        path = tools / name
        path.write_text(stage)
        path.chmod(0o755)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"minimal onnx graph")
    return tools, model, log


def _run(tmp_path: Path, *, fail_optimizer: bool = False):
    tools, model, log = _tools(tmp_path, fail_optimizer=fail_optimizer)
    output = tmp_path / "stablehlo.mlir"
    metadata = tmp_path / "boundary.json"
    env = dict(os.environ, ANNC_ONNX_MLIR_VERSION="v0.5.0.3",
               ANNC_STABLEHLO_PRODUCER_VERSION="1.8.9",
               ANNC_LLVM_MLIR_VERSION="20.1.8")
    result = subprocess.run(
        [
            str(tools / DRIVER.name), str(model), "-o", str(output),
            "--metadata", str(metadata), "--tool-dir", str(tools),
        ],
        text=True, capture_output=True, timeout=30, env=env,
    )
    return result, output, metadata, log


def test_external_tools_publish_stablehlo_and_metadata(tmp_path):
    result, output, metadata, log = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'stablehlo.add' in output.read_text()
    report = json.loads(metadata.read_text())
    assert report["schema"] == "annc-stablehlo-boundary/1"
    assert report["producer"]["tool"] == "onnx-mlir"
    assert report["stages"]["source_import"]["exit_code"] == 0
    assert report["stages"]["onnx_ir_optimization"]["exit_code"] == 0
    assert report["stages"]["onnx_to_stablehlo"]["exit_code"] == 0
    assert report["stages"]["preprocessing_passes"]["exit_code"] == 0
    assert report["stages"]["residual_lowering"]["exit_code"] == 0
    assert report["stages"]["source_finalization"]["exit_code"] == 0
    assert report["op_counts"]["stablehlo.add"] == 1
    assert report["dialect_counts"]["stablehlo"] == 1
    assert log.read_text().count("--annc-import-onnx") == 1
    assert "onnx-mlir-opt" in log.read_text()
    assert "annc-onnx-hybrid-opt" not in log.read_text()
    assert "framework-protect-custom-calls" not in log.read_text()
    assert "framework-restore-custom-calls" not in log.read_text()
    assert "atir." not in output.read_text()
    assert not list(tmp_path.glob("annc-onnx2stablehlo-*"))


def test_framework_boundary_never_invokes_atir_helper(tmp_path):
    result, output, _, log = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert "annc-onnx-hybrid-opt" not in log.read_text()
    assert "atir." not in output.read_text()


def test_failed_external_stage_preserves_existing_output(tmp_path):
    tools, model, _ = _tools(tmp_path, fail_optimizer=True)
    output = tmp_path / "stablehlo.mlir"
    metadata = tmp_path / "boundary.json"
    output.write_text("previous output\n")
    result = subprocess.run(
        [
            str(tools / DRIVER.name), str(model), "-o", str(output),
            "--metadata", str(metadata), "--tool-dir", str(tools),
        ],
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode != 0
    assert "ONNX IR optimization" in result.stderr
    assert output.read_text() == "previous output\n"
    assert not metadata.exists()
    failure = json.loads(Path(str(metadata) + ".failure.json").read_text())
    assert failure["exit_code"] == 17
    assert failure["stage"] == "ONNX IR optimization"


def test_metadata_publication_failure_restores_output(tmp_path):
    tools, model, _ = _tools(tmp_path)
    output = tmp_path / "stablehlo.mlir"
    output.write_text("previous output\n")
    metadata = tmp_path / "metadata-directory"
    # The destination changes after the preflight check, while a child runs.
    helper = tools / "annc-onnx-framework-opt"
    helper.write_text(helper.read_text() + f'\nmkdir -p "{metadata}"\n')
    result = subprocess.run(
        [
            str(tools / DRIVER.name), str(model), "-o", str(output),
            "--metadata", str(metadata), "--tool-dir", str(tools),
        ],
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode != 0
    assert f"cannot publish {metadata}" in result.stderr
    assert metadata.is_dir()
    assert output.read_text() == "previous output\n"
