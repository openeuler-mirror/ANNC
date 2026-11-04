import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "kp_fused_gather_match.pbtxt"
CONFIG = ROOT / "tests" / "FileCheck" / "Dialect" / "Atir" / (
    "config-fusion-kp-fused-gather.json"
)


def _build_dir():
    configured = os.environ.get("ANNC_BUILD_DIR")
    if configured:
        return Path(configured)
    for candidate in (ROOT / "build-v2", ROOT / "build"):
        if candidate.is_dir():
            return candidate
    return ROOT / "build"


def _binary(build_dir, name):
    path = build_dir / "bin" / name
    if not path.is_file():
        pytest.skip(f"{name} is not built: {path}")
    return path


def _run(command):
    completed = subprocess.run(
        [os.fspath(arg) for arg in command],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"command failed ({completed.returncode}): {' '.join(map(str, command))}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_kp_fused_gather_can_be_materialized_from_external_config(tmp_path):
    build_dir = _build_dir()
    tf2atir = _binary(build_dir, "annc-tf2atir")
    annc_opt = _binary(build_dir, "annc-opt")
    raw_atir = tmp_path / "model_raw_atir.mlir"
    fused_atir = tmp_path / "model_fused_atir.mlir"

    _run([
        tf2atir,
        FIXTURE,
        "--output_tensor",
        "restore_first_unique:0",
        "--output_tensor",
        "post_outer_gather:0",
        "-o",
        raw_atir,
    ])
    _run([
        annc_opt,
        raw_atir,
        "--atir-identity-canonicalize",
        f"--atir-config-fusion=config={CONFIG} run-builtin-after=false",
        "-o",
        fused_atir,
    ])

    text = fused_atir.read_text()
    assert 'func.func private @fused_kp_fused_gather_external_' in text
    assert 'abi = "annc_context_v3"' in text
    assert '!llvm.ptr' not in text
    assert text.count('role = "dynamic"') >= 3
    assert 'fusion.pattern = "kp_fused_gather_external"' in text
    assert 'tf_name = "first_unique:0"' in text
    assert 'tf_name = "first_unique:1"' in text
    assert 'tf_name = "outer_gather:0"' in text

    metadata_path = tmp_path / "fusion_metadata.json"
    _run([
        _binary(build_dir, "annc-fusion-metadata"), fused_atir,
        "-o", metadata_path,
    ])
    fusions = json.loads(metadata_path.read_text())["fusions"]
    assert len(fusions) == 1
    # Positional context records retain only graph endpoints and tensor facts.
    expected_fields = {"role", "tf_name", "shape", "rank", "dtype"}
    for record in fusions[0]["args"] + fusions[0]["outputs"]:
        assert set(record) == expected_fields

    main = text[text.rfind("func.func @main") :]
    assert "call @fused_kp_fused_gather_external_" not in main
    assert "atir.Unique" in main
    assert 'tf.name = "restore_first_unique"' in main
    assert main.count("atir.Gather") == 3

    lowered = tmp_path / "model_fast_codegen.mlir"
    _run([
        _binary(build_dir, "annc-asm"),
        fused_atir,
        "--atir-prune-func",
        "--atir-fast-codegen",
        "-o",
        lowered,
    ])
    lowered_text = lowered.read_text()
    assert 'custom.op_name = "KPFusedGather"' in lowered_text
    assert "atir.Unique" not in lowered_text


def test_config_fusion_rejects_retired_execution_v2(tmp_path):
    annc_opt = _binary(_build_dir(), "annc-opt")
    config = tmp_path / "retired_abi.json"
    config.write_text(json.dumps({"version": 1, "patterns": [{
        "name": "retired", "abi": "annc_execution_v2",
        "graph": ["out = Relu(input)"],
        "inputs": ["input"], "outputs": ["out"],
    }]}))
    source = tmp_path / "input.mlir"
    source.write_text("module { func.func @main() { return } }\n")
    completed = subprocess.run(
        [os.fspath(annc_opt), os.fspath(source),
         f"--atir-config-fusion=config={config} run-builtin-after=false"],
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode != 0
    assert ".abi must be 'mlir_ciface' or 'annc_context_v3'" in completed.stderr
