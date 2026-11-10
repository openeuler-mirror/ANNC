"""TF1.15 acceptance: collection needs pytest only, execution uses the SDK Python.

Covers MatMul compatibility and tensordot fusion using existing ANNC artifacts.
Set ANNC_REQUIRE_E2E=1 to fail on missing prerequisites instead of skipping.
"""

import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).with_name("tf_acceptance_runner.py")

# Explicit names keep the collected inventory reviewable without importing TF.
FUSION_CASES = ["TF-F01-plain", "TF-F01-dynamic"]
FUSION_CASES += [
    "TF-F11-" + name
    for name in (
        "static",
        "dynamic-b",
        "dynamic-s",
        "dynamic-bs",
        "unfolded",
        "partial",
        "identity",
    )
]


def require_environment(condition, reason):
    if not condition:
        if os.environ.get("ANNC_REQUIRE_E2E") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


@pytest.fixture(scope="module")
def sdk():
    build = Path(os.environ.get("ANNC_BUILD_DIR", ROOT / "build")).resolve()
    cache = build / "CMakeCache.txt"
    require_environment(
        cache.is_file(), f"Missing existing build configuration: {cache}"
    )
    values = {}
    for line in cache.read_text().splitlines():
        if ":" in line and "=" in line and not line.startswith(("//", "#")):
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    python = values.get("ANNC_TENSORFLOW_PYTHON_EXECUTABLE")
    require_environment(
        python and Path(python).is_file(), "Configure the target TF Python first"
    )
    for relative in (
        "bin/annc-tf-pipeline",
        "bin/annc-tf2atir",
        "bin/annc-opt",
        "bin/annc-converter",
        "bin/annc-fusion-metadata",
        "lib/tensorflow/1.15/libannc_optimizer.so",
        "lib/tensorflow/1.15/libannc_fused_op.so",
    ):
        require_environment(
            (build / relative).is_file(), f"Missing existing artifact: {relative}"
        )
    return build, python


def run_case(sdk, tmp_path, case):
    build, python = sdk
    tmp_path.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        str(RUNNER),
        "--build-dir",
        str(build),
        "--case",
        case,
        "--work-dir",
        str(tmp_path),
    ]
    with (tmp_path / "runner.log").open("w") as log:
        result = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=240
        )
    assert result.returncode == 0, (
        f"{case}: see {tmp_path / 'runner.log'}\n"
        + (tmp_path / "runner.log").read_text()[-5000:]
    )
    evidence = json.loads((tmp_path / "result.json").read_text())
    assert evidence["case"] == case and evidence["status"] == "passed"
    return evidence


@pytest.mark.parametrize("case", FUSION_CASES, ids=FUSION_CASES)
def test_fusion(sdk, tmp_path, case):
    run_case(sdk, tmp_path, case)


@pytest.mark.parametrize("case", ["transpose", "wrong-shape", "external-use"])
def test_tensordot_rejects(sdk, tmp_path, case):
    run_case(sdk, tmp_path, "TF-C-reject-" + case)


def test_tensordot_empty_batch(sdk, tmp_path):
    run_case(sdk, tmp_path, "TF-F11-dynamic-empty")


@pytest.mark.parametrize("admission", ["eligible", "mismatch"])
def test_jit_prepack(sdk, tmp_path, admission):
    run_case(sdk, tmp_path, "TF-JIT-prepack-" + admission)
