import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _binary():
    build = Path(os.environ.get("ANNC_BUILD_DIR", ROOT / "build"))
    binary = build / "bin" / "annc-partition"
    if not binary.is_file():
        pytest.skip(f"annc-partition is not built: {binary}")
    return binary


def _run(*args, cwd=None):
    return subprocess.run(
        [os.fspath(_binary()), *args], cwd=cwd, text=True,
        capture_output=True, check=False,
    )


def _task_args(**overrides):
    args = {
        "--framework": "tensorflow",
        "--model": "missing-model",
        "--partition-config": "partition.json",
        "--comm-config": "comm.json",
        "--out-dir": "output",
    }
    args.update(overrides)
    flat = []
    for flag, value in args.items():
        flat += [flag, value]
    return flat


@pytest.mark.parametrize("option", ["--help", "-h", "--version"])
def test_query_modes_run_without_task_inputs(option, tmp_path):
    result = _run(option, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "annc-partition" in result.stdout
    assert "skeleton only" in result.stdout
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_task_rejects_missing_required_arguments(tmp_path):
    result = _run("--model", "m", cwd=tmp_path)
    assert result.returncode == 2
    assert "[annc-partition][argument][E_ARGUMENT]" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_task_rejects_unsupported_capability(tmp_path):
    result = _run(*_task_args(**{"--framework": "onnx"}), cwd=tmp_path)
    assert result.returncode == 3
    assert "[annc-partition][capability][E_CAPABILITY_UNAVAILABLE]" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_task_fails_at_config_read_when_files_missing(tmp_path):
    result = _run(*_task_args(), cwd=tmp_path)
    assert result.returncode == 4
    assert "[annc-partition][config][E_CONFIG_UNREADABLE]" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_task_reaches_adapter_placeholder_with_minimal_configs(tmp_path):
    (tmp_path / "partition.json").write_text(
        json.dumps({"schema_version": 1, "default_partition": "B", "rules": []})
    )
    (tmp_path / "comm.json").write_text(
        json.dumps({"schema_version": 1, "transport": "in_process"})
    )
    result = _run(*_task_args(), cwd=tmp_path)
    assert result.returncode == 5
    assert "[annc-partition][adapter]" in result.stderr
    assert "not implemented" in result.stderr
    # 骨架流程在模型接入占位处终止：不创建输出目录，无其他产物。
    assert not (tmp_path / "output").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "comm.json", "partition.json",
    ]
