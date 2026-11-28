"""Shared helpers for Context V3 TensorFlow pipeline tests."""

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _require(condition, reason):
    if condition:
        return
    if os.environ.get("ANNC_REQUIRE_E2E") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _run(command, env, log, timeout=300):
    result = subprocess.run(
        [os.fspath(arg) for arg in command],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    log.write_text(result.stdout + "\n" + result.stderr)
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {command}\n"
        f"{result.stdout}\n{result.stderr}\nlog: {log}"
    )
    return result
