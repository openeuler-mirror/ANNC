import os
from pathlib import Path
import subprocess

import pytest


def test_llvm_compat_patch_applies():
    repo = Path(__file__).resolve().parents[3]
    source = repo / "third-party" / "onnx-mlir"
    if not all(
        (source / "third_party" / name / "CMakeLists.txt").is_file()
        for name in ("onnx", "stablehlo")
    ):
        pytest.skip("initialize third-party/onnx-mlir recursively to check the patch")

    patch = repo / "patches" / "onnx-mlir" / "llvm-20.1.patch"
    subprocess.run(
        ["git", "apply", "--check", str(patch)],
        cwd=source,
        env={**os.environ, "GIT_CEILING_DIRECTORIES": str(source.parent)},
        check=True,
    )
