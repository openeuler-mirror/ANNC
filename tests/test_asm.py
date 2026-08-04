from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD_PYTHON = ROOT / "build" / "python_packages"

if not BUILD_PYTHON.exists():
    pytest.skip(
        "annc Python package is not built; run ninja -C build ANNCPythonModules",
        allow_module_level=True,
    )

sys.path.insert(0, str(BUILD_PYTHON))

annc_builder = pytest.importorskip("annc.builder")
mlir = annc_builder.mlir
from annc.helper import kp
from annc.ops import matmul
from annc.passmanager import PassManager
from annc.types import ElemType


@kp.jit
def fused_matmul(inputs, **kwargs):
    lhs = inputs[0]
    rhs = inputs[1]
    cm = inputs[2]
    matmul(ElemType.FP32(), [1024, 1024], lhs, rhs, cm, False)


def build_fused_matmul_module() -> mlir.Module:
    return fused_matmul(
        name="demo",
        outputs=["out"],
        inputs=[
            np.random.randn(1024, 1024).astype(np.float32),
            np.random.randn(1024, 1024).astype(np.float32),
            np.zeros((1024, 1024)).astype(np.float32),
            np.random.randn(1024).astype(np.float32),
        ],
    )


def run_pipeline(label: str, passes: list[str]) -> None:
    module = build_fused_matmul_module()

    print(f"=== {label}: before ===")
    module.operation.print(large_elements_limit=16)

    module.context.__enter__()
    try:
        pm = PassManager.parse("builtin.module()")
        for pass_name in passes:
            pm.add(pass_name)
        pm.run(module.operation)
    finally:
        module.context.__exit__(None, None, None)

    print(f"=== {label}: after ===")
    module.operation.print(large_elements_limit=16)


PIPELINES = {
    "select-lowering-strategy": [
        "atir-select-lowering-strategy",
    ],
    "linalg-cache-parallel": [
        "atir-select-lowering-strategy",
        "convert-atir-to-linalg",
        "cache-parallel",
    ],
    "aarch64-matmul-tiling": [
        "atir-select-lowering-strategy",
        "convert-atir-to-linalg",
        "annc-one-shot-bufferize",
        "cache-parallel",
        "cache-reduction",
        "vector-common-parallel",
        "vector-reduction",
    ],
    "aarch64-matmul-pack-affine": [
        "atir-select-lowering-strategy",
        "convert-atir-to-linalg",
        "annc-one-shot-bufferize",
        "matmul-pack-affine",
    ],
    "fast-codegen-affine": [
        "atir-fast-codegen",
        "convert-atir-to-affine",
    ],
}


def main() -> None:
    run_pipeline(
        "aarch64-matmul-pack-affine",
        PIPELINES["aarch64-matmul-pack-affine"],
    )


def test_annc_builder_imports() -> None:
    assert mlir is not None


if __name__ == "__main__":
    main()
