#!/usr/bin/env python3
"""Manual smoke probe for the ANNC Python binding surface.

This is intentionally *not* collected by pytest (it is a diagnostic script,
not a numerical test): the original root-level ``test_matmul.py`` had no
assertions, imported the source tree instead of the built package, and could
never run.  It now checks that the built bindings import and that the tensor
type used by the matmul builder API can be constructed.

Run from the repository root after building the Python bindings::

    python3 tests/e2e/python/matmul_api_smoke.py

The script looks for ``build/python_packages`` and ``install/python_packages``;
set ``ANNC_PYTHON_PACKAGES`` to override the location.
"""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _candidate_packages():
    configured = os.environ.get("ANNC_PYTHON_PACKAGES")
    if configured:
        yield Path(configured)
    yield ROOT / "build" / "python_packages"
    yield ROOT / "install" / "python_packages"


def _bootstrap():
    for candidate in _candidate_packages():
        if (candidate / "annc" / "__init__.py").is_file():
            sys.path.insert(0, os.fspath(candidate))
            return candidate
    raise SystemExit(
        "ANNC Python bindings not found. Build the python target first or set "
        "ANNC_PYTHON_PACKAGES to the directory containing the annc package."
    )


def check_imports():
    import annc
    from annc.ops import Atir, TensorType, Value
    from annc.types import ElemType

    assert annc.ir is not None
    assert Atir is not None
    assert TensorType is not None
    assert Value is not None
    assert ElemType.FP32 is not None
    return "annc package imports"


def check_tensor_type():
    from annc import ir as mlir
    from annc.ops import TensorType
    from annc.types import ElemType

    with mlir.Context() as ctx:
        from annc.dialects import atir

        atir.register_dialect()
        ctx.load_all_available_dialects()
        tensor = TensorType.get([2, 3], ElemType.FP32())
        assert tensor.get_shape() == [2, 3]
        assert "f32" in tensor.get_element_type()
    return "TensorType.get([2, 3], FP32)"


def main():
    packages = _bootstrap()
    checks = (check_imports, check_tensor_type)
    failed = 0
    for check in checks:
        try:
            detail = check()
            print(f"PASS: {check.__name__}: {detail}")
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            failed += 1
            print(f"FAIL: {check.__name__}: {exc}")
    print(f"python packages: {packages}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
