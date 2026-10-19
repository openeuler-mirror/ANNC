#!/usr/bin/env python3
"""Verify that ANNC_EP_FUSION_*=1 enables exactly one fused pattern.

The context fixture is reused as the runtime-compile pipeline: it reports a
site whose pattern is selected by ANNC_EP_FIXTURE_PATTERN and compiles a
kernel that appends "compute" to ANNC_EP_FIXTURE_MARKER when it runs. For each
of the four fused patterns we run the same MatMul model twice:

  * switch unset  -> the EP drops the site (default off); the CPU EP runs
  * switch = 1    -> the EP claims the site; the marker records "compute"

The numeric output must match the CPU EP in both cases.
"""

import argparse
import ctypes
import os
from pathlib import Path
import shlex
import sys
import tempfile

import numpy as np
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ep_context_fixture as fixture

PATTERNS = {
    "tensordot_matmul": "ANNC_EP_FUSION_TENSORDOT_MATMUL",
    "layer_norm": "ANNC_EP_FUSION_LAYER_NORM",
    "masked_mul": "ANNC_EP_FUSION_MASKED_MUL",
    "batched_matmul_add": "ANNC_EP_FUSION_BATCHED_MATMUL_ADD",
}


def make_pipeline(work):
    """Point the EP at the fixture with the test interpreter, not PATH."""
    fixture_path = Path(fixture.__file__).resolve()
    wrapper = work / "annc-pipeline-fixture"
    wrapper.write_text(
        "#!/bin/sh\nexec "
        + shlex.quote(sys.executable)
        + " "
        + shlex.quote(str(fixture_path))
        + ' "$@"\n'
    )
    wrapper.chmod(0o755)
    return wrapper


def clear_toggles():
    for name in PATTERNS.values():
        os.environ.pop(name, None)


def run_case(work, pipeline, pattern, disabled):
    case = work / pattern / ("disabled" if disabled else "enabled")
    case.mkdir(parents=True, exist_ok=True)
    cache = case / "cache"
    marker = case / "compute.log"
    model = case / "matmul.onnx"
    fixture.make_model(model, 2)

    clear_toggles()
    os.environ.update(
        {
            "ANNC_EP_PIPELINE": str(pipeline),
            "ANNC_EP_CACHE_DIR": str(cache),
            "ANNC_EP_FIXTURE_PATTERN": pattern,
            "ANNC_EP_FIXTURE_MARKER": str(marker),
            "ANNC_EP_DEBUG": "1",
        }
    )
    if not disabled:
        os.environ[PATTERNS[pattern]] = "1"

    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 11
    expected = fixture.session(model, ["CPUExecutionProvider"]).run(None, {"x": x})[0]
    actual = fixture.session(
        model, ["AnncExecutionProvider", "CPUExecutionProvider"]
    ).run(None, {"x": x})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-7)

    computed = marker.exists() and "compute" in marker.read_text()
    assert computed != disabled, (
        f"{pattern}: disabled={disabled}, marker={computed}; "
        "the EP did not honor the fusion switch"
    )


def cache_key_case(work, pipeline):
    """The switches are part of the EP cache key: same model, same cache dir,
    enabled then disabled must produce two entries. If the switch were missing
    from the fingerprint the second session would hit the first entry."""
    case = work / "cache-key"
    case.mkdir(parents=True, exist_ok=True)
    cache = case / "cache"
    marker = case / "compute.log"
    model = case / "matmul.onnx"
    fixture.make_model(model, 2)

    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 11
    for enabled in (True, False):
        clear_toggles()
        os.environ.update(
            {
                "ANNC_EP_PIPELINE": str(pipeline),
                "ANNC_EP_CACHE_DIR": str(cache),
                "ANNC_EP_FIXTURE_PATTERN": "masked_mul",
                "ANNC_EP_FIXTURE_MARKER": str(marker),
                "ANNC_EP_DEBUG": "1",
            }
        )
        os.environ["ANNC_EP_FUSION_MASKED_MUL"] = "1" if enabled else "0"
        actual = fixture.session(
            model, ["AnncExecutionProvider", "CPUExecutionProvider"]
        ).run(None, {"x": x})[0]
        expected = fixture.session(model, ["CPUExecutionProvider"]).run(
            None, {"x": x}
        )[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-7)

    entries = list(cache.glob("v1/*/*/DONE"))
    assert len(entries) == 2, (
        f"expected two cache entries (switch in fingerprint), got {entries}"
    )
    print("Cache key: fusion switch changes the EP signature")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-library", required=True)
    parser.add_argument("--work-dir")
    args = parser.parse_args()

    work = Path(args.work_dir or tempfile.mkdtemp(prefix="annc-ep-fusion-"))
    work.mkdir(parents=True, exist_ok=True)

    runtime = Path(rtpy.__file__).parent / "libonnxruntime.so.1.26.0"
    ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
    # ORT resolves the plugin path relative to its own runtime directory, so
    # make a relative --ep-library absolute before handing it over.
    ep_library = str(Path(args.ep_library).resolve())
    rtpy.register_execution_provider_library(
        "AnncExecutionProvider", ep_library
    )

    pipeline = make_pipeline(work)
    for pattern in PATTERNS:
        run_case(work, pipeline, pattern, disabled=False)
        run_case(work, pipeline, pattern, disabled=True)
        print(f"{pattern}: enabled claims, disabled falls back to CPU")

    cache_key_case(work, pipeline)
    clear_toggles()
    print("Fusion toggles: all four patterns default off and honor ANNC_EP_FUSION_*=1")


if __name__ == "__main__":
    main()
