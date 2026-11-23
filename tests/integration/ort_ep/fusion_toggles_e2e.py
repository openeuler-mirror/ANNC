#!/usr/bin/env python3
"""Fusion switches: real pattern contracts, host execution and cache isolation."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnxruntime as ort
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests/e2e/kernel_context"))
from batched_matmul_add_context_e2e import build_model as bmm, merge_kernel_libs
from layer_norm_context_e2e import build_model as layer_norm
from masked_mul_context_e2e import build_product_model
from ftdmm_context_e2e import make_wide_model

PATTERNS = {
    "tensordot_matmul": "ANNC_EP_FUSION_TENSORDOT_MATMUL",
    "layer_norm": "ANNC_EP_FUSION_LAYER_NORM",
    "masked_mul": "ANNC_EP_FUSION_MASKED_MUL",
    "batched_matmul_add": "ANNC_EP_FUSION_BATCHED_MATMUL_ADD",
}


def session(model, providers, profile):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_profiling = True
    options.profile_file_prefix = str(profile)
    return ort.InferenceSession(str(model), options, providers=providers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("ep-library", "pipeline", "kernel-lib-dir"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--llvm-lib-dir", type=Path)
    parser.add_argument("--extra-lib-dir", type=Path, action="append", default=[])
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="annc-fusion-toggles-"))
    work.mkdir(parents=True, exist_ok=True)
    dirs = [args.kernel_lib_dir, *args.extra_lib_dir]
    if args.llvm_lib_dir:
        dirs.append(args.llvm_lib_dir)
    os.environ.update(
        ANNC_EP_PIPELINE=str(args.pipeline.resolve()),
        ANNC_KERNEL_LIB_PATH=str(merge_kernel_libs(work, dirs)),
    )
    runtime = next(Path(rtpy.__file__).parent.glob("libonnxruntime.so.*"))
    ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
    rtpy.register_execution_provider_library(
        "AnncExecutionProvider", str(args.ep_library.resolve())
    )
    rng = np.random.default_rng(71)
    cases = {
        "tensordot_matmul": (lambda p: make_wide_model(p, 1), {"t558_0": (2, 3, 8)}),
        "layer_norm": (lambda p: layer_norm(p, 4, 16), {"x": (4, 16)}),
        "masked_mul": (build_product_model, {"a": (2, 3, 4), "b": (2, 3, 4)}),
        "batched_matmul_add": (lambda p: bmm(p, batch=2), {"x": (2, 5, 12)}),
    }
    for pattern, (builder, shapes) in cases.items():
        case = work / pattern
        case.mkdir()
        model = case / "model.onnx"
        builder(model)
        feed = {
            name: rng.standard_normal(shape).astype(np.float32)
            for name, shape in shapes.items()
        }
        if pattern == "masked_mul":
            feed["ids"] = rng.integers(0, 3, (2, 3), dtype=np.int64)
        cpu = session(model, ["CPUExecutionProvider"], case / "cpu")
        expected = cpu.run(None, feed)
        cpu.end_profiling()
        cache = case / "cache"
        os.environ["ANNC_EP_CACHE_DIR"] = str(cache)
        for enabled in (True, False):
            for switch in PATTERNS.values():
                os.environ.pop(switch, None)
            if enabled:
                os.environ[PATTERNS[pattern]] = "1"
            host = session(
                model,
                ["AnncExecutionProvider", "CPUExecutionProvider"],
                case / str(enabled),
            )
            actual = host.run(None, feed)
            assert len(actual) == len(expected)
            for value, target in zip(actual, expected):
                np.testing.assert_allclose(value, target, rtol=1e-3, atol=1e-6)
            events = json.loads(Path(host.end_profiling()).read_text())
            fused = any(
                e.get("args", {}).get("provider") == "AnncExecutionProvider"
                for e in events
            )
            assert fused == enabled, (pattern, enabled, events)
        assert len(list(cache.glob("context-v3/*/*/DONE"))) == 2
        print(
            f"{pattern}: enabled executes, default disabled uses CPU; cache keys isolated",
            flush=True,
        )


if __name__ == "__main__":
    main()
