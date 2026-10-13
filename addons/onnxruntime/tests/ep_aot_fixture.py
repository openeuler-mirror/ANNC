#!/usr/bin/env python3
"""Exercise the ORT EP contract with a small AOT-compatible MatMul fixture.

The pipeline mode substitutes only ANNC's current JIT-only MatMul codegen.
Graph export, site validation, cache publication, ORT fusion and Compute use
the real plugin.
"""

import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy


KERNEL = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
} MemRef3D;
typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} MemRef2D;

int32_t _mlir_ciface_fixture_matmul(void *lhs, void *rhs, void *result) {
  MemRef3D *x = lhs;
  MemRef2D *w = rhs;
  MemRef3D *y = result;
  for (int64_t b = 0; b < x->sizes[0]; ++b)
    for (int64_t m = 0; m < x->sizes[1]; ++m)
      for (int64_t n = 0; n < w->sizes[1]; ++n) {
        float sum = 0;
        for (int64_t k = 0; k < x->sizes[2]; ++k)
          sum += x->aligned[x->offset + b*x->strides[0] + m*x->strides[1] + k*x->strides[2]] *
                 w->aligned[w->offset + k*w->strides[0] + n*w->strides[1]];
        y->aligned[y->offset + b*y->strides[0] + m*y->strides[1] + n*y->strides[2]] = sum;
      }
  const char *marker = getenv("ANNC_EP_FIXTURE_MARKER");
  if (marker) {
    FILE *out = fopen(marker, "a");
    if (out) { fputs("compute\n", out); fclose(out); }
  }
  return 0;
}
"""


def run_pipeline(args):
    if args.toolchain_info:
        print("annc-ep-aot-fixture-v1")
        return
    if os.environ.get("ANNC_EP_FIXTURE_MODE") == "pipeline-failure":
        with open(os.environ["ANNC_EP_FIXTURE_COMPILE_LOG"], "a") as log:
            log.write("compile\n")
        raise SystemExit(2)
    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    nodes = list(model.graph.node)
    if len(nodes) != 1 or nodes[0].op_type != "MatMul" or nodes[0].name != "n0":
        raise ValueError("fixture requires exactly one exported MatMul named n0")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    kernel_name = "fixture_matmul"
    columns = model.graph.initializer[0].dims[1]
    if os.environ.get("ANNC_EP_FIXTURE_MODE") == "concurrent":
        kernel_name += f"_{columns}"
        # Keep both independent compiler calls in flight long enough for the
        # test's two session threads to overlap their cache work.
        time.sleep(0.25)
    (output / "kernel.c").write_text(KERNEL.replace("fixture_matmul", kernel_name))
    if os.environ.get("ANNC_EP_FIXTURE_MODE") not in {"missing-library", "zero-site"}:
        subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-o",
                        str(output / "libannc_kernels.so"), str(output / "kernel.c")],
                       check=True)
    site = {
        "pattern": "matmul", "execution_mode": "aot", "abi": "mlir_ciface",
        "kernel_name": kernel_name, "member_node_names": ["n0"],
        "kernel_arg_order": [0, 1, 2],
        "args": [
            {"tf_name": nodes[0].input[0], "role": "dynamic", "rank": 3,
             "shape": [2, 3, 4]},
            {"tf_name": nodes[0].input[1], "role": "fixed", "rank": 2,
             "shape": [4, columns]},
        ],
        "outputs": [{"tf_name": nodes[0].output[0], "role": "output",
                     "rank": 3, "shape": [2, 3, columns]}],
    }
    if os.environ.get("ANNC_EP_FIXTURE_MODE") == "missing-symbol":
        site["kernel_name"] = "absent_matmul"
    if os.environ.get("ANNC_EP_FIXTURE_MODE") == "malformed-report":
        # Valid JSON, but a wrong field type. The EP must reject this cache
        # entry and fall back to CPU instead of terminating the host process.
        site["member_node_names"] = "n0"
    if os.environ.get("ANNC_EP_FIXTURE_MODE") == "zero-site":
        sites = []
    elif os.environ.get("ANNC_EP_FIXTURE_MODE") == "malformed-report":
        sites = ["not-an-object"]
    else:
        sites = [site]
    (output / "sites.json").write_text(json.dumps({
        "schema": "annc-ep-site-report/1", "contract_version": "1.0.0",
        "sites": sites,
    }))
    with open(os.environ["ANNC_EP_FIXTURE_COMPILE_LOG"], "a") as log:
        log.write("compile\n")


def session(model, providers):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model), options, providers=providers)


def run_test(args):
    work = Path(args.work_dir or tempfile.mkdtemp(prefix="annc-ep-aot-"))
    work.mkdir(parents=True, exist_ok=True)
    model = work / "matmul.onnx"
    weight = np.arange(20, dtype=np.float32).reshape(4, 5) / 7
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="matmul")],
        "fixture", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 5])],
        [numpy_helper.from_array(weight, "w")])
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    proto.ir_version = 7
    onnx.checker.check_model(proto)
    onnx.save(proto, model)

    runtime = Path(rtpy.__file__).parent / "libonnxruntime.so.1.26.0"
    ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
    rtpy.register_execution_provider_library("AnncExecutionProvider", args.ep_library)
    os.environ.update({
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "ANNC_EP_PIPELINE": str(Path(__file__).resolve()),
        "ANNC_EP_CACHE_DIR": str(work / "cache"),
        "ANNC_EP_FIXTURE_MARKER": str(work / "compute.log"),
        "ANNC_EP_FIXTURE_COMPILE_LOG": str(work / "compile.log"),
        "ANNC_EP_FIXTURE_MODE": args.mode,
        "ANNC_EP_DEBUG": "1",
    })
    if args.mode == "missing-pipeline":
        os.environ["ANNC_EP_PIPELINE"] = str(work / "missing-pipeline")
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 11
    expected = session(model, ["CPUExecutionProvider"]).run(None, {"x": x})[0]
    if args.mode in {"missing-pipeline", "pipeline-failure", "zero-site"}:
        session_count = 2 if args.mode == "pipeline-failure" else 1
        for _ in range(session_count):
            actual = session(model, ["AnncExecutionProvider", "CPUExecutionProvider"]).run(
                None, {"x": x})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-7)
        entries = list((work / "cache").glob("v1/*/*/DONE"))
        assert not list((work / "cache").glob("v1/*/*.work"))
        assert not (work / "compute.log").exists()
        if args.mode == "zero-site":
            assert len(entries) == 1, entries
            assert not (entries[0].parent / "libannc_kernels.so").exists()
            assert (work / "compile.log").read_text().splitlines() == ["compile"]
            print("Zero sites: published cache without kernel; CPU fallback")
        else:
            assert not entries, entries
            if args.mode == "pipeline-failure":
                assert (work / "compile.log").read_text().splitlines() == ["compile", "compile"]
            print(f"{args.mode}: CPU fallback; no cache entry")
        return
    if args.mode == "concurrent":
        other = work / "other.onnx"
        other_proto = onnx.load(model)
        other_proto.graph.initializer[0].CopyFrom(
            numpy_helper.from_array(np.arange(24, dtype=np.float32).reshape(4, 6) / 7, "w"))
        other_proto.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 6
        onnx.save(other_proto, other)
        expected_other = session(other, ["CPUExecutionProvider"]).run(
            None, {"x": x})[0]
        start = threading.Barrier(2)

        def run_one(path, input_name, reference):
            start.wait(timeout=15)
            for _ in range(2):
                actual = session(path, ["AnncExecutionProvider", "CPUExecutionProvider"]).run(
                    None, {input_name: x})[0]
                np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-7)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_one, model, "x", expected),
                       pool.submit(run_one, other, "x", expected_other)]
            for future in futures:
                future.result()
        entries = list((work / "cache").glob("v1/*/*/DONE"))
        assert len(entries) == 2, entries
        names = {json.loads((entry.parent / "sites.json").read_text())["sites"][0]["kernel_name"]
                 for entry in entries}
        assert names == {"fixture_matmul_5", "fixture_matmul_6"}, names
        assert (work / "compile.log").read_text().splitlines() == ["compile"] * 2
        assert (work / "compute.log").read_text().splitlines() == ["compute"] * 4
        assert not list((work / "cache").glob("v1/*/*.work"))
        print("Concurrent different models: overlapping compilers, isolated caches, warm hits")
        return
    if args.mode == "cache-config":
        first = session(model, ["AnncExecutionProvider", "CPUExecutionProvider"])
        actual = first.run(None, {"x": x})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-7)
        os.environ["ANNC_GEMM_CONFIG"] = "alternative"
        second = session(model, ["AnncExecutionProvider", "CPUExecutionProvider"])
        actual = second.run(None, {"x": x})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-7)
        entries = list((work / "cache").glob("v1/*/*/DONE"))
        assert len(entries) == 2, entries
        assert (work / "compile.log").read_text().splitlines() == ["compile", "compile"]
        assert (work / "compute.log").read_text().splitlines() == ["compute", "compute"]
        print("Different GEMM config: separate cache entries and kernels")
        return
    for attempt in range(2):
        actual = session(model, ["AnncExecutionProvider", "CPUExecutionProvider"]).run(
            None, {"x": x})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-7,
                                   equal_nan=True)
        print(f"session {attempt + 1}: numeric match")
    entries = list((work / "cache").glob("v1/*/*/DONE"))
    assert not list((work / "cache").glob("v1/*/*.work")), "work directories were not cleaned"
    if args.mode == "missing-library":
        assert not entries, "incomplete cache must not be published"
        assert not (work / "compute.log").exists()
        print("Missing kernel library: CPU fallback; no DONE marker")
        return
    if args.mode == "malformed-report":
        assert not entries, "malformed report must not be published"
        assert not (work / "compute.log").exists()
        assert (work / "compile.log").read_text().splitlines() == ["compile", "compile"]
        print("Malformed report: CPU fallback; no DONE marker")
        return
    assert len(entries) == 1, entries
    assert (work / "compile.log").read_text().splitlines() == ["compile"]
    library = entries[0].parent / "libannc_kernels.so"
    assert library.is_file()
    if args.mode == "missing-symbol":
        assert not (work / "compute.log").exists()
        print("Missing kernel symbol: CPU fallback; no EP Compute")
        return
    assert (work / "compute.log").read_text().splitlines() == ["compute", "compute"]
    library.unlink()
    actual = session(model, ["AnncExecutionProvider", "CPUExecutionProvider"]).run(
        None, {"x": x})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-7,
                               equal_nan=True)
    assert (work / "compile.log").read_text().splitlines() == ["compile", "compile"]
    assert (work / "compute.log").read_text().splitlines() == ["compute"] * 3
    assert library.is_file()
    print("EP Compute called three times; cold, warm, and damaged-cache rebuild")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolchain-info", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--out-dir")
    parser.add_argument("--ep-library")
    parser.add_argument("--work-dir")
    parser.add_argument("--mode", choices=["normal", "missing-symbol",
                                            "missing-library", "malformed-report", "concurrent",
                                            "cache-config", "missing-pipeline",
                                            "pipeline-failure", "zero-site"],
                        default="normal")
    args = parser.parse_args()
    if args.toolchain_info or args.model:
        run_pipeline(args)
    else:
        run_test(args)


if __name__ == "__main__":
    main()
