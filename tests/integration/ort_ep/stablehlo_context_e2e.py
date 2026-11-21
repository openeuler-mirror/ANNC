#!/usr/bin/env python3
"""Compare both ONNX frontends on one dynamic BMMAdd and a shared EP cache."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "e2e/kernel_context"))
from batched_matmul_add_context_e2e import build_rank4_model, merge_kernel_libs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--ep-library", type=Path, required=True)
    parser.add_argument("--kernel-lib-dir", type=Path, required=True)
    parser.add_argument("--llvm-lib-dir", type=Path, required=True)
    parser.add_argument("--extra-lib-dir", type=Path, action="append", default=[])
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="annc-stablehlo-ep-"))
    library_dirs = [args.kernel_lib_dir, args.llvm_lib_dir, *args.extra_lib_dir]
    merged = merge_kernel_libs(work, [directory.resolve() for directory in library_dirs])
    cache = work / "cache"
    cache.mkdir()
    pipeline = args.pipeline.resolve()
    os.environ.update(
        ANNC_EP_PIPELINE=str(pipeline),
        ANNC_EP_CACHE_DIR=str(cache.resolve()),
        ANNC_EP_FUSION_BATCHED_MATMUL_ADD="1",
        ANNC_EP_KEEP_TEMPS="1",
        ANNC_KERNEL_LIB_PATH=str(merged.resolve()),
    )

    runtime = next(Path(rtpy.__file__).parent.glob("libonnxruntime.so.*"))
    ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
    rtpy.register_execution_provider_library(
        "AnncExecutionProvider", str(args.ep_library.resolve())
    )

    model = work / "model.onnx"
    build_rank4_model(model, dynamic_m=True)
    graph = onnx.load(model)
    graph.graph.node.insert(
        0, onnx.helper.make_node("Mul", ["x", "scale"], ["scaled"], name="n_scale")
    )
    graph.graph.node[1].input[0] = "scaled"
    graph.graph.initializer.append(
        onnx.numpy_helper.from_array(np.array(0.9995004, dtype=np.float32), "scale")
    )
    onnx.checker.check_model(graph)
    onnx.save(graph, model)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    cpu = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(23)
    feeds = [
        {"x": rng.standard_normal((2, 1, rows, 12), dtype=np.float32)}
        for rows in (3, 7)
    ]
    expected = [cpu.run(None, feed)[0] for feed in feeds]

    def run(frontend, stablehlo):
        if stablehlo:
            os.environ["ANNC_ONNX_USE_STABLEHLO"] = "1"
        else:
            os.environ.pop("ANNC_ONNX_USE_STABLEHLO", None)
        options.enable_profiling = True
        options.profile_file_prefix = str(work / frontend)
        session = ort.InferenceSession(
            str(model), options,
            providers=["AnncExecutionProvider", "CPUExecutionProvider"],
        )
        for feed, reference in zip(feeds, expected):
            np.testing.assert_allclose(
                session.run(None, feed)[0], reference, rtol=1e-3, atol=1e-7
            )
        profile = json.loads(Path(session.end_profiling()).read_text())
        executions = [
            event for event in profile
            if event.get("cat") == "Node"
            and event.get("args", {}).get("provider") == "AnncExecutionProvider"
        ]
        assert len(executions) == len(feeds), (frontend, executions)

    run("atir", False)
    (atir,) = cache.rglob("DONE")
    atir_library = atir.parent / "libannc_kernels.so"
    assert (atir.parent / "sites.json").is_file()
    assert not (atir.parent / "model.fused.mlir").exists()
    atir_mtime = atir_library.stat().st_mtime_ns

    run("stablehlo", True)
    assert atir in set(cache.rglob("DONE"))
    (done,) = set(cache.rglob("DONE")) - {atir}
    compiled = done.parent
    assert (compiled / "libannc_kernels.so").is_file()
    assert not (compiled / "sites.json").exists()
    report = work / "metadata.json"
    subprocess.run(
        [str(pipeline.with_name("annc-fusion-metadata")),
         str(compiled / "model.fused.mlir"), "-o", str(report)],
        check=True,
    )
    (site,) = json.loads(report.read_text())["fusions"]
    assert site["pattern"] == "batched_matmul_add"
    assert site["abi"] == "annc_context_v3"
    assert [arg["rank"] for arg in site["args"]] == [4, 4, 4]
    exported = onnx.load(next(cache.glob("tmp-*/model.onnx")))
    assert len(site["member_node_names"]) == 3
    assert set(site["member_node_names"]) == {node.name for node in exported.graph.node}
    run("atir-warm", False)
    assert set(cache.rglob("DONE")) == {atir, done}
    assert atir_library.stat().st_mtime_ns == atir_mtime
    print("Dynamic BMMAdd: both frontends executed and reused the correct cache")


if __name__ == "__main__":
    main()
