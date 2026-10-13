#!/usr/bin/env python3
"""End-to-end: annc-onnx-pipeline -> batched_matmul_add context-ABI kernel -> EP.

The fixture is the ONNX rank-3 MatMul + broadcast bias Add the real models'
attention QKV projections carry:

    MatMul(x [1, T, K], W [1, K, N]) -> Add(., bias [1, 1, N])

All three arguments are rank-3 f32, so the EP cannot tell them apart by rank
and type -- the site's arguments must be bound by tensor name.

Everything runs from this source tree: the pipeline produces the site report
and the kernel library, the EP executes it through the ONNX execution context,
and the numbers are checked against the CPU EP.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

SEQ, K, N = 5, 12, 16


def build_model(path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, SEQ, K])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, SEQ, N])
    w = numpy_helper.from_array(
        (np.arange(K * N, dtype=np.float32).reshape(1, K, N) - 96.0) / 37.0, "w")
    b = numpy_helper.from_array(
        np.linspace(-0.2, 0.2, N, dtype=np.float32).reshape(1, 1, N), "b")
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["mm"], name="n_mm"),
        helper.make_node("Add", ["mm", "b"], ["y"], name="n_add"),
    ]
    graph = helper.make_graph(nodes, "bmm_add", [x], [y], initializer=[w, b])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)


def merge_kernel_libs(work, lib_dirs):
    """The annc driver links with a single -L: every archive it needs -- ANNC
    kernels and the LLVM support libraries -- must sit in one directory."""
    merged = work / "kernel-libs"
    merged.mkdir(parents=True, exist_ok=True)
    for lib_dir in lib_dirs:
        if lib_dir is None or not Path(lib_dir).is_dir():
            continue
        for archive in Path(lib_dir).glob("*.a"):
            target = merged / archive.name
            if not target.exists():
                target.symlink_to(archive)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-library", required=True)
    parser.add_argument("--pipeline", required=True,
                        help="annc-onnx-pipeline built from this tree")
    parser.add_argument("--kernel-lib-dir", required=True,
                        help="directory holding libANNCBuiltinKernels.a")
    parser.add_argument("--llvm-lib-dir",
                        help="directory holding the LLVM support archives")
    parser.add_argument("--extra-lib-dir", action="append", default=[],
                        help="further archive directories the driver links "
                             "against (ANNCGemmSupport, GEMM microkernels)")
    parser.add_argument("--work-dir")
    args = parser.parse_args()

    work = Path(args.work_dir or tempfile.mkdtemp(prefix="annc-bmm-add-"))
    (work / "cache").mkdir(parents=True, exist_ok=True)
    merged = merge_kernel_libs(
        work, [args.kernel_lib_dir, args.llvm_lib_dir, *args.extra_lib_dir])
    for required in ("libANNCBuiltinKernels.a", "libLLVMSupport.a",
                     "libANNCGemmSupport.a", "libannc_gemm_microkernels.a"):
        assert (merged / required).exists(), (
            f"{required} missing from {merged}; build annc-onnx-pipeline, "
            "ANNCGemmSupport and make the GEMM microkernel archive available")

    model_path = work / "bmm_add.onnx"
    build_model(model_path)

    runtime = Path(rtpy.__file__).parent / "libonnxruntime.so.1.26.0"
    ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
    rtpy.register_execution_provider_library("AnncExecutionProvider",
                                             args.ep_library)
    env = dict(os.environ,
               ANNC_EP_PIPELINE=args.pipeline,
               ANNC_EP_CACHE_DIR=str(work / "cache"),
               ANNC_KERNEL_LIB_PATH=str(merged))
    gcc_runtime = Path("/usr/lib/gcc/aarch64-openEuler-linux/12")
    if gcc_runtime.is_dir() and not env.get("LIBRARY_PATH"):
        env["LIBRARY_PATH"] = str(gcc_runtime)
    if not env.get("ANNC_CLANG") and Path("/tmp/annc-clang-wrapper").is_file():
        env["ANNC_CLANG"] = "/tmp/annc-clang-wrapper"
    os.environ.update({k: v for k, v in env.items()
                       if k.startswith("ANNC_") or k == "LIBRARY_PATH"})

    rng = np.random.default_rng(11)
    feed = {"x": rng.standard_normal((1, SEQ, K)).astype(np.float32)}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    reference = ort.InferenceSession(
        str(model_path), options,
        providers=["CPUExecutionProvider"]).run(None, feed)[0]

    for attempt in range(2):
        session = ort.InferenceSession(
            str(model_path), options,
            providers=["AnncExecutionProvider", "CPUExecutionProvider"])
        actual = session.run(None, feed)[0]
        np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-7,
                                   equal_nan=True)
        print(f"session {attempt + 1}: numeric match")

    entries = [p for p in (work / "cache").glob("v1/*/*")
               if p.is_dir() and (p / "DONE").is_file()]
    assert len(entries) == 1, entries
    report = json.loads((entries[0] / "sites.json").read_text())
    sites = report["sites"]
    assert len(sites) == 1, sites
    site = sites[0]
    assert site["pattern"] == "batched_matmul_add", site
    assert site["abi"] == "annc_execution_v2", site
    assert site["kernel_arg_order"] == [], site
    # MatMul + Add (the names the EP sees are its own export's).
    assert len(site["member_node_names"]) == 2, site
    assert [arg["rank"] for arg in site["args"]] == [3, 3, 3], site
    assert (entries[0] / "libannc_kernels.so").is_file()
    print("real pipeline produced a batched_matmul_add context-ABI site; the "
          "three rank-3 arguments were bound by name (cold + warm cache)")


if __name__ == "__main__":
    main()
