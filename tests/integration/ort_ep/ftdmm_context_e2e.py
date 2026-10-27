#!/usr/bin/env python3
"""End-to-end: annc-onnx-pipeline -> context-ABI AOT kernel -> ORT EP.

The fixture model is the 16-node `tensordot_matmul` region extracted from the
real profile1 OrtGraph export (the same model ftdmm_aot_e2e.py uses). Unlike
that test, the kernel here is built from the source tree's execution-v2 FTDMM
(fused function takes the execution context, allocates its output through the
callbacks), so the site's report carries abi=annc_execution_v2 and the EP
executes it through the ONNX execution context -- engine, pattern, kernel
library and EP all from this tree, no external install.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

MODEL = Path(__file__).parent / "fixtures" / "ftdmm_aot_e2e.onnx"
INPUT = "t558_0"


def make_wide_model(path, n=16):
    """The real fixture's weight is [8, 1]; rebuild it with [8, n] so the
    pipeline->kernel integration also covers a wide (vectorised) output.
    Everything else in the 16-node region stays identical."""
    model = onnx.load(str(MODEL))
    graph = model.graph
    for index, initializer in enumerate(graph.initializer):
        if initializer.name == "w_136":
            values = (np.arange(8 * n, dtype=np.float32).reshape(8, n) - 64) / 37
            graph.initializer[index].CopyFrom(
                numpy_helper.from_array(values, "w_136"))
        elif initializer.name == "w_154":
            graph.initializer[index].CopyFrom(
                numpy_helper.from_array(np.array([n], dtype=np.int32), "w_154"))
    graph.output[0].type.tensor_type.shape.dim[2].dim_value = n
    onnx.checker.check_model(model)
    onnx.save(model, path)


def run_case(model_path, input_name, x, cache_dir, expected_weight_shape):
    os.environ["ANNC_EP_CACHE_DIR"] = str(cache_dir)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    reference = ort.InferenceSession(
        str(model_path), options,
        providers=["CPUExecutionProvider"]).run(None, {input_name: x})[0]

    for attempt in range(2):
        session = ort.InferenceSession(
            str(model_path), options,
            providers=["AnncExecutionProvider", "CPUExecutionProvider"])
        actual = session.run(None, {input_name: x})[0]
        np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-7,
                                   equal_nan=True)
        print(f"{Path(model_path).stem} session {attempt + 1}: numeric match")

    entries = [p for p in Path(cache_dir).glob("v1/*/*")
               if p.is_dir() and (p / "DONE").is_file()]
    assert len(entries) == 1, entries
    report = json.loads((entries[0] / "sites.json").read_text())
    sites = report["sites"]
    assert len(sites) == 1, sites
    site = sites[0]
    assert site["pattern"] == "tensordot_matmul", site
    assert site["abi"] == "annc_execution_v2", site
    assert site["kernel_arg_order"] == [], site
    assert len(site["member_node_names"]) == 16, site
    assert site["args"][-1]["shape"] == expected_weight_shape, site
    assert (entries[0] / "libannc_kernels.so").is_file()
    return actual


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

    work = Path(args.work_dir or tempfile.mkdtemp(prefix="annc-ftdmm-context-"))
    (work / "cache").mkdir(parents=True, exist_ok=True)
    merged = merge_kernel_libs(
        work, [args.kernel_lib_dir, args.llvm_lib_dir, *args.extra_lib_dir])
    for required in ("libANNCBuiltinKernels.a", "libLLVMSupport.a",
                     "libANNCGemmSupport.a"):
        assert (merged / required).exists(), (
            f"{required} missing from {merged}; build annc-onnx-pipeline, "
            "and ANNCGemmSupport")

    runtime = Path(rtpy.__file__).parent / "libonnxruntime.so.1.26.0"
    ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
    rtpy.register_execution_provider_library("AnncExecutionProvider",
                                             args.ep_library)
    env = dict(os.environ,
               ANNC_EP_PIPELINE=args.pipeline,
               ANNC_EP_CACHE_DIR=str(work / "cache"),
               ANNC_EP_FUSION_TENSORDOT_MATMUL="1",
               ANNC_KERNEL_LIB_PATH=str(merged))
    gcc_runtime = Path("/usr/lib/gcc/aarch64-openEuler-linux/12")
    if gcc_runtime.is_dir() and not env.get("LIBRARY_PATH"):
        env["LIBRARY_PATH"] = str(gcc_runtime)
    os.environ.update({k: v for k, v in env.items()
                       if k.startswith("ANNC_") or k == "LIBRARY_PATH"})

    x = np.arange(16, dtype=np.float32).reshape(1, 2, 8) / 7
    run_case(MODEL, INPUT, x, work / "cache", [8, 1])

    # The real fixture has a single output column; the kernel's wide path
    # (16-wide blocking) only ran in unit tests.  Rebuild the same region with
    # a [8, 16] weight and run it through the same pipeline/EP.
    wide_work = work / "wide"
    wide_work.mkdir(parents=True, exist_ok=True)
    wide_model = wide_work / "ftdmm_wide.onnx"
    make_wide_model(wide_model, 16)
    x_wide = np.arange(24, dtype=np.float32).reshape(1, 3, 8) / 7
    actual = run_case(wide_model, INPUT, x_wide, wide_work / "cache", [8, 16])
    assert actual.shape == (1, 3, 16), actual.shape

    print("real pipeline produced context-ABI FTDMM sites for [8,1] and "
          "[8,16] weights; both kernels executed through the ONNX execution "
          "context (cold compile + warm cache)")


if __name__ == "__main__":
    main()
