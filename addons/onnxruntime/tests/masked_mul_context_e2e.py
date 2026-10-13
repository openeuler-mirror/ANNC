#!/usr/bin/env python3
"""End-to-end: annc-onnx-pipeline -> masked_mul context-ABI kernel -> ORT EP.

The fixture is a synthetic ONNX model carrying one mask chain feeding *two*
Muls -- the shape the real models have (one chain, 1..4 Mul consumers):

    ids --Equal(pad)--> Not --Cast--> Unsqueeze --> Tile --+--> Mul(a) -> out_a
                                                           +--> Mul(b) -> out_b

Two things this covers that the FTDMM e2e cannot:

* a site whose kernel arguments cannot be told apart by rank+type (two rank-3
  f32 data operands) -- the EP binds them by tensor name, which only works
  when the report's names translate back to the graph's;
* a site with *two* outputs, whose order in the pattern need not match ORT's
  numbering of the fused node's outputs -- the EP resolves that permutation at
  Compile time and the kernel allocates through it.

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
from onnx import TensorProto, helper
import onnxruntime as ort
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

INPUT_NAMES = ("ids", "a", "b")
OUTPUT_NAMES = ("out_a", "out_b")


def build_model(path):
    ids = helper.make_tensor_value_info("ids", TensorProto.INT64, [2, 3])
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3, 4])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3, 4])
    out_a = helper.make_tensor_value_info("out_a", TensorProto.FLOAT, [2, 3, 4])
    out_b = helper.make_tensor_value_info("out_b", TensorProto.FLOAT, [2, 3, 4])
    pad = helper.make_tensor("pad", TensorProto.INT64, [], [0])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [2])
    repeats = helper.make_tensor("repeats", TensorProto.INT64, [3], [1, 1, 4])
    nodes = [
        helper.make_node("Equal", ["ids", "pad"], ["eq"], name="n_eq"),
        helper.make_node("Not", ["eq"], ["not_eq"], name="n_not"),
        helper.make_node("Cast", ["not_eq"], ["mask2"], name="n_cast",
                         to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["mask2", "axes"], ["mask3"],
                         name="n_unsqueeze"),
        helper.make_node("Tile", ["mask3", "repeats"], ["tiled"], name="n_tile"),
        helper.make_node("Mul", ["a", "tiled"], ["out_a"], name="n_mul_a"),
        helper.make_node("Mul", ["b", "tiled"], ["out_b"], name="n_mul_b"),
    ]
    graph = helper.make_graph(nodes, "masked_mul_chain", [ids, a, b],
                              [out_a, out_b],
                              initializer=[pad, axes, repeats])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    # Match the FTDMM fixture's metadata: ORT 1.26 has no trouble with it.
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


def inputs():
    ids = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int64)
    a = (np.arange(24, dtype=np.float32) - 5.0).reshape(2, 3, 4) / 3.0
    b = (np.arange(24, dtype=np.float32) + 1.0).reshape(2, 3, 4) / 7.0
    return {"ids": ids, "a": a, "b": b}


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

    work = Path(args.work_dir or tempfile.mkdtemp(prefix="annc-masked-mul-"))
    (work / "cache").mkdir(parents=True, exist_ok=True)
    merged = merge_kernel_libs(
        work, [args.kernel_lib_dir, args.llvm_lib_dir, *args.extra_lib_dir])
    for required in ("libANNCBuiltinKernels.a", "libLLVMSupport.a",
                     "libANNCGemmSupport.a", "libannc_gemm_microkernels.a"):
        assert (merged / required).exists(), (
            f"{required} missing from {merged}; build annc-onnx-pipeline, "
            "ANNCGemmSupport and make the GEMM microkernel archive available")

    model_path = work / "masked_mul.onnx"
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

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    feed = inputs()
    reference = ort.InferenceSession(
        str(model_path), options,
        providers=["CPUExecutionProvider"]).run(None, feed)

    for attempt in range(2):
        session = ort.InferenceSession(
            str(model_path), options,
            providers=["AnncExecutionProvider", "CPUExecutionProvider"])
        actual = session.run(None, feed)
        for name, expected, got in zip(OUTPUT_NAMES, reference, actual):
            np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-7,
                                       equal_nan=True)
        print(f"session {attempt + 1}: numeric match ({', '.join(OUTPUT_NAMES)})")

    entries = [p for p in (work / "cache").glob("v1/*/*")
               if p.is_dir() and (p / "DONE").is_file()]
    assert len(entries) == 1, entries
    report = json.loads((entries[0] / "sites.json").read_text())
    sites = report["sites"]
    assert len(sites) == 1, sites
    site = sites[0]
    assert site["pattern"] == "masked_mul", site
    assert site["abi"] == "annc_execution_v2", site
    assert site["kernel_arg_order"] == [], site
    # Equal/Not/Cast/Unsqueeze/Tile + two Muls.
    assert len(site["member_node_names"]) == 7, site
    assert len(site["outputs"]) == 2, site
    assert len(site["args"]) == 3, site
    by_rank = sorted(arg["rank"] for arg in site["args"])
    assert by_rank == [2, 3, 3], site
    assert (entries[0] / "libannc_kernels.so").is_file()
    print("real pipeline produced a two-output masked_mul context-ABI site; "
          "both outputs matched the CPU EP (cold compile + warm cache)")


if __name__ == "__main__":
    main()
