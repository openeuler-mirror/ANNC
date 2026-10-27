#!/usr/bin/env python3
"""End-to-end: annc-onnx-pipeline -> masked_mul context-ABI kernels -> ORT EP.

Three fixtures, matching the shapes the real models have (one mask chain, its
Mul consumers, and -- in model1 -- the element-wise path between them):

    ids --Equal(pad)--> Not --Cast--> Unsqueeze --> Tile --+--> Mul(a) -> out_a
                                                           +--> Mul(b) -> out_b

* shared chain (`shared_chain`): one site for the whole subgraph -- the chain and
  both Muls -- whose outputs are both Mul results.  The mask is never
  materialised.
* single Mul (`one_mul`): same, with one Mul.
* Sum/Div between two Muls (`norm`, model1's shape): the two raw results feed a
  Sum (which also takes an outside tensor) and a Div by a constant, and the
  result feeds a third Mul.  Those nodes are members too, so the kernel computes
  the sum, the division and the third result -- one node where the patch needs
  three fused nodes plus the Sum/Div left on the CPU.
* plain product between two Muls (`product`, model1's other shape): the two raw
  results are multiplied, and the product is masked again -- the interior Mul is
  computed in the kernel as well.

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


def build_norm_model(path):
    """model1's shape: Sum -> Div(const) between two masked Muls."""
    ids = helper.make_tensor_value_info("ids", TensorProto.INT64, [2, 3])
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3, 4])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3, 4])
    e = helper.make_tensor_value_info("e", TensorProto.FLOAT, [2, 3, 4])
    outs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [2, 3, 4])
            for n in OUTPUT_NAMES]
    pad = helper.make_tensor("pad", TensorProto.INT64, [], [0])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [2])
    two = helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])
    nodes = [
        helper.make_node("Equal", ["ids", "pad"], ["eq"], name="n_eq"),
        helper.make_node("Not", ["eq"], ["not_eq"], name="n_not"),
        helper.make_node("Cast", ["not_eq"], ["mask"], name="n_cast",
                         to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["mask", "axes"], ["mask3"],
                         name="n_unsqueeze"),
        helper.make_node("Mul", ["a", "mask3"], ["out_a"], name="n_mul_a"),
        helper.make_node("Mul", ["b", "mask3"], ["out_b"], name="n_mul_b"),
        helper.make_node("Sum", ["out_a", "out_b", "e"], ["sum"],
                         name="n_sum"),
        helper.make_node("Div", ["sum", "two"], ["scaled"], name="n_div"),
        helper.make_node("Mul", ["scaled", "mask3"], ["out_c"],
                         name="n_mul_c"),
    ]
    graph = helper.make_graph(nodes, "masked_mul_norm", [ids, a, b, e],
                              outs + [helper.make_tensor_value_info(
                                  "out_c", TensorProto.FLOAT, [2, 3, 4])],
                              initializer=[pad, axes, two])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)


def build_product_model(path):
    """model1's NotEqual chain: a plain element-wise product between two
    masked Muls (out_a * out_b), whose result is masked again."""
    ids = helper.make_tensor_value_info("ids", TensorProto.INT64, [2, 3])
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3, 4])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3, 4])
    outs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [2, 3, 4])
            for n in OUTPUT_NAMES]
    pad = helper.make_tensor("pad", TensorProto.INT64, [], [0])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [2])
    nodes = [
        helper.make_node("Equal", ["ids", "pad"], ["eq"], name="n_eq"),
        helper.make_node("Not", ["eq"], ["not_eq"], name="n_not"),
        helper.make_node("Cast", ["not_eq"], ["mask"], name="n_cast",
                         to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["mask", "axes"], ["mask3"],
                         name="n_unsqueeze"),
        helper.make_node("Mul", ["a", "mask3"], ["out_a"], name="n_mul_a"),
        helper.make_node("Mul", ["b", "mask3"], ["out_b"], name="n_mul_b"),
        helper.make_node("Mul", ["out_a", "out_b"], ["mid"], name="n_mul_mid"),
        helper.make_node("Mul", ["mid", "mask3"], ["out_c"], name="n_mul_c"),
    ]
    graph = helper.make_graph(nodes, "masked_mul_product", [ids, a, b],
                              outs + [helper.make_tensor_value_info(
                                  "out_c", TensorProto.FLOAT, [2, 3, 4])],
                              initializer=[pad, axes])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)


def build_model(path, two_muls=True, flat=False):
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
    ]
    if two_muls:
        nodes.append(helper.make_node("Mul", ["b", "tiled"], ["out_b"],
                                      name="n_mul_b"))
    outputs = [out_a] + ([out_b] if two_muls else [])
    if flat:
        # model3 reads the pre-unsqueeze mask outside the site.  A Shape node
        # would be constant-folded away for these static shapes, so use a
        # dtype Cast (not foldable, and the matcher allows any non-Mul user of
        # the chain's Cast result): the fused node must expose the flat mask
        # as an extra output and the Cast checks its values end to end.
        nodes.append(helper.make_node("Cast", ["mask2"], ["flat_mask"],
                                      name="n_flat_cast", to=TensorProto.DOUBLE))
        outputs.append(helper.make_tensor_value_info(
            "flat_mask", TensorProto.DOUBLE, [2, 3]))
    graph = helper.make_graph(nodes, "masked_mul_chain", [ids, a, b],
                              outputs, initializer=[pad, axes, repeats])
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
               ANNC_EP_FUSION_MASKED_MUL="1",
               ANNC_KERNEL_LIB_PATH=str(merged))
    gcc_runtime = Path("/usr/lib/gcc/aarch64-openEuler-linux/12")
    if gcc_runtime.is_dir() and not env.get("LIBRARY_PATH"):
        env["LIBRARY_PATH"] = str(gcc_runtime)
    os.environ.update({k: v for k, v in env.items()
                       if k.startswith("ANNC_") or k == "LIBRARY_PATH"})

    for name in ("shared_chain", "one_mul", "norm", "product", "flat"):
        case = work / name
        case.mkdir(parents=True, exist_ok=True)
        os.environ["ANNC_EP_CACHE_DIR"] = str(case / "cache")
        run_case(case, name)


def run_case(work, kind):
    model_path = work / "masked_mul.onnx"
    if kind == "norm":
        build_norm_model(model_path)
    elif kind == "product":
        build_product_model(model_path)
    elif kind == "flat":
        build_model(model_path, two_muls=False, flat=True)
    else:
        build_model(model_path, kind == "shared_chain")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    feed = inputs()
    if kind == "norm":
        feed["e"] = feed["a"] * 0.25 + 0.5
    output_names = list(OUTPUT_NAMES if kind != "one_mul" else OUTPUT_NAMES[:1])
    if kind in ("norm", "product"):
        output_names.append("out_c")
    elif kind == "flat":
        output_names = ["out_a", "flat_mask"]
    reference = ort.InferenceSession(
        str(model_path), options,
        providers=["CPUExecutionProvider"]).run(None, feed)

    for attempt in range(2):
        options.enable_profiling = True
        options.profile_file_prefix = str(work / f"ep-{attempt}")
        session = ort.InferenceSession(
            str(model_path), options,
            providers=["AnncExecutionProvider", "CPUExecutionProvider"])
        actual = session.run(None, feed)
        for name, expected, got in zip(output_names, reference, actual):
            np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-7,
                                       equal_nan=True)
        profile = json.loads(Path(session.end_profiling()).read_text())
        ep_nodes = [event for event in profile if
                    event.get("cat") == "Node" and
                    event.get("args", {}).get("provider") ==
                    "AnncExecutionProvider"]
        assert len(ep_nodes) == 1, profile
    options.enable_profiling = False

    entries = [p for p in (work / "cache").glob("v1/*/*")
               if p.is_dir() and (p / "DONE").is_file()]
    assert len(entries) == 1, entries
    sites = json.loads((entries[0] / "sites.json").read_text())["sites"]
    assert len(sites) == 1, sites
    site = sites[0]
    assert site["pattern"] == "masked_mul", site
    assert site["abi"] == "annc_execution_v2", site
    assert site["kernel_arg_order"] == [], site
    if kind == "norm":
        # chain (4, no Tile) + three Muls + the Sum and the Div.
        assert len(site["member_node_names"]) == 9, site
        assert [o["rank"] for o in site["outputs"]] == [3, 3, 3], site
        # two data + the Sum's other input + ids
        assert [a["rank"] for a in site["args"]] == [3, 3, 3, 2], site
    elif kind == "product":
        # chain (4, no Tile) + three Muls + the interior product.
        assert len(site["member_node_names"]) == 8, site
        assert [o["rank"] for o in site["outputs"]] == [3, 3, 3], site
        assert [a["rank"] for a in site["args"]] == [3, 3, 2], site
    elif kind == "flat":
        # chain (5, with Tile) + one Mul, plus the flat mask as extra output.
        assert len(site["member_node_names"]) == 6, site
        assert [o["rank"] for o in site["outputs"]] == [3, 2], site
        assert [a["rank"] for a in site["args"]] == [3, 2], site
    elif kind == "shared_chain":
        # chain (5) + two Muls, both results are the site's outputs.
        assert len(site["member_node_names"]) == 7, site
        assert [o["rank"] for o in site["outputs"]] == [3, 3], site
        assert [a["rank"] for a in site["args"]] == [3, 3, 2], site
    else:
        assert kind == "one_mul", kind
        assert len(site["member_node_names"]) == 6, site
        assert len(site["outputs"]) == 1, site
        assert [a["rank"] for a in site["args"]] == [3, 2], site
    assert (entries[0] / "libannc_kernels.so").is_file()
    print(f"{work.name}: one masked_mul context-ABI site covering the whole "
          "subgraph, members and outputs as expected, every output matched the "
          "CPU EP (cold compile + warm cache)")


if __name__ == "__main__":
    main()
