#!/usr/bin/env python3
"""End-to-end: annc-onnx-pipeline -> batched_matmul_add context-ABI kernel -> EP.

The fixture is the ONNX rank-3 MatMul + per-batch bias Add the real models'
carry, including batched weights and a constant pre-affine input chain:

    x [B, T, K] * scale[K] + offset[K]
      -> MatMul(., W [B, K, N]) -> Add(., bias [B, 1, N])

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


def build_model(path, batch=1, affine=False, commuted=False, version=0,
                broadcast_weight=False, batched_affine=False,
                broadcast_data=False):
    data_batch = 1 if broadcast_data else batch
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT,
                                     [data_batch, SEQ, K])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, SEQ, N])
    weight_batch = 1 if broadcast_weight else batch
    w = numpy_helper.from_array(
        ((np.arange(weight_batch * K * N, dtype=np.float32)
          .reshape(weight_batch, K, N) - 96.0) / 37.0) + version * 0.25, "w")
    b = numpy_helper.from_array(
        np.linspace(-0.2, 0.2, batch * N,
                    dtype=np.float32).reshape(batch, 1, N) + version, "b")
    scale = numpy_helper.from_array(
        np.linspace(0.75, 1.25, batch * K if batched_affine else K,
                    dtype=np.float32).reshape((batch, 1, K)
                    if batched_affine else (K,)) + version * 0.125, "scale")
    offset = numpy_helper.from_array(
        np.linspace(-0.3, 0.3, K, dtype=np.float32) + version * 0.1, "offset")
    nodes = []
    if affine:
        nodes = [
            helper.make_node("Mul", ["scale", "x"] if commuted
                             else ["x", "scale"], ["scaled"], name="n_mul"),
            helper.make_node("Add", ["offset", "scaled"] if commuted
                             else ["scaled", "offset"], ["affined"],
                             name="n_affine"),
        ]
    nodes.extend([
        helper.make_node("MatMul", ["affined" if affine else "x", "w"],
                         ["mm"], name="n_mm"),
        helper.make_node("Add", ["b", "mm"] if commuted else ["mm", "b"],
                         ["y"], name="n_add"),
    ])
    graph = helper.make_graph(nodes, "bmm_add", [x], [y],
                              initializer=[w, b, scale, offset] if affine
                              else [w, b])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)


def build_rank4_model(path, broadcast_data=False):
    """Rank-4 sites: the per-domain projections of model3.

    x [B0, 1, T, K] -> MatMul(., W [B0, 1, K, N]) -> Add(., bias [B0, 1, 1, N])
    -> y [B0, 1, T, N].  The second batch dim stays 1 in the target models, but
    the kernel treats both leading dims the same way, so `broadcast_data` also
    exercises a data batch that is broadcast to the output batch.
    """
    batch, seq, k, n = 2, 5, 12, 16
    data_batch = 1 if broadcast_data else batch
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT,
                                     [data_batch, 1, seq, k])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT,
                                     [batch, 1, seq, n])
    w = numpy_helper.from_array(
        (np.arange(batch * k * n, dtype=np.float32)
         .reshape(batch, 1, k, n) - 96.0) / 37.0, "w")
    b = numpy_helper.from_array(
        np.linspace(-0.2, 0.2, batch * n, dtype=np.float32)
        .reshape(batch, 1, 1, n), "b")
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["mm"], name="n_mm"),
        helper.make_node("Add", ["mm", "b"], ["y"], name="n_add"),
    ]
    graph = helper.make_graph(nodes, "bmm_add_rank4", [x], [y],
                              initializer=[w, b])
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

    cases = [
        ("original", 1, False, False, False, False),
        ("batched", 2, False, False, False, False),
        ("broadcast_weight", 2, False, False, True, False),
        ("affine_broadcast_weight", 2, True, False, True, False),
        ("affine", 2, True, False, False, False),
        ("commuted", 2, True, True, False, False),
        ("batched_affine", 2, True, True, False, True),
    ]
    for name, batch, affine, commuted, broadcast_weight, batched_affine in cases:
        case_work = work / name
        case_work.mkdir(parents=True, exist_ok=True)
        os.environ["ANNC_EP_CACHE_DIR"] = str(case_work / "cache")
        run_case(case_work, batch, affine, commuted, broadcast_weight,
                 batched_affine)
    case_work = work / "broadcast_data"
    case_work.mkdir(parents=True, exist_ok=True)
    os.environ["ANNC_EP_CACHE_DIR"] = str(case_work / "cache")
    run_case(case_work, 2, False, False, False, False, broadcast_data=True)

    for name, broadcast_data in (("rank4", False), ("rank4_broadcast_data",
                                                     True)):
        case_work = work / name
        case_work.mkdir(parents=True, exist_ok=True)
        os.environ["ANNC_EP_CACHE_DIR"] = str(case_work / "cache")
        run_rank4_case(case_work, broadcast_data)


def run_rank4_case(work, broadcast_data):
    model_path = work / "bmm_add_rank4.onnx"
    build_rank4_model(model_path, broadcast_data)
    rng = np.random.default_rng(13)
    feed = {"x": rng.standard_normal(
        (1 if broadcast_data else 2, 1, 5, 12)).astype(np.float32)}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    reference = ort.InferenceSession(
        str(model_path), options,
        providers=["CPUExecutionProvider"]).run(None, feed)[0]

    for attempt in range(2):
        options.enable_profiling = True
        options.profile_file_prefix = str(work / f"ep-{attempt}")
        session = ort.InferenceSession(
            str(model_path), options,
            providers=["AnncExecutionProvider", "CPUExecutionProvider"])
        actual = session.run(None, feed)[0]
        np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-7,
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
    site = json.loads((entries[0] / "sites.json").read_text())["sites"][0]
    assert site["pattern"] == "batched_matmul_add", site
    assert site["abi"] == "annc_execution_v2", site
    assert [arg["rank"] for arg in site["args"]] == [4, 4, 4], site
    assert [len(arg["shape"]) for arg in site["args"]] == [4, 4, 4], site
    assert len(site["member_node_names"]) == 2, site
    print(f"{work.name}: real rank-4 BMM site, cold/warm cache match")


def run_case(work, batch, affine, commuted, broadcast_weight, batched_affine,
             broadcast_data=False):
    model_path = work / "bmm_add.onnx"
    rng = np.random.default_rng(11)
    feed = {"x": rng.standard_normal((1 if broadcast_data else batch, SEQ, K))
            .astype(np.float32)}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # ORT graph fusions would otherwise hide the four-node affine chain from
    # ANNC, preventing the test from exercising compile-time folding.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for version in range(2 if affine else 1):
        build_model(model_path, batch, affine, commuted, version,
                    broadcast_weight, batched_affine, broadcast_data)
        run_model(work, model_path, options, feed, affine, version,
                  folded=affine and not broadcast_weight)


def run_model(work, model_path, options, feed, affine, version, folded):
    reference = ort.InferenceSession(
        str(model_path), options,
        providers=["CPUExecutionProvider"]).run(None, feed)[0]

    for attempt in range(2):
        options.enable_profiling = True
        options.profile_file_prefix = str(work / f"ep-{version}-{attempt}")
        session = ort.InferenceSession(
            str(model_path), options,
            providers=["AnncExecutionProvider", "CPUExecutionProvider"])
        actual = session.run(None, feed)[0]
        np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-7,
                                   equal_nan=True)
        profile = json.loads(Path(session.end_profiling()).read_text())
        ep_nodes = [event for event in profile if
                    event.get("cat") == "Node" and
                    event.get("args", {}).get("provider") ==
                    "AnncExecutionProvider"]
        assert len(ep_nodes) == 1, profile
        print(f"session {attempt + 1}: numeric match")
    options.enable_profiling = False

    entries = [p for p in (work / "cache").glob("v1/*/*")
               if p.is_dir() and (p / "DONE").is_file()]
    assert len(entries) == version + 1, entries
    reports = [json.loads((entry / "sites.json").read_text())
               for entry in entries]
    report = reports[0]
    sites = report["sites"]
    assert len(sites) == 1, sites
    site = sites[0]
    assert site["pattern"] == "batched_matmul_add", site
    assert site["abi"] == "annc_execution_v2", site
    assert site["kernel_arg_order"] == [], site
    # MatMul + Add (the names the EP sees are its own export's).
    assert len(site["member_node_names"]) == (4 if folded else 2), site
    assert [arg["rank"] for arg in site["args"]] == [3, 3, 3], site
    assert all((entry / "libannc_kernels.so").is_file() for entry in entries)
    print(f"{work.name} version {version}: real BMM site, cold/warm cache match")


if __name__ == "__main__":
    main()
