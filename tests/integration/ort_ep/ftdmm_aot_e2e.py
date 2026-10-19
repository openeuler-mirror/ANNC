#!/usr/bin/env python3
"""End-to-end: annc-onnx-pipeline -> AOT kernel -> ORT EP -> numeric check.

The fixture model is a 16-node `tensordot_matmul` region extracted from the
real profile1 OrtGraph export. It compiles in well under a second and only
depends on the ANNC install next to the pipeline binary.
"""

import argparse
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import onnxruntime as ort
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

MODEL = Path(__file__).parent / "fixtures" / "ftdmm_aot_e2e.onnx"
INPUT = "t558_0"


def find_pipeline():
    override = os.environ.get("ANNC_EP_PIPELINE")
    if override and Path(override).is_file():
        return Path(override)
    for candidate in ("/home/y00897125/ANNC/install-dev/bin/annc-onnx-pipeline",):
        if Path(candidate).is_file():
            return Path(candidate)
    raise RuntimeError("annc-onnx-pipeline not found; set ANNC_EP_PIPELINE")


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
    parser.add_argument("--work-dir")
    parser.add_argument("--llvm-lib-dir")
    parser.add_argument("--extra-lib-dir", action="append", default=[])
    args = parser.parse_args()
    work = Path(args.work_dir or tempfile.mkdtemp(prefix="annc-ftdmm-e2e-"))
    (work / "cache").mkdir(parents=True, exist_ok=True)
    runtime = Path(rtpy.__file__).parent / "libonnxruntime.so.1.26.0"
    ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
    rtpy.register_execution_provider_library("AnncExecutionProvider", args.ep_library)
    env = dict(os.environ,
               ANNC_EP_PIPELINE=str(find_pipeline()),
               ANNC_EP_CACHE_DIR=str(work / "cache"))
    merged_lib_dir = None
    if args.llvm_lib_dir or args.extra_lib_dir:
        merged_lib_dir = merge_kernel_libs(
            work, [args.llvm_lib_dir] + args.extra_lib_dir)
        env["ANNC_KERNEL_LIB_PATH"] = str(merged_lib_dir)
    if not env.get("ANNC_KERNEL_LIB_PATH"):
        lib_dir = Path(env["ANNC_EP_PIPELINE"]).parent.parent / "lib"
        if not (lib_dir / "libANNCBuiltinKernels.a").exists():
            lib_dir = Path(env["ANNC_EP_PIPELINE"]).parent.parent / "lib64"
        if not (lib_dir / "libLLVMSupport.a").exists() and Path("/tmp/annc-dev-kernels").is_dir():
            lib_dir = Path("/tmp/annc-dev-kernels")
        env["ANNC_KERNEL_LIB_PATH"] = str(lib_dir)
    gcc_runtime = Path("/usr/lib/gcc/aarch64-openEuler-linux/12")
    if gcc_runtime.is_dir() and not env.get("LIBRARY_PATH"):
        env["LIBRARY_PATH"] = str(gcc_runtime)
    if not env.get("ANNC_CLANG") and Path("/tmp/annc-clang-wrapper").is_file():
        env["ANNC_CLANG"] = "/tmp/annc-clang-wrapper"
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    x = np.arange(16, dtype=np.float32).reshape(1, 2, 8) / 7
    ref = ort.InferenceSession(str(MODEL), o,
                               providers=["CPUExecutionProvider"]).run(None, {INPUT: x})[0]
    os.environ.update({k: v for k, v in env.items()
                       if k.startswith("ANNC_") or k == "LIBRARY_PATH"})
    outputs = []
    for attempt in range(2):
        session = ort.InferenceSession(
            str(MODEL), o,
            providers=["AnncExecutionProvider", "CPUExecutionProvider"])
        outputs.append(session.run(None, {INPUT: x})[0])
        np.testing.assert_allclose(outputs[-1], ref, rtol=1e-3, atol=1e-7,
                                   equal_nan=True)
        print(f"session {attempt + 1}: numeric match")
    entries = [p for p in (work / "cache").glob("v1/*/*") if p.is_dir() and
               (p / "DONE").is_file()]
    assert len(entries) == 1 and (entries[0] / "DONE").is_file(), entries
    assert (entries[0] / "sites.json").is_file()
    print("real AOT pipeline warm cache reused across sessions")


if __name__ == "__main__":
    main()
