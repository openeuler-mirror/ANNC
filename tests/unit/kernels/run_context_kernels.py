#!/usr/bin/env python3
"""Run real context kernels; optionally compile their fusion entries with MLIR."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


def compile_entries(args, root, build):
    objects = []
    fixtures = ("context-abi-v3.mlir", "context-gather-v3.mlir")
    for index, fixture in enumerate(fixtures):
        lowered = build / f"entry{index}.mlir"
        llvm_mlir = build / f"entry{index}.llvm.mlir"
        llvm_ir = build / f"entry{index}.ll"
        obj = build / f"entry{index}.o"
        subprocess.run([
            args.annc_asm,
            str(root / "tests/FileCheck/Dialect/Atir" / fixture),
            "--atir-fast-codegen", "--lower-context-kernel-entries",
            "--convert-atir-to-linalg",
            "-o", str(lowered),
        ], check=True)
        subprocess.run([
            args.mlir_opt, str(lowered), "--convert-func-to-llvm",
            "--reconcile-unrealized-casts", "-o", str(llvm_mlir),
        ], check=True)
        subprocess.run([
            args.mlir_translate, "--mlir-to-llvmir", str(llvm_mlir),
            "-o", str(llvm_ir),
        ], check=True)
        subprocess.run([
            args.llc, "-filetype=obj", "-relocation-model=pic",
            str(llvm_ir), "-o", str(obj),
        ], check=True)
        objects.append(str(obj))
    return objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annc-asm", help="Also compile and execute fusion entries using this tool"
    )
    parser.add_argument("--mlir-opt", default="mlir-opt")
    parser.add_argument("--mlir-translate", default="mlir-translate")
    parser.add_argument("--llc", default="llc")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    compiler = shlex.split(os.environ.get("CXX", ""))
    if not compiler:
        found = shutil.which("clang++") or shutil.which("c++")
        if not found:
            raise SystemExit("C++17 compiler required (set CXX)")
        compiler = [found]
    common = compiler + ["-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror",
                         "-I", str(root / "annc/include"),
                         "-I", str(root / "context/include"), "-I", str(root)]
    if os.environ.get("ANNC_CONTEXT_SANITIZE") == "1":
        common += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    with tempfile.TemporaryDirectory(prefix="annc-context-kernels-") as directory:
        build = Path(directory)
        wrappers = []
        builtin = root / "annc/lib/Kernel/builtin_kernels"
        for spec in sorted(builtin.glob("*_kernel_specs.inc")):
            if "ANNC_CONTEXT_KERNEL_SPEC" not in spec.read_text():
                continue
            stem = spec.stem.removesuffix("_kernel_specs")
            wrapper = build / (stem + ".cpp")
            include = spec.relative_to(root)
            implementation = spec.with_name(stem + "_aarch64.cpp").relative_to(root)
            wrapper.write_text(
                f'#define ANNC_BUILTIN_KERNEL_SPECS_FILE "{include}"\n'
                '#include "Kernel/BuiltinKernelSpecDeclare.h"\n'
                f'#include "{implementation}"\n'
                f'#define ANNC_BUILTIN_KERNEL_SPECS_FILE "{include}"\n'
                '#include "Kernel/BuiltinKernelSpecDefine.h"\n')
            wrappers.append(str(wrapper))
        objects = compile_entries(args, root, build) if args.annc_asm else []
        library = build / "libcontext_kernels.so"
        driver = build / "context_kernels"
        subprocess.run(common + ["-shared", "-fPIC"] + wrappers + objects +
                       ["-o", str(library)], check=True)
        command = common + [str(root / "tests/unit/kernels/kernel_context_driver.cpp"),
                            "-o", str(driver)]
        if os.uname().sysname != "Darwin":
            command += ["-ldl"]
        subprocess.run(command, check=True)
        invocation = [str(driver), str(library)]
        if args.annc_asm:
            invocation += ["_mlir_ciface_context_matmul",
                           "_mlir_ciface_fused_kp_fused_gather"]
        subprocess.run(invocation, check=True)
        if args.annc_asm:
            print("PASS: compiled model entries forward one context to real kernels")


if __name__ == "__main__":
    main()
