"""Check the support JIT's executed compiler command in both build routes."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("v2", [False, True], ids=["legacy", "v2"])
def test_jit_pipeline_options(tmp_path, v2):
    for tool in ("cmake", "c++"):
        if not shutil.which(tool):
            pytest.skip(f"{tool} is required to build the support JIT")
    if not (ROOT / "third_party/json/include/nlohmann/json.hpp").exists():
        pytest.skip("the managed nlohmann/json source is required")

    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.15)\n"
        "project(jit_pipeline_probe LANGUAGES CXX)\n"
        f'add_subdirectory("{ROOT}/addons/tensorflow/support" support)\n'
        "add_executable(probe probe.cpp)\n"
        "target_link_libraries(probe PRIVATE annc_tensorflow_support)\n"
    )
    (tmp_path / "probe.cpp").write_text(
        '#include "jit_compiler.h"\n'
        "int main() {\n"
        "  annc::jit::JitCompileRequest request{};\n"
        '  request.atir_module_path = "template.mlir";\n'
        '  request.kernel_name = "fused";\n'
        "  request.argument_shapes = {{2, 3}, {3, 4}};\n"
        "  AnncJitKernel kernel;\n"
        "  return annc::jit::CompileAnncJitKernel(request, &kernel) == "
        "ANNC_JIT_OK;\n"
        "}\n"
    )
    build = tmp_path / "build"
    subprocess.run(
        [
            "cmake", "-S", str(tmp_path), "-B", str(build),
            f"-DANNC_ENABLE_TF2ATIR_V2={'ON' if v2 else 'OFF'}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--parallel", "2"],
        check=True,
        capture_output=True,
        text=True,
    )

    capture = tmp_path / "compiler-arguments.txt"
    compiler = tmp_path / "annc-asm"
    compiler.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$JIT_CAPTURE"\nexit 1\n')
    compiler.chmod(0o755)
    env = dict(
        os.environ,
        ANNC_PIPELINE_PATH=str(tmp_path / "annc-tf-pipeline"),
        ANNC_WORK_DIR=str(tmp_path),
        JIT_CAPTURE=str(capture),
    )
    subprocess.run([str(build / "probe")], env=env, check=True)
    arguments = capture.read_text().splitlines()
    shape_option = arguments[4 if v2 else 2].split("=shape-spec=", 1)
    assert len(shape_option) == 2
    expected = ["template.mlir", "--annc-select-kernel=kernel-name=fused"]
    if v2:
        expected += ["--shape-legalize-to-stablehlo", "--reconcile-unrealized-casts"]
    expected += [
        f"--{'annc' if v2 else 'atir'}-specialize-shapes=shape-spec={shape_option[1]}",
        "--annc-fast-codegen", "--annc-aarch64-gemm-pipeline", "-o",
        str(Path(shape_option[1]).with_name("kernel_lowered.mlir")),
    ]
    assert arguments == expected
