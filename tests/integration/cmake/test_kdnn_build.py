"""Exercise KDNN import capability detection without downloading dependencies."""

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "cmake/ThirdPartyKDNN.cmake"


def _run(command):
    result = subprocess.run(
        [str(arg) for arg in command], text=True, capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


@pytest.fixture
def toolchain():
    tools = {name: shutil.which(name) for name in ("cmake", "g++", "ar")}
    if not all(tools.values()):
        pytest.skip("requires CMake, g++ and ar")
    return tools


def _library(path, capability, tools):
    include = path / "include"
    archive = path / "src"
    include.mkdir(parents=True)
    archive.mkdir()
    # Declarations alone must not enable registration with an older binary.
    methods = [
        "void RunWithPackedB(const void*, const void*, void*) const",
        "void RunWithPackedB(const void*, const void*, void*, const void*) const",
        "void RunWithPackedB(const void*, const void*, void*, const void*, "
        "const PostOpsDataPtrs&) const",
    ]
    declared = methods if capability != "ordinary" else []
    (include / "kdnn.hpp").write_text(
        "#pragma once\n#include <vector>\nnamespace KDNN {\n"
        "using PostOpsDataPtrs = std::vector<const void*>;\n"
        "struct Gemm {\n" + ";\n".join(declared) + ";\n};\n}\n"
    )
    (include / "kdnn_config.h").write_text("#pragma once\n")
    defined = methods if capability == "packed" else []
    if capability == "partial":
        defined = methods[:2]
    source = path / "stub.cpp"
    source.write_text(
        '#include "kdnn.hpp"\nnamespace KDNN {\n'
        + "\n".join(method.replace("void Run", "void Gemm::Run", 1) + " {}"
                    for method in defined)
        + "\n}\n"
    )
    _run([tools["g++"], "-c", source, "-I", include, "-o", path / "stub.o"])
    _run([tools["ar"], "rcs", archive / "libkdnn.a", path / "stub.o"])
    # A staged RELEASE tree should take the cache path with no network access.
    (path / ".annc_kdnn_release_stamp").write_text("unused\nsve-threadpool\n")


def _project(path):
    path.mkdir()
    (path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(KdnnImportTest LANGUAGES CXX)\n"
        f'include("{MODULE.as_posix()}")\n'
        'if(ANNC_KDNN_HAS_PACKED_B)\n'
        '  set(capability "packed")\n'
        'else()\n'
        '  set(capability "ordinary")\n'
        'endif()\n'
        'file(WRITE "${CMAKE_BINARY_DIR}/capability.txt" "${capability}")\n'
    )


def _configure(project, build, library, source, tools):
    _run([
        tools["cmake"], "-S", project, "-B", build,
        "-DANNC_ENABLE_KDNN_ADAPTOR=ON", f"-DANNC_KDNN_SOURCE={source}",
        f"-DANNC_KDNN_DIR={library}", f"-DKDNN_DIR={library}",
        f"-DANNC_KDNN_RELEASE_DIR={library}",
        "-DANNC_KDNN_RELEASE_URL=unused", f'-DCMAKE_CXX_COMPILER={tools["g++"]}',
    ])
    return (build / "capability.txt").read_text()


@pytest.mark.parametrize("source", ["LOCAL", "RELEASE"])
@pytest.mark.parametrize(
    "capability", ["ordinary", "header_only", "partial", "packed"]
)
def test_import_requires_all_packed_overloads(
    tmp_path, toolchain, source, capability
):
    """Missing declarations or symbols keep ordinary KDNN imports usable."""
    library = tmp_path / "kdnn"
    _library(library, capability, toolchain)
    project = tmp_path / "project"
    _project(project)
    actual = _configure(project, tmp_path / "build", library, source, toolchain)
    assert actual == ("packed" if capability == "packed" else "ordinary")


def test_reconfigure_rechecks_dependency(tmp_path, toolchain):
    """A cached successful probe cannot enable adapters for another library."""
    project = tmp_path / "project"
    _project(project)
    build = tmp_path / "build"
    for capability in ("packed", "ordinary", "header_only"):
        library = tmp_path / capability
        _library(library, capability, toolchain)
        actual = _configure(project, build, library, "LOCAL", toolchain)
        assert actual == ("packed" if capability == "packed" else "ordinary")
