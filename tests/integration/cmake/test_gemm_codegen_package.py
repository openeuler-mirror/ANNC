"""Exercise ANNC binary-package discovery without private implementation code."""

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "cmake/ThirdPartyGemmKernels.cmake"


@pytest.fixture
def project(tmp_path):
    if not shutil.which("cmake"):
        pytest.skip("CMake is required")
    source = tmp_path / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.21)\n"
        "project(GemmPackageContract LANGUAGES NONE)\n"
        'set(CMAKE_SYSTEM_PROCESSOR aarch64)\n'
        'set(CMAKE_INSTALL_LIBDIR lib)\n'
        f'include("{MODULE}")\n'
        'file(WRITE "${CMAKE_BINARY_DIR}/capabilities" '
        '"${ANNC_GEMM_PACKAGE_AVAILABLE}")\n'
    )
    return source


def configure(project, build, package, required=False):
    result = subprocess.run(
        [
            "cmake", "-S", str(project), "-B", str(build),
            f"-DANNC_GEMM_PACKAGE_DIR={package}",
            f"-DANNC_REQUIRE_GEMM_PACKAGE={'ON' if required else 'OFF'}",
        ],
        text=True, capture_output=True, timeout=60,
    )
    return result.returncode, result.stdout + result.stderr


def package_stub(path, api=1):
    """A discovery contract fixture; it is never linked or executed."""
    (path / "cmake").mkdir(parents=True)
    (path / "libannc_gemm_microkernels.a").write_bytes(b"matching archive")
    (path / "cmake/AnncGemmKernelsConfig.cmake").write_text(
        f"set(AnncGemmKernels_API_VERSION {api})\n"
        "set(AnncGemmKernels_KERNEL_ABI_VERSION 1)\n"
        "set(AnncGemmKernels_VERSION test)\n"
        "set(AnncGemmKernels_BUILD_ID fixture)\n"
        "add_library(ANNC::GemmCodegen INTERFACE IMPORTED)\n"
        "add_library(ANNC::AArch64GemmMicrokernels INTERFACE IMPORTED)\n"
    )


def test_missing_package_preserves_ordinary_build(project, tmp_path):
    build = tmp_path / "build"
    status, log = configure(project, build, tmp_path / "absent")
    assert status == 0, log
    assert (build / "capabilities").read_text() == "OFF"
    status, log = configure(project, build, tmp_path / "absent", required=True)
    assert status != 0 and "Complete GEMM-ASM package required" in log


def test_archive_only_is_not_a_package(project, tmp_path):
    package = tmp_path / "archive-only"
    package.mkdir()
    (package / "libannc_gemm_microkernels.a").write_bytes(b"basic archive")
    build = tmp_path / "build"
    status, log = configure(project, build, package)
    assert status == 0, log
    assert (build / "capabilities").read_text() == "OFF"


def test_matching_package_and_cached_path_recheck(project, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    package_stub(first)
    package_stub(second, api=2)
    build = tmp_path / "build"
    status, log = configure(project, build, first, required=True)
    assert status == 0, log
    assert (build / "capabilities").read_text() == "ON"
    status, log = configure(project, build, second, required=True)
    assert status != 0 and "API 1 and kernel ABI 1 are required" in log
