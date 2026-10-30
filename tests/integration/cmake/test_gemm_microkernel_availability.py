"""Exercise built compiler tools with and without the optional GEMM archive.

Run against both configurations using ANNC_BUILD_DIR. The missing-archive
configuration must build annc-asm/annc-opt with
-DANNC_AARCH64_GEMM_KERNEL_LIBRARY=/nonexistent/libannc_gemm_microkernels.a.
"""

import ctypes
import os
from pathlib import Path
import platform
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/FileCheck/Target/aarch64"


@pytest.fixture(scope="module")
def compiler():
    build = Path(os.environ.get("ANNC_BUILD_DIR", ROOT / "build")).resolve()
    cache_file = build / "CMakeCache.txt"
    if not cache_file.is_file():
        pytest.skip(f"ANNC build is not configured: {build}")
    cache = {}
    for line in cache_file.read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    for name in ("annc-asm", "annc-opt"):
        if not (build / "bin" / name).is_file():
            pytest.skip(f"{name} is not built")
    archive = cache.get("ANNC_AARCH64_GEMM_KERNEL_LIBRARY", "")
    available = platform.machine().lower() in ("aarch64", "arm64") and bool(
        archive and Path(archive).is_file()
    )
    return build, available


def run(command):
    return subprocess.run(
        [os.fspath(arg) for arg in command],
        text=True, capture_output=True, timeout=60,
    )


def tuning(isa):
    name = "gemm-tuning-test.json" if isa == "sve" else (
        "gemm-tuning-hip09-neon-test.json"
    )
    return FIXTURES / "Inputs" / name


@pytest.mark.parametrize("tool", ["annc-asm", "annc-opt"])
def test_tools_start_without_optional_symbols(compiler, tool):
    build, _ = compiler
    result = run([build / "bin" / tool, "--help"])
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("isa", ["neon", "sve"])
def test_ordinary_codegen_without_host_packing(compiler, isa):
    build, _ = compiler
    result = run([
        build / "bin/annc-asm", FIXTURES / "gemm-jit-prepack.mlir",
        f"--annc-aarch64-gemm-pipeline=config-path={tuning(isa)}",
    ])
    assert result.returncode == 0, result.stderr
    assert f"annc_aarch64_{isa}_packb_f32" in result.stdout


def test_static_prepack_reports_missing_archive(compiler, tmp_path):
    build, available = compiler
    source = tmp_path / "empty.mlir"
    source.write_text("module {}\n")
    result = run([
        build / "bin/annc-asm", source, "--aarch64-gemm-prepack-rhs",
    ])
    if available:
        assert result.returncode == 0, result.stderr
        assert "no GEMM packed" in result.stderr
    else:
        assert result.returncode != 0
        assert "static RHS prepacking requires the AArch64 GEMM" in result.stderr
        assert "ANNC_AARCH64_GEMM_KERNEL_LIBRARY" in result.stderr


@pytest.mark.parametrize("isa", ["neon", "sve"])
def test_jit_prepack_respects_host_sizing_capability(compiler, tmp_path, isa):
    build, available = compiler
    if isa == "sve" and available:
        vl = ctypes.CDLL(None).prctl(51, 0, 0, 0, 0)  # PR_SVE_GET_VL
        if vl < 0:
            pytest.skip("host SVE sizing helper requires SVE support")
    manifest = tmp_path / "inputs.json"
    manifest.write_text("[1]\n")
    packed = tmp_path / "pack.c"
    result = run([
        build / "bin/annc-asm", FIXTURES / "gemm-jit-prepack.mlir",
        f"--annc-aarch64-gemm-pipeline=config-path={tuning(isa)} "
        f"packed-c={packed} jit-inputs={manifest}",
    ])
    if isa == "sve" and not available:
        assert result.returncode != 0
        assert "JIT SVE RHS prepacking requires the AArch64 GEMM" in result.stderr
        return
    if isa == "sve":
        # This checks host sizing, without executing a generated SVE kernel.
        if vl & 0xffff != 32:
            assert result.returncode != 0
            assert "SVE vector length does not match target" in result.stderr
            return
    assert result.returncode == 0, result.stderr
    assert f"annc_aarch64_{isa}_packb_f32" in packed.read_text()
    assert "*count=1" in packed.read_text()
    assert "__annc_aarch64_gemm_pack_b_leaf" not in result.stdout


def test_sve_prepacked_leaf_reports_missing_archive(compiler, tmp_path):
    build, available = compiler
    if available and ctypes.CDLL(None).prctl(51, 0, 0, 0, 0) < 0:
        pytest.skip("host SVE sizing helper requires SVE support")
    source = tmp_path / "sve-prepacked.mlir"
    text = (FIXTURES / "gemm-prepacked-thread-tasks.mlir").read_text()
    replacements = {
        'target_arch = "hip09"': 'target_arch = "hip12"',
        'isa = "neon"': 'isa = "sve"',
        'kernel_family = "annc-neon-f32-v1"':
            'kernel_family = "annc-sve-f32-v1"',
        'pack_b_schema = "annc-neon-packed-b-v1"':
            'pack_b_schema = "annc-sve-packed-b-v2"',
        "vector_length_bytes = 16": "vector_length_bytes = 32",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    source.write_text(text)
    result = run([
        build / "bin/annc-asm", source,
        "--aarch64-gemm-thread-tiling", "--aarch64-gemm-cache-blocking",
        "--aarch64-gemm-kernel-tiling", "--aarch64-gemm-leaf-materialization",
    ])
    if available:
        assert result.returncode == 0, result.stderr
        assert "__annc_aarch64_gemm_microkernel_leaf" in result.stdout
    else:
        assert result.returncode != 0
        assert "SVE prepacked RHS requires the AArch64 GEMM" in result.stderr
