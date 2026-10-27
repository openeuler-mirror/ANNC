"""Check weight-independent JIT kernels against full numerical references.

Requires an AArch64 constant-folding build. Set ANNC_BUILD_DIR to select it;
the usual driver environment overrides remain available.
"""

import ctypes as ct
import json
import os
from pathlib import Path
import platform
import subprocess

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/FileCheck/Target/aarch64/Inputs" / (
    "gemm-tuning-hip09-neon-test.json"
)


class Memref(ct.Structure):
    _fields_ = [
        ("allocated", ct.c_void_p),
        ("aligned", ct.c_void_p),
        ("offset", ct.c_int64),
        ("sizes", ct.c_int64 * 2),
        ("strides", ct.c_int64 * 2),
    ]


class PackedRhsSpec(ct.Structure):
    _fields_ = [
        (name, ct.c_int64)
        for name in (
            "version argument k n kc nc isa panel elements alignment"
        ).split()
    ]


def _descriptor(array, shape):
    rows, cols = shape
    return Memref(
        array.ctypes.data, array.ctypes.data, 0,
        (ct.c_int64 * 2)(rows, cols), (ct.c_int64 * 2)(cols, 1),
    )


def _run(command, env, directory):
    result = subprocess.run(
        [os.fspath(arg) for arg in command],
        env=env, cwd=directory, text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, (
        f"command failed: {command}\n{result.stdout}\n{result.stderr}"
    )


@pytest.fixture(scope="module")
def toolchain():
    if platform.machine().lower() not in ("aarch64", "arm64"):
        pytest.skip("requires native AArch64 execution")
    build = Path(os.environ.get("ANNC_BUILD_DIR", ROOT / "build")).resolve()
    cache_file = build / "CMakeCache.txt"
    if not cache_file.is_file():
        pytest.skip(f"ANNC build is not configured: {build}")
    cache = {}
    for line in cache_file.read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    if cache.get("ANNC_ENABLE_CONSTANT_FOLDING") != "ON":
        pytest.skip("requires ANNC_ENABLE_CONSTANT_FOLDING=ON")
    for name in ("annc", "annc-asm"):
        if not (build / "bin" / name).is_file():
            pytest.skip(f"{name} is not built")
    env = os.environ.copy()
    llvm = build / "_deps/llvm-build"
    env["PATH"] = os.pathsep.join([
        str(build / "bin"), str(llvm / "bin"), env.get("PATH", ""),
    ])
    libraries = [build / "lib", llvm / "lib"]
    if cache.get("CMAKE_INSTALL_PREFIX"):
        libraries.append(Path(cache["CMAKE_INSTALL_PREFIX"]) / "lib")
    libraries.append(ROOT / "third_party/annc-gemm-kernels")
    env["LIBRARY_PATH"] = os.pathsep.join(
        [str(path) for path in libraries] + [env.get("LIBRARY_PATH", "")]
    )
    if (llvm / "bin/clang").is_file():
        env.setdefault("ANNC_CLANG", str(llvm / "bin/clang"))
    env.update(ANNC_GEMM="2", ANNC_BACKEND="generic")
    return build, env


def _compile(directory, batch, threads, prepack, toolchain, config=CONFIG):
    build, env = toolchain
    directory.mkdir()
    source = directory / "input.mlir"
    source.write_text(f'''
func.func @test(%a: memref<{batch}x73xf32>, %b: memref<73x67xf32>,
                %c: memref<{batch}x67xf32>)
    attributes {{llvm.emit_c_interface}} {{
  linalg.matmul {{annc.aarch64.rhs_name = "weight"}}
      ins(%a, %b : memref<{batch}x73xf32>, memref<73x67xf32>)
      outs(%c : memref<{batch}x67xf32>)
  return
}}
''')
    options = f"config-path={config} intra-thread-count={threads}"
    packed_c = directory / "pack.c"
    if prepack:
        manifest = directory / "inputs.json"
        manifest.write_text(json.dumps([1]))
        options += f" packed-c={packed_c} jit-inputs={manifest}"
    lowered = directory / "lowered.mlir"
    _run([
        build / "bin/annc-asm", source,
        "--annc-aarch64-gemm-pipeline=" + options, "-o", lowered,
    ], env, directory)
    if prepack:
        text = lowered.read_text()
        assert "memref.get_global" not in text
        assert "annc_jit_packed_rhs_v1" in packed_c.read_text()
    library = directory / "kernel.so"
    command = [
        build / "bin/annc", lowered, "--shared",
        "--kernel-lib-path", build / "lib", "-o", library,
    ]
    if prepack:
        command += ["--packed-rhs-c", packed_c]
    _run(command, env, directory)
    return ct.CDLL(str(library))


@pytest.mark.parametrize("batch", [1, 8, 32, 100])
@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("isa", ["neon", "sve"])
def test_shared_kernel_uses_each_weight(tmp_path, toolchain, batch, threads, isa):
    """Tail tiles and thread slices must work with two independent weights.

    Without a TF thread pool the dispatcher executes tasks serially; these
    cases exercise task coordinates, not parallel scheduler throughput.
    """
    config = CONFIG
    if isa == "sve":
        vl = ct.CDLL(None).prctl(51, 0, 0, 0, 0)  # PR_SVE_GET_VL
        if vl < 0 or vl & 0xffff != 32:
            pytest.skip("requires SVE VL=32 bytes for the target ABI")
        config = CONFIG.with_name("gemm-tuning-test.json")
    rng = np.random.default_rng(421 + batch)
    lhs = rng.normal(0, 0.1, (batch, 73)).astype("float32")
    weights = [
        rng.normal(0, 0.1, (73, 67)).astype("float32") for _ in range(2)
    ]
    for prepack in (False, True):
        library = _compile(
            tmp_path / str(prepack), batch, threads, prepack, toolchain, config,
        )
        count = ct.c_int64(0)
        if prepack:
            getter = library.annc_jit_packed_rhs_v1
            getter.argtypes = [ct.POINTER(ct.c_int64)]
            getter.restype = ct.POINTER(PackedRhsSpec)
            specs = getter(ct.byref(count))
            assert count.value == (0 if batch == 1 and isa == "neon" else 1)
        packed_weights = []
        for weight in weights:
            rhs = weight
            if count.value:
                spec = specs[0]
                assert (spec.version, spec.argument, spec.k, spec.n) == (
                    1, 1, 73, 67,
                )
                storage = np.zeros(spec.elements + 16, dtype="float32")
                offset = (-storage.ctypes.data % 64) // 4
                rhs = storage[offset:offset + spec.elements]
                prepare = library.annc_jit_prepare_rhs_v1
                prepare.argtypes = [ct.c_int64, ct.c_void_p, ct.c_void_p]
                prepare.restype = None
                prepare(0, weight.ctypes.data, rhs.ctypes.data)
            packed_weights.append(rhs)
        function = library._mlir_ciface_test
        function.argtypes = [ct.POINTER(Memref)] * 3
        function.restype = None
        # Revisit the first buffer after using another weight with the same code.
        for index in (0, 1, 0):
            weight, rhs = weights[index], packed_weights[index]
            output = np.full((batch, 67), np.nan, dtype="float32")
            descriptors = (
                _descriptor(lhs, lhs.shape), _descriptor(rhs, weight.shape),
                _descriptor(output, output.shape),
            )
            function(*(ct.byref(value) for value in descriptors))
            expected = lhs.astype("float64") @ weight.astype("float64")
            np.testing.assert_allclose(output, expected, atol=1e-5, rtol=1e-5)
