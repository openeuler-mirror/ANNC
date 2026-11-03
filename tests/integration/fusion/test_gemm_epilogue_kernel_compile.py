"""GEMM object compilation and driver tests against a built ANNC toolchain."""

import ctypes
import os
from pathlib import Path
import platform
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/FileCheck/Target/aarch64"


@pytest.fixture(scope="module")
def tools():
    build = Path(os.environ.get("ANNC_BUILD_DIR", ROOT / "build")).resolve()
    for name in ("annc", "annc-opt", "annc-asm", "annc-gemm-epilogue-kernel-compile"):
        if not (build / "bin" / name).is_file():
            pytest.skip(f"{name} is not built in {build}")
    env = dict(os.environ, ANNC_GEMM="2",
               ANNC_KERNEL_LIB_PATH=str(build / "lib"))
    env["LIBRARY_PATH"] = os.pathsep.join([
        str(build / "lib"), str(ROOT / "third_party/annc-gemm-kernels"),
        env.get("LIBRARY_PATH", ""),
    ])
    return build, env


def run(command, env):
    return subprocess.run([str(arg) for arg in command], env=env,
                          text=True, capture_output=True, timeout=180)


def compile_kernels(tools, tmp_path, text):
    build, env = tools
    source = tmp_path / "input.mlir"
    source.write_text(text)
    return run([build / "bin/annc-gemm-epilogue-kernel-compile", source,
                "--output-dir", tmp_path], env)


def objects_dir(tmp_path):
    return tmp_path / "model_generated_kernels"


def test_empty_model_creates_empty_object_directory(tools, tmp_path):
    result = compile_kernels(tools, tmp_path, "module {}\n")
    assert result.returncode == 0, result.stderr
    assert list(objects_dir(tmp_path).iterdir()) == []


@pytest.mark.parametrize("attribute,diagnostic", [
    ('{symbol = "old_kernel", asm = "private assembly"}', "re-run lowering"),
    ('{version = 9 : i32}', "invalid versioned"),
    ('"wrong type"', "invalid versioned"),
])
def test_invalid_metadata_leaves_no_objects(tools, tmp_path, attribute, diagnostic):
    result = compile_kernels(tools, tmp_path,
                    f"module attributes {{annc.aarch64.generated_microkernel = {attribute}}} {{}}\n")
    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert not objects_dir(tmp_path).exists()


def test_parse_failure_leaves_no_objects(tools, tmp_path):
    result = compile_kernels(tools, tmp_path, "module { this is invalid }")
    assert result.returncode != 0
    assert not objects_dir(tmp_path).exists()


def test_existing_object_directory_is_rejected(tools, tmp_path):
    directory = objects_dir(tmp_path)
    directory.mkdir()
    stale = directory / "stale.o"
    stale.write_bytes(b"previous compilation")
    result = compile_kernels(tools, tmp_path, "module {}\n")
    assert result.returncode != 0
    assert "must be new" in result.stderr
    assert stale.read_bytes() == b"previous compilation"


def generated_module(
    tools, tmp_path, expect_unavailable=False, isa="neon",
    prefix_cases=False,
):
    build, env = tools
    source = tmp_path / "source.mlir"
    text = (FIXTURES / "annc-aarch64-gemm-pipeline.mlir").read_text()
    production = text.split("//--- production.mlir\n", 1)[1].split("//---", 1)[0]
    if prefix_cases:
        # Reuse the existing partial-prefix fixture with its first Add removed.
        partial = production.split(
            "\nfunc.func @add_then_m_mul(", 1
        )[1].split("\n}\n", 1)[0]
        zero = "func.func @zero_m_mul(" + partial + "\n}\n"
        begin = zero.index("  %add_tmp =")
        end = zero.index("  %mul =")
        zero = zero[:begin] + zero[end:].replace(
            "%out, %add, %scale", "%out, %matmul, %scale"
        )
        production += zero
    production = re.sub(r'(func\.func @[^{}]+\)) \{',
                        r'\1 attributes {llvm.emit_c_interface} {', production)
    source.write_text(production)
    fused = tmp_path / "fused.mlir"
    result = run([build / "bin/annc-opt", source, "--atir-gemm-epilogue-fusion",
                  "-o", fused], env)
    assert result.returncode == 0, result.stderr
    lowered = tmp_path / "lowered.mlir"
    config = FIXTURES / "Inputs" / (
        "gemm-tuning-hip09-neon-test.json" if isa == "neon"
        else "gemm-tuning-test.json"
    )
    options = f"config-path={config}"
    diagnostics = []
    if prefix_cases:
        options += " intra-thread-count=4"
        diagnostics = ["--mlir-print-ir-after=aarch64-finalize-gemm-plan"]
    result = run([build / "bin/annc-asm", fused,
                  f"--annc-aarch64-gemm-pipeline={options}", *diagnostics,
                  "-o", lowered], env)
    if prefix_cases:
        (tmp_path / "planned.mlir").write_text(result.stderr)
    if expect_unavailable:
        assert result.returncode != 0
        assert "GEMM epilogue codegen component is unavailable" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    return lowered


def test_missing_component_rejects_epilogue_without_source_fallback(tools, tmp_path):
    build, _ = tools
    cache = (build / "CMakeCache.txt").read_text()
    match = re.search(r"^ANNC_GEMM_PACKAGE_DIR:PATH=(.*)$", cache, re.MULTILINE)
    if match and (Path(match.group(1)) / "include/annc_gemm_codegen.h").is_file():
        pytest.skip("requires the separate no-codegen build configuration")
    generated_module(tools, tmp_path, expect_unavailable=True)


def memref(shape, values):
    storage = (ctypes.c_float * len(values))(*values)
    rank = len(shape)

    class Descriptor(ctypes.Structure):
        _fields_ = [("allocated", ctypes.c_void_p),
                    ("aligned", ctypes.c_void_p),
                    ("offset", ctypes.c_int64),
                    ("sizes", ctypes.c_int64 * rank),
                    ("strides", ctypes.c_int64 * rank)]

    strides = [1] * rank
    for index in range(rank - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    desc = Descriptor(ctypes.addressof(storage), ctypes.addressof(storage), 0,
                      (ctypes.c_int64 * rank)(*shape),
                      (ctypes.c_int64 * rank)(*strides))
    return storage, desc


def check_model_numerics(library):
    # Exactly representable inputs isolate GEMM/epilogue placement and operand
    # indexing, without inventing a transcendental approximation tolerance.

    for name, m, k, n, scalar in (
        ("single_tail_add_relu", 2, 5, 501, False),
        ("gemv_suffix_add_relu", 8, 641, 1, True),
    ):
        a_values = [(i % 7 - 3) * 0.125 for i in range(m * k)]
        b_values = [(i % 11 - 5) * 0.0625 for i in range(k * n)]
        bias_values = [0.375] if scalar else [(i % 9 - 4) * 0.25 for i in range(n)]
        a, da = memref((m, k), a_values)
        b, db = memref((k, n), b_values)
        output, dout = memref((m, n), [float("nan")] * (m * n))
        bias, d_bias = memref((len(bias_values),), bias_values)
        function = getattr(library, "_mlir_ciface_" + name)
        function.argtypes = [ctypes.c_void_p] * 4
        function.restype = None
        function(ctypes.byref(da), ctypes.byref(db), ctypes.byref(dout),
                 ctypes.byref(d_bias))
        expected = []
        for row in range(m):
            for column in range(n):
                value = sum(a_values[row * k + p] * b_values[p * n + column]
                            for p in range(k)) + bias_values[0 if scalar else column]
                expected.append(max(value, 0) if scalar else min(max(value, 0), 6))
        assert list(output) == expected, name


@pytest.mark.parametrize("isa", ["neon", "sve"])
def test_shared_pipeline_direct_and_prefix_numerics(tools, tmp_path, isa):
    if platform.machine() != "aarch64":
        pytest.skip("requires AArch64 execution")
    if isa == "sve" and ctypes.CDLL(None).prctl(51, 0, 0, 0, 0) != 32:
        pytest.skip("requires target SVE vector length of 32 bytes")
    build, env = tools
    lowered = generated_module(tools, tmp_path, isa=isa, prefix_cases=True)
    text = lowered.read_text()
    planned = (tmp_path / "planned.mlir").read_text()
    # Assert only public plan attributes; the normalized request is opaque.
    for name, packing in (("single_tail_add_relu", "packed"),
                          ("single_add", "direct")):
        function = planned.split(f"func.func @{name}(", 1)[1].split("func.func @", 1)[0]
        assert f'isa = "{isa}"' in function
        assert f'rhs_packing = "{packing}"' in function
    zero = text.split('func.func @zero_m_mul(', 1)[1].split('func.func @', 1)[0]
    assert "generated_microkernel" not in zero and "arith.mulf" in zero
    partial = text.split('func.func @add_then_m_mul(', 1)[1].split('func.func @', 1)[0]
    assert "generated_microkernel" in partial and "arith.mulf" in partial
    library = tmp_path / f"prefix_{isa}.so"
    result = run([build / "bin/annc", lowered, "--shared", "-o", library], env)
    assert result.returncode == 0, result.stderr
    loaded = ctypes.CDLL(str(library))
    check_model_numerics(loaded)
    for name, m, k, n, count, multiply in (
        ("single_add", 1, 5, 5, 1, False),
        ("add_then_m_mul", 3, 5, 5, 1, True),
        ("zero_m_mul", 3, 5, 5, 0, True),
    ):
        a_values = [(i % 7 - 3) * 0.125 for i in range(m * k)]
        b_values = [(i % 11 - 5) * 0.0625 for i in range(k * n)]
        bias_values = [(i - 2) * 0.25 for i in range(n)]
        scales = [-0.5, 0.25, 2.0][:m]
        inputs = [memref((m, k), a_values), memref((k, n), b_values),
                  memref((m, n), [float("nan")] * (m * n)),
                  memref((n,), bias_values)]
        if multiply:
            inputs.append(memref((m, 1), scales))
        function = getattr(loaded, "_mlir_ciface_" + name)
        function.argtypes = [ctypes.c_void_p] * len(inputs)
        function.restype = None
        function(*(ctypes.byref(descriptor) for _, descriptor in inputs))
        expected = []
        for row in range(m):
            for column in range(n):
                value = sum(a_values[row * k + p] * b_values[p * n + column]
                            for p in range(k)) + count * bias_values[column]
                expected.append(value * scales[row] if multiply else value)
        assert list(inputs[2][0]) == expected, (isa, name)


def test_objects_roundtrip_deduplicate_and_link_without_sources(tools, tmp_path):
    if platform.machine() != "aarch64":
        pytest.skip("requires AArch64 execution")
    build, env = tools
    lowered = generated_module(tools, tmp_path)
    text = lowered.read_text()
    assert "normalized_request" in text and "artifact_key" in text
    assert "asm =" not in text and ".include" not in text
    result = run([build / "bin/annc-gemm-epilogue-kernel-compile", lowered,
                  "--output-dir", tmp_path], env)
    assert result.returncode == 0, result.stderr
    objects = sorted(objects_dir(tmp_path).glob("*.o"))
    assert objects
    assert len(objects) == len({path.stem for path in objects})
    assert set(objects_dir(tmp_path).iterdir()) == set(objects)
    for path in objects:
        object_bytes = path.read_bytes()
        assert object_bytes[:4] == b"\x7fELF"
        assert object_bytes[4:7] == b"\x02\x01\x01"
        assert object_bytes[16:20] == b"\x01\x00\xb7\x00"
    assert not list(objects_dir(tmp_path).glob("*.S"))
    library = tmp_path / "model.so"
    result = run([build / "bin/annc", lowered, "--shared", "-o", library], env)
    assert result.returncode == 0, result.stderr
    loaded = ctypes.CDLL(str(library))
    check_model_numerics(loaded)
    dynamic = run(["readelf", "-d", library], env)
    assert "gemm_codegen" not in dynamic.stdout
    # A different codegen build must not silently reinterpret stored metadata.
    stale = re.sub(r'build_id = "[^"]+"', 'build_id = "stale-build"', text)
    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    result = compile_kernels(tools, stale_dir, stale)
    assert result.returncode != 0 and "build" in result.stderr.lower()
    assert not objects_dir(stale_dir).exists()
    # Conflicting saved descriptions cannot overwrite one artifact silently.
    attr = re.search(r'annc.aarch64.generated_microkernel = (\{[^}]+\})', text).group(1)
    changed = re.sub(r'normalized_request = "[0-9a-f]+"',
                     'normalized_request = "00"', attr)
    conflict_dir = tmp_path / "conflict"
    conflict_dir.mkdir()
    result = compile_kernels(tools, conflict_dir,
                    f"module {{ func.func @a() attributes {{annc.aarch64.generated_microkernel = {attr}}} {{return}}\n"
                    f"func.func @b() attributes {{annc.aarch64.generated_microkernel = {changed}}} {{return}} }}")
    assert result.returncode != 0 and "conflicting" in result.stderr
    assert not objects_dir(conflict_dir).exists()
