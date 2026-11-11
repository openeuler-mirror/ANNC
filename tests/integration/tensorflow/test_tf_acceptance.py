"""TensorFlow acceptance using the configured SDK Python and built plugins.

Reuse per-kernel graphs without importing TensorFlow during collection.
Set ANNC_REQUIRE_E2E=1 to fail on missing prerequisites instead of skipping.
"""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).with_name("tf_acceptance_runner.py")

# Case behavior is explicit; names are only pytest IDs and evidence labels.
GENERATED_GEMM = {
    "ANNC_GEMM": "1",
    "ANNC_FAST_CODEGEN_DISABLE_CUSTOM_OPS": "MatMul,MatMulAdd,MatMulAddRelu",
}
FUSION_CASES = {
    "TF-F01-"
    + variant: dict(
        builder="matmul",
        variant=variant,
        pattern="matmul",
        abi="mlir_ciface",
        mode="jit",
        env=GENERATED_GEMM,
    )
    for variant in ("plain", "dynamic")
}
FUSION_CASES.update(
    {
        "TF-F11-"
        + variant: dict(
            builder="tensordot",
            variant=variant,
            pattern="tensordot_matmul",
            abi="mlir_ciface",
            mode=mode,
            env=GENERATED_GEMM,
        )
        for variant, mode in (
            ("static", "aot"),
            ("dynamic-b", "jit"),
            ("dynamic-s", "jit"),
            ("dynamic-bs", "jit"),
            ("unfolded", "jit"),
            ("partial", "aot"),
            ("identity", "aot"),
        )
    }
)

# Import case descriptions only; existing builders import TF when executed.
sys.path.insert(0, str(ROOT / "tests/e2e/kernel_context"))
KERNEL_CASES = {}
for module_name in (
    "matmul_add_relu",
    "hash_bucket",
    "action_id_gather",
    "target_behavior_interaction",
    "sparse_dynamic_stitch",
    "sparse_segment_reduce",
    "embedding_padding",
    "sparse_reshape",
    "sparse_select",
    "kdnn",
):
    module_name = "test_tensorflow_" + module_name
    module = importlib.import_module(module_name)
    for name, spec in module.CASES.items():
        if spec.get("n", 0) > 16:
            continue
        KERNEL_CASES[name] = dict(spec, module=module_name, graph_case=name)

# TF1.15 has no native int64 SparseSegment CPU kernel. These existing inputs
# fit int32, so use the same graph with int32 indices as the numeric reference.
for combiner in ("sum", "mean"):
    KERNEL_CASES["segment_i64_" + combiner]["reference_case"] = (
        "segment_i32_" + combiner
    )


def require_environment(condition, reason):
    if not condition:
        if os.environ.get("ANNC_REQUIRE_E2E") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


@pytest.fixture(scope="module")
def sdk():
    build = Path(os.environ.get("ANNC_BUILD_DIR", ROOT / "build")).resolve()
    runtime = build / "tests/tensorflow-runtime.json"
    require_environment(runtime.is_file(), "Configure BUILD_TESTING with TF enabled")
    config = json.loads(runtime.read_text())
    require_environment(
        Path(config["python"]).is_file(), "Configure the target TF Python first"
    )
    for name in ("optimizer", "fused_op", "pipeline"):
        require_environment(
            Path(config[name]).is_file(), "Missing artifact: " + config[name]
        )
    for name in ("annc-tf2atir", "annc-opt", "annc-converter", "annc-fusion-metadata"):
        path = build / "bin" / name
        require_environment(path.is_file(), "Missing existing artifact: " + str(path))
    return build, runtime, config


def run_case(sdk, tmp_path, case, spec):
    build, runtime, config = sdk
    if spec.get("fast_codegen") == "enable-kdnn=true":
        require_environment(config["kdnn"], "Requires ANNC_ENABLE_KDNN_ADAPTOR=ON")
    tmp_path.mkdir(parents=True, exist_ok=True)
    case_config = tmp_path / "case.json"
    case_config.write_text(json.dumps(spec, indent=2))
    command = [
        config["python"],
        str(RUNNER),
        "--build-dir",
        str(build),
        "--case",
        case,
        "--case-config",
        str(case_config),
        "--runtime",
        str(runtime),
        "--work-dir",
        str(tmp_path),
    ]
    with (tmp_path / "runner.log").open("w") as log:
        result = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600
        )
    assert result.returncode == 0, (
        f"{case}: see {tmp_path / 'runner.log'}\n"
        + (tmp_path / "runner.log").read_text()[-5000:]
    )
    evidence = json.loads((tmp_path / "result.json").read_text())
    assert evidence["case"] == case and evidence["status"] == "passed"
    return evidence


@pytest.mark.parametrize("case", FUSION_CASES, ids=FUSION_CASES)
def test_fusion(sdk, tmp_path, case):
    run_case(sdk, tmp_path, case, FUSION_CASES[case])


@pytest.mark.parametrize("case", ["transpose", "wrong-shape", "external-use"])
def test_tensordot_rejects(sdk, tmp_path, case):
    spec = dict(FUSION_CASES["TF-F11-static"], variant=case, reject=True)
    run_case(sdk, tmp_path, "TF-C-reject-" + case, spec)


def test_tensordot_empty_batch(sdk, tmp_path):
    spec = dict(FUSION_CASES["TF-F11-dynamic-b"], variant="dynamic-empty")
    run_case(sdk, tmp_path, "TF-F11-dynamic-empty", spec)


@pytest.mark.parametrize("admission", ["eligible", "mismatch"])
def test_jit_prepack(sdk, tmp_path, admission):
    spec = dict(
        handler="prepack",
        admission=admission,
        env=dict(GENERATED_GEMM, ANNC_GEMM="3"),
    )
    run_case(sdk, tmp_path, "TF-JIT-prepack-" + admission, spec)


@pytest.mark.parametrize("case", sorted(KERNEL_CASES))
def test_existing_kernel(sdk, tmp_path, case):
    run_case(sdk, tmp_path, case, KERNEL_CASES[case])


def test_configured_gather(sdk, tmp_path):
    run_case(
        sdk,
        tmp_path,
        "configured_gather",
        dict(
            module="test_tensorflow_gather",
            graph_case="gather",
            pattern="kp_fused_gather_external",
            abi="annc_context_v3",
            mode="aot",
            pipeline_args=[
                "--fusion-config",
                str(
                    ROOT
                    / "tests/FileCheck/Dialect/Atir/config-fusion-kp-fused-gather.json"
                ),
            ],
            output_slots=[None, None, 0, 1, 2],
            exact=True,
        ),
    )


def test_gemm_epilogue(sdk, tmp_path):
    run_case(
        sdk,
        tmp_path,
        "gemm_epilogue",
        dict(
            builder="epilogue",
            pattern="matmul_add_mul_relu",
            abi="mlir_ciface",
            mode="jit",
            env=dict(GENERATED_GEMM, ANNC_GEMM="2"),
        ),
    )


@pytest.mark.parametrize("enabled", [False, True], ids=["ordinary", "packed"])
def test_aot_prepack(sdk, tmp_path, enabled):
    spec = dict(
        FUSION_CASES["TF-F11-static"],
        handler="aot_prepack",
        packed=enabled,
        env=dict(GENERATED_GEMM, ANNC_GEMM="3" if enabled else "1"),
    )
    run_case(sdk, tmp_path, "aot_prepack", spec)


def test_pipeline_failure_falls_back(sdk, tmp_path):
    spec = dict(FUSION_CASES["TF-F11-static"], fallback=True)
    run_case(sdk, tmp_path, "pipeline_failure", spec)
