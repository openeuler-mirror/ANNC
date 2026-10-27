"""GraphDef -> configured fusion -> AArch64 compile -> ANNCFused regression."""

import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import pytest

from e2e_utils import ROOT, _require, _run


FIXTURE = ROOT / "tests/fixtures/kp_fused_gather_match.pbtxt"
CONFIG = ROOT / "tests/FileCheck/Dialect/Atir/config-fusion-kp-fused-gather.json"
OUTPUTS = ("restore_first_unique:0", "post_outer_gather:0")
FUSED_OUTPUTS = ("first_unique:0", "first_unique:1", "outer_gather:0")
PRODUCTION_SYMBOL = (
    "ANNCKernel_ContextV3_KPFusedGather_float_int64_t_int32_t_int32_t_auto"
)


def test_gather_graphdef_fusion_compile_and_execute(tmp_path):
    """Match a TF GraphDef; don't hand-write ANNCFused or mock the kernel."""
    _require(
        platform.system() == "Linux"
        and platform.machine().lower() in ("aarch64", "arm64"),
        "Gather production E2E requires AArch64 Linux",
    )
    _require(importlib.util.find_spec("tensorflow"), "TensorFlow is required")
    build = Path(os.environ.get("ANNC_BUILD_DIR", ROOT / "build")).resolve()
    names = (
        "annc-tf2atir",
        "annc-opt",
        "annc-fusion-metadata",
        "annc-asm",
        "annc",
        "annc-converter",
    )
    tools = {name: build / "bin" / name for name in names}
    for name, path in tools.items():
        _require(path.is_file(), f"{name} is not built: {path}")
    configured_addon = os.environ.get("ANNC_FUSED_OP_PATH")
    candidates = (
        [Path(configured_addon)]
        if configured_addon
        else [
            build / "lib/libannc_fused_op.so",
            build / "tensorflow_addons/libannc_fused_op.so",
        ]
    )
    addon = next((path.resolve() for path in candidates if path.is_file()), None)
    _require(addon is not None, "ANNCFused TensorFlow addon is not built")

    work = tmp_path / "pipeline"
    work.mkdir()
    env = os.environ.copy()
    kernel_lib = env.get("ANNC_KERNEL_LIB_PATH", os.fspath(build / "lib"))
    env.update(
        {
            "ANNC_KERNEL_LIB_PATH": kernel_lib,
            "ANNC_VERBOSE": "1",
            "ANNC_FUSED_PROFILE": "1",
            "ANNC_FUSED_PROFILE_INTERVAL": "1",
            "ANNC_ENABLE_GEMM_EPILOGUE": "0",
        }
    )
    env["PATH"] = os.pathsep.join(
        [
            os.fspath(build / "bin"),
            os.fspath(build / "_deps/llvm-build/bin"),
            env.get("PATH", ""),
        ]
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [
            os.fspath(build / "lib"),
            env.get("LD_LIBRARY_PATH", ""),
        ]
    )

    raw = work / "model_raw_atir.mlir"
    fused = work / "model_fused_atir.mlir"
    metadata = work / "fusion_metadata.json"
    pruned = work / "model_kernel_atir.mlir"
    lowered = work / "model_lowered.mlir"
    shared = work / "annc_generated_kernel.so"
    rewritten = work / "rewritten.pb"

    frontend = [tools["annc-tf2atir"], FIXTURE]
    for output in OUTPUTS:
        frontend.extend(["--output_tensor", output])
    frontend.extend(["-o", raw])
    _run(frontend, env, tmp_path / "tf2atir.log")

    # Run the same matcher used by the existing Gather config test. Rank
    # inference must not mark these source ops before config fusion sees them.
    _run(
        [
            tools["annc-opt"],
            raw,
            "--atir-identity-canonicalize",
            f"--atir-config-fusion=config={CONFIG} run-builtin-after=false",
            "-o",
            fused,
        ],
        env,
        tmp_path / "fusion.log",
    )
    fused_text = fused.read_text()
    assert 'abi = "annc_context_v3"' in fused_text
    assert 'fusion.pattern = "kp_fused_gather_external"' in fused_text
    assert "atir.Unique" in fused_text
    assert "atir.Gather" in fused_text

    _run(
        [
            tools["annc-fusion-metadata"],
            fused,
            "-o",
            metadata,
        ],
        env,
        tmp_path / "metadata.log",
    )
    metadata_value = json.loads(metadata.read_text())
    assert "kp_fused_gather_external" in json.dumps(metadata_value)

    # The config fusion function has no explicit execution-mode attr. Prune
    # only the non-kernel main function; filtering for "aot" would drop it.
    _run(
        [
            tools["annc-opt"],
            fused,
            "--atir-prune-func",
            "-o",
            pruned,
        ],
        env,
        tmp_path / "prune.log",
    )
    assert 'fusion.pattern = "kp_fused_gather_external"' in pruned.read_text()
    _run(
        [
            tools["annc-asm"],
            pruned,
            "--atir-fast-codegen",
            "--annc-aarch64-gemm-pipeline",
            "-o",
            lowered,
        ],
        env,
        tmp_path / "asm.log",
    )
    lowered_text = lowered.read_text()
    assert PRODUCTION_SYMBOL in lowered_text
    _run(
        [
            tools["annc"],
            lowered,
            "--shared",
            "--kernel-lib-path",
            kernel_lib,
            "-o",
            shared,
        ],
        env,
        tmp_path / "link.log",
    )
    assert shared.is_file() and shared.stat().st_size > 0

    _run(
        [
            tools["annc-converter"],
            fused,
            "--tf-graphdef-rewrite",
            "--input_graphdef",
            FIXTURE,
            "--output_graphdef",
            rewritten,
            "--shared_lib_path",
            shared,
            "--atir_module_path",
            fused,
        ],
        env,
        tmp_path / "rewrite.log",
    )
    assert rewritten.is_file()

    report_path = tmp_path / "execution.json"
    execution = _run(
        [
            sys.executable,
            Path(__file__).resolve(),
            "--worker",
            rewritten,
            addon,
            report_path,
        ],
        env,
        tmp_path / "execution.log",
    )
    report = json.loads(report_path.read_text())
    assert report["cases"] == [
        "duplicates",
        "single_unique",
        "different_column",
        "repeat",
    ]
    assert report["compared_tensors_per_case"] == 5
    assert report["invalid_key_rejected"]
    assert report["executed_fused_node"] == report["fused_node"]
    assert "[ANNC-FUSED-PROFILE]" in execution.stderr
    assert "pattern=kp_fused_gather_external" in execution.stderr


def _worker(rewritten_path, addon_path, report_path):
    """Run in a subprocess to isolate TensorFlow registration and caches."""
    import numpy as np
    import tensorflow as tf
    from google.protobuf import text_format

    original_def = tf.compat.v1.GraphDef()
    text_format.Parse(FIXTURE.read_text(), original_def)
    rewritten_def = tf.compat.v1.GraphDef()
    rewritten_def.ParseFromString(Path(rewritten_path).read_bytes())
    nodes = {node.name: node for node in rewritten_def.node}
    fused_nodes = [node for node in rewritten_def.node if node.op == "ANNCFused"]
    assert len(fused_nodes) == 1, "the Gather pattern must match exactly once"
    fused = fused_nodes[0]
    assert not any(key.endswith("_names") for key in fused.attr)
    assert fused.attr["abi"].s == b"annc_context_v3"
    assert fused.attr["fusion_pattern"].s == b"kp_fused_gather_external"
    shared_lib = fused.attr["shared_lib_path"].s.decode()
    assert shared_lib and Path(shared_lib).is_file()
    assert not fused.attr["atir_module_path"].s
    assert {name.split(":")[0] for name in fused.input} == {
        "data",
        "keys",
        "begin",
    }
    slots = {
        name: f"{fused.name}:{slot}"
        for slot, name in enumerate(("unique_values", "unique_indices", "gathered"))
    }
    assert nodes["restore_first_unique"].input[:2] == [
        slots["unique_values"],
        slots["unique_indices"],
    ]
    assert nodes["post_outer_gather"].input[0] == slots["gathered"]

    def import_graph(graph_def):
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
        return graph

    baseline = import_graph(original_def)
    tf.load_op_library(os.fspath(addon_path))
    compiled = import_graph(rewritten_def)
    session_config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=4,
        inter_op_parallelism_threads=1,
    )
    session_config.graph_options.optimizer_options.opt_level = -1
    session_config.graph_options.rewrite_options.disable_meta_optimizer = True

    data = np.array([[-3, 2, -1, 4], [5, -6, 7, -8]], dtype=np.float32)
    keys = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    cases = [
        ("duplicates", 0),
        ("single_unique", 1),
        ("different_column", 2),
        ("repeat", 0),
    ]
    names = [*OUTPUTS, *FUSED_OUTPUTS]
    rewritten_names = [
        *OUTPUTS,
        slots["unique_values"],
        slots["unique_indices"],
        slots["gathered"],
    ]
    executed_nodes = set()
    with tf.compat.v1.Session(
        graph=baseline, config=session_config
    ) as reference, tf.compat.v1.Session(
        graph=compiled, config=session_config
    ) as actual:
        for label, column in cases:
            feeds = {
                "data:0": data,
                "keys:0": keys,
                "begin:0": np.array([0, column], dtype=np.int32),
            }
            expected = reference.run(names, feed_dict=feeds)
            run_metadata = tf.compat.v1.RunMetadata()
            got = actual.run(
                rewritten_names,
                feed_dict=feeds,
                run_metadata=run_metadata,
                options=tf.compat.v1.RunOptions(
                    trace_level=tf.compat.v1.RunOptions.FULL_TRACE
                ),
            )
            for tensor_name, actual_value, expected_value in zip(names, got, expected):
                assert actual_value.dtype == expected_value.dtype, (
                    label,
                    tensor_name,
                    actual_value.dtype,
                    expected_value.dtype,
                )
                assert actual_value.shape == expected_value.shape, (
                    label,
                    tensor_name,
                    actual_value.shape,
                    expected_value.shape,
                )
                np.testing.assert_array_equal(
                    actual_value,
                    expected_value,
                    err_msg=f"{label}: {tensor_name}",
                )
            step_nodes = {
                step.node_name
                for device in run_metadata.step_stats.dev_stats
                for step in device.node_stats
            }
            assert fused.name in step_nodes, (label, sorted(step_nodes))
            executed_nodes.update(step_nodes)

        bad_keys = keys.copy()
        bad_keys[0, 0] = data.shape[0]
        bad_feed = {
            "data:0": data,
            "keys:0": bad_keys,
            "begin:0": np.array([0, 0], dtype=np.int32),
        }
        with pytest.raises(tf.errors.InvalidArgumentError):
            reference.run(list(OUTPUTS), feed_dict=bad_feed)
        with pytest.raises(tf.errors.InternalError, match="idx out of table range"):
            actual.run(list(OUTPUTS), feed_dict=bad_feed)

    Path(report_path).write_text(
        json.dumps(
            {
                "cases": [label for label, _ in cases],
                "compared_tensors_per_case": len(names),
                "fused_node": fused.name,
                "executed_fused_node": (
                    fused.name if fused.name in executed_nodes else None
                ),
                "invalid_key_rejected": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    assert len(sys.argv) == 5 and sys.argv[1] == "--worker"
    _worker(*sys.argv[2:])
