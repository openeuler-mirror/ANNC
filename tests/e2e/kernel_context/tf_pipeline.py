"""Production kernel specializations through real TF matching and execution.

Run on AArch64 Linux with ANNC_REQUIRE_E2E=1 to make missing prerequisites
fail instead of skip. Each case retains its graph, IR, metadata and logs in
pytest's --basetemp directory. No ANNCFused nodes or ABI wrappers are mocked.
"""

import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys

from e2e_utils import ROOT, _require, _run


def run_pipeline(spec, make_graph, case, tmp_path):
    _require(
        platform.system() == "Linux"
        and platform.machine().lower() in ("aarch64", "arm64"),
        "Production kernel E2E requires AArch64 Linux",
    )
    _require(importlib.util.find_spec("tensorflow"), "TensorFlow is required")
    build = Path(os.environ.get("ANNC_BUILD_DIR", ROOT / "build")).resolve()
    tools = {
        name: build / "bin" / name
        for name in (
            "annc-tf2atir",
            "annc-opt",
            "annc-fusion-metadata",
            "annc-asm",
            "annc",
            "annc-converter",
        )
    }
    for name, path in tools.items():
        _require(path.is_file(), f"{name} is not built: {path}")
    addon = Path(
        os.environ.get("ANNC_FUSED_OP_PATH", build / "lib/libannc_fused_op.so")
    ).resolve()
    _require(addon.is_file(), f"TensorFlow addon is not built: {addon}")
    env = os.environ.copy()
    if spec["abi"] == "mlir_ciface":
        config = env.get("ANNC_GEMM_CONFIG")
        _require(
            config and Path(config).is_file(),
            "Generated GEMM requires ANNC_GEMM_CONFIG pointing to a tuning JSON",
        )
    kernel_lib = env.get("ANNC_KERNEL_LIB_PATH", os.fspath(build / "lib"))
    env.update(
        {
            "ANNC_KERNEL_LIB_PATH": kernel_lib,
            "ANNC_PIPELINE_PATH": os.fspath(build / "bin/annc-tf-pipeline"),
            "ANNC_VERBOSE": "1",
            "ANNC_FUSED_PROFILE": "1",
            "ANNC_FUSED_PROFILE_INTERVAL": "1",
            "ANNC_WORK_DIR": os.fspath(tmp_path / "jit-work"),
            "ANNC_KEEP_TEMPS": "1",
        }
    )
    for variable, paths in {
        "PATH": [build / "bin", build / "_deps/llvm-build/bin"],
        "LD_LIBRARY_PATH": [
            build / "lib",
            build / "_deps/llvm-build/lib",
            Path(kernel_lib),
        ],
        "LIBRARY_PATH": [Path(kernel_lib), build / "_deps/llvm-build/lib"],
    }.items():
        env[variable] = os.pathsep.join(
            [
                *(os.fspath(path) for path in paths),
                env.get(variable, ""),
            ]
        )
    graph, outputs, _, _ = make_graph(case)
    fixture = tmp_path / "original.pb"
    fixture.write_bytes(graph.SerializeToString())
    raw, fused, metadata, aot, lowered, shared, rewritten = [
        tmp_path / name
        for name in (
            "raw.mlir",
            "fused.mlir",
            "metadata.json",
            "aot.mlir",
            "lowered.mlir",
            "kernel.so",
            "rewritten.pb",
        )
    ]
    frontend = [tools["annc-tf2atir"], fixture, "-o", raw]
    for output in outputs:
        frontend.extend(["--output_tensor", output])
    _run(frontend, env, tmp_path / "frontend.log")
    _run(
        [
            tools["annc-opt"],
            raw,
            "--atir-identity-canonicalize",
            "--atir-op-fusion",
            "-o",
            fused,
        ],
        env,
        tmp_path / "fusion.log",
    )
    text = fused.read_text()
    assert f'fusion.pattern = "{spec["pattern"]}"' in text
    assert f'abi = "{spec["abi"]}"' in text
    if spec["abi"] == "annc_context_v3":
        assert f'custom.op_name = "{spec["op"]}"' in text
    else:
        assert '"atir.MatMul"' in text
        assert '"atir.Customize"' not in text
    _run(
        [tools["annc-fusion-metadata"], fused, "-o", metadata],
        env,
        tmp_path / "metadata.log",
    )
    assert spec["pattern"] in metadata.read_text()
    if spec["mode"] == "aot":
        _run(
            [
                tools["annc-opt"],
                fused,
                "--atir-prune-func=execution-mode=aot",
                "-o",
                aot,
            ],
            env,
            tmp_path / "prune.log",
        )
        config = env.get("ANNC_GEMM_CONFIG")
        gemm = "--annc-aarch64-gemm-pipeline"
        if config:
            gemm += f"=config-path={config}"
        _run(
            [tools["annc-asm"], aot, "--atir-fast-codegen", gemm, "-o", lowered],
            env,
            tmp_path / "asm.log",
        )
        assert (
            spec["symbol"] in lowered.read_text()
        ), f"Expected production specialization {spec['symbol']} was not emitted"
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
    rewrite = [
        tools["annc-converter"],
        fused,
        "--tf-graphdef-rewrite",
        "--input_graphdef",
        fixture,
        "--output_graphdef",
        rewritten,
        "--atir_module_path",
        fused,
    ]
    if spec["mode"] == "aot":
        rewrite.extend(["--shared_lib_path", shared])
    _run(rewrite, env, tmp_path / "rewrite.log")
    report = tmp_path / "execution.json"
    execution = _run(
        [
            sys.executable,
            Path(__file__).resolve(),
            "--worker",
            sys.modules[make_graph.__module__].__file__,
            case,
            rewritten,
            addon,
            report,
        ],
        env,
        tmp_path / "execution.log",
        timeout=600,
    )
    result = json.loads(report.read_text())
    if spec["mode"] == "jit":
        compiled_ir = list((tmp_path / "jit-work").glob("*/kernel_lowered.mlir"))
        assert compiled_ir, "Runtime did not compile the JIT kernel"
        for path in compiled_ir:
            lowered_text = path.read_text()
            assert re.search(
                rf'func\.func\s+(?:private\s+)?@{re.escape(result["kernel_name"])}\(',
                lowered_text,
            ), "JIT did not emit the rewritten graph's kernel entry"
            if spec["symbol"] is not None:
                assert spec["symbol"] in lowered_text
            else:
                assert "ANNCKernel_ContextV3_MatMul_float_auto" not in lowered_text
    assert result["abi"] == spec["abi"]
    assert all(item["fused_executed"] for item in result["scenarios"])
    assert f'pattern={spec["pattern"]}' in execution.stderr


def _worker(module_path, case, rewritten_path, addon_path, report_path):
    module_spec = importlib.util.spec_from_file_location("kernel_case", module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec = module.CASES[case]
    make_graph = module.make_graph
    import numpy as np
    import tensorflow as tf

    original, _, outputs, scenarios = make_graph(case)
    rewritten = tf.compat.v1.GraphDef()
    rewritten.ParseFromString(Path(rewritten_path).read_bytes())
    nodes = [node for node in rewritten.node if node.op == "ANNCFused"]
    assert len(nodes) == 1, [(node.name, node.op) for node in nodes]
    fused = nodes[0]
    assert not any(key.endswith("_names") for key in fused.attr)
    assert fused.attr["abi"].s.decode() == spec["abi"]
    kernel_name = fused.attr["kernel_name"].s.decode()
    assert kernel_name
    assert fused.attr["fusion_pattern"].s.decode() == spec["pattern"]
    assert fused.attr["num_outputs"].i == len(outputs)
    slots = {label: f"{fused.name}:{i}" for i, (_, label) in enumerate(outputs)}

    def import_graph(definition):
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(definition, name="")
        return graph

    baseline = import_graph(original)
    tf.load_op_library(os.fspath(addon_path))
    compiled = import_graph(rewritten)
    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=4, inter_op_parallelism_threads=1
    )
    config.graph_options.optimizer_options.opt_level = -1
    config.graph_options.rewrite_options.disable_meta_optimizer = True
    results = []
    with tf.compat.v1.Session(
        graph=baseline, config=config
    ) as reference, tf.compat.v1.Session(graph=compiled, config=config) as actual:
        for label, feeds in scenarios:
            expected = reference.run([name for name, _ in outputs], feed_dict=feeds)
            # Rewritten graphs remove placeholders internal to the fusion.
            # Feed only the boundary placeholders retained by the converter.
            retained = {
                name: value
                for name, value in feeds.items()
                if name.split(":")[0] in {node.name for node in rewritten.node}
            }
            metadata = tf.compat.v1.RunMetadata()
            got = actual.run(
                [slots[slot] for _, slot in outputs],
                feed_dict=retained,
                run_metadata=metadata,
                options=tf.compat.v1.RunOptions(
                    trace_level=tf.compat.v1.RunOptions.FULL_TRACE
                ),
            )
            for (name, _), value, truth in zip(outputs, got, expected):
                assert (value.dtype, value.shape) == (truth.dtype, truth.shape), name
                if np.issubdtype(truth.dtype, np.floating):
                    np.testing.assert_allclose(
                        value, truth, rtol=1e-5, atol=1e-6, err_msg=name
                    )
                else:
                    np.testing.assert_array_equal(value, truth, err_msg=name)
            executed = {
                step.node_name
                for device in metadata.step_stats.dev_stats
                for step in device.node_stats
            }
            assert fused.name in executed, sorted(executed)
            results.append(
                {
                    "name": label,
                    "fused_executed": True,
                    "compared_outputs": len(outputs),
                }
            )
    Path(report_path).write_text(
        json.dumps(
            {
                "case": case,
                "abi": fused.attr["abi"].s.decode(),
                "kernel_name": kernel_name,
                "scenarios": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    assert len(sys.argv) == 7 and sys.argv[1] == "--worker"
    _worker(*sys.argv[2:])
