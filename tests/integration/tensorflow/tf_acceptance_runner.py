#!/usr/bin/env python3
"""Run one acceptance graph with the TF1.15 SDK Python (Python 3.7 supported)."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[3]


def build_matmul_case(variant):
    if variant not in ("plain", "dynamic"):
        raise ValueError("Unknown MatMul case: " + variant)
    graph = tf.Graph()
    m, k, n = 3, 8, 4
    dynamic = variant == "dynamic"
    with graph.as_default():
        x = tf.placeholder(tf.float32, [None if dynamic else m, k], name="input")
        w = tf.constant(np.arange(k * n).reshape(k, n) / 32, tf.float32, name="weight")
        y = tf.matmul(x, w, name="matmul")
        members = ["matmul"]
        outputs = [y.op.name]
    feeds = [
        {"input": (np.arange(rows * k, dtype=np.float32).reshape(rows, k) - 12) / 16}
        for rows in ([m, m + 2, m] if dynamic else [m])
    ]
    return graph.as_graph_def(add_shapes=True), feeds, outputs, members


def build_tensordot_case(variant):
    b, s, k, n = 2, 3, 8, 4
    graph = tf.Graph()
    with graph.as_default():
        input_shape = [
            None
            if variant in ("dynamic-b", "dynamic-bs", "dynamic-empty", "unfolded")
            else b,
            None if variant in ("dynamic-s", "dynamic-bs", "unfolded") else s,
            k,
        ]
        x = tf.placeholder(tf.float32, input_shape, name="input")
        w = tf.constant(
            np.arange(k * n).reshape(k, n) / (k * n), tf.float32, name="weight"
        )
        source = x
        members = []
        if variant in ("identity", "transpose"):
            source = tf.transpose(
                x, [0, 1, 2] if variant == "identity" else [1, 0, 2], name="transpose"
            )
            members.append("transpose")
        if variant.startswith("dynamic") or variant == "static":
            y = tf.tensordot(source, w, axes=[[2], [0]], name="tensordot")
            members += ["tensordot/Reshape", "tensordot/MatMul", "tensordot"]
        else:
            dims = tf.shape(x, name="shape")
            leading = tf.gather(dims, [0, 1], name="leading")
            contraction = tf.gather(dims, [2], name="contraction")
            flat_shape = tf.stack(
                [tf.reduce_prod(leading), tf.reduce_prod(contraction)]
            )
            flat = tf.reshape(
                source, [-1, k] if variant == "partial" else flat_shape, name="flatten"
            )
            mm = tf.matmul(flat, w, name="matmul")
            restored = tf.concat([leading, [n]], 0)
            if variant == "wrong-shape":
                restored = tf.constant([s, b, n])
            y = tf.reshape(mm, restored, name="output")
            members += ["flatten", "matmul", "output"]
        outputs = [y.op.name]
        if variant == "external-use":
            outputs.append(tf.reduce_sum(flat, name="external").op.name)
    sizes = [(b, s)]
    if variant == "dynamic-empty":
        sizes += [(0, s), (b, s)]
    elif variant == "dynamic-b":
        sizes += [(b + 1, s), (b, s)]
    elif variant == "dynamic-s":
        sizes += [(b, s + 1), (b, s)]
    elif variant in ("dynamic-bs", "unfolded"):
        sizes += [(b + 1, s + 1), (b, s)]
    feeds = [
        {
            "input": (
                np.arange(rows * seq * k, dtype=np.float32).reshape(rows, seq, k) % 17
                - 8
            )
            / 16
        }
        for rows, seq in sizes
    ]
    return graph.as_graph_def(add_shapes=True), feeds, outputs, members


def build_case(case):
    family, variant = case.split("-", 2)[1:]
    if family == "F01":
        data = build_matmul_case(variant)
        return data + ("matmul", False)
    if family == "F11":
        return build_tensordot_case(variant) + ("tensordot_matmul", False)
    raise ValueError(case)


def compare_outputs(expected, actual, exact):
    assert len(expected) == len(actual), "Output count changed"
    for index, (ref, value) in enumerate(zip(expected, actual)):
        assert ref.dtype == value.dtype, "Output {} dtype changed".format(index)
        assert ref.shape == value.shape, "Output {} shape: TF {}, ANNC {}".format(
            index, ref.shape, value.shape
        )
        assert np.isfinite(ref).all() and np.isfinite(value).all()
        if exact or ref.dtype.kind not in "fc":
            np.testing.assert_array_equal(value, ref)
        else:
            error = np.abs(value - ref)
            bound = 1e-5 * np.maximum(1, np.abs(ref))
            assert np.all(error <= bound), "Output {} max error {}".format(
                index, error.max()
            )


def open_session(graph_def, enabled, pipeline=None):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    # TF1.15 import_graph_def does not retain _output_shapes on imported nodes.
    # Restore TF-inferred annotations required by the ANNC frontend.
    for node in graph.as_graph_def(add_shapes=True).node:
        graph.get_operation_by_name(node.name)._set_attr(
            "_output_shapes", node.attr["_output_shapes"]
        )
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        device_count={"GPU": 0},
    )
    config.graph_options.optimizer_options.opt_level = -1
    rewrite = config.graph_options.rewrite_options
    rewrite.min_graph_nodes = -1
    rewrite.meta_optimizer_iterations = rewrite.ONE
    if enabled:
        rewrite.optimizers.append("ANNCOptimizer")
        optimizer = rewrite.custom_optimizers.add()
        optimizer.name = "ANNCOptimizer"
        optimizer.parameter_map["enabled"].s = b"true"
        optimizer.parameter_map["pipeline_path"].s = str(pipeline).encode()
    else:
        rewrite.disable_meta_optimizer = True
    return tf.Session(graph=graph, config=config)


def execute(session, feeds, outputs, metadata=None):
    def tensor(name):
        return session.graph.get_tensor_by_name(name if ":" in name else name + ":0")

    options = None
    if metadata is not None:
        options = tf.RunOptions(
            output_partition_graphs=True, trace_level=tf.RunOptions.FULL_TRACE
        )
    return session.run(
        [tensor(name) for name in outputs],
        feed_dict={tensor(name): value for name, value in feeds.items()},
        options=options,
        run_metadata=metadata,
    )


def fusion_evidence(metadata, pattern, members, outputs, work, mode, reject=False):
    nodes = {n.name: n for p in metadata.partition_graphs for n in p.node}
    executed = {
        n.node_name for d in metadata.step_stats.dev_stats for n in d.node_stats
    }
    fused = [n for n in nodes.values() if n.op == "ANNCFused"]
    matching = [n for n in fused if n.attr["fusion_pattern"].s.decode() == pattern]
    # Match the pipeline's exported metadata by kernel identity.
    records = (work / "fusion_metadata.jsonl").read_text().splitlines()
    assert records, "Missing pipeline fusion metadata"
    entries = [entry for record in records for entry in json.loads(record)["fusions"]]
    if reject:
        assert not matching and not any(
            entry["pattern"] == pattern for entry in entries
        ), "Invalid whole tensordot fusion"
        return []
    assert len(matching) == 1, "Expected one {} fusion, got {}".format(
        pattern, len(matching)
    )
    target = matching[0]
    assert target.name in executed, "Fused node was not executed"

    def uses_target_data(name, seen):
        name = name.split(":", 1)[0]
        if name == target.name:
            return True
        if name in seen or name not in nodes:
            return False
        seen.add(name)
        return any(
            uses_target_data(value, seen)
            for value in nodes[name].input
            if not value.startswith("^")
        )

    # _Retval indices follow the fetch order passed to Session.run.
    fetches = {n.attr["index"].i: n.name for n in nodes.values() if n.op == "_Retval"}
    for index, output in enumerate(outputs):
        root = fetches[index] if fetches else output
        assert uses_target_data(root, set()), "Fusion does not supply " + output
    kernel = target.attr["kernel_name"].s.decode()
    selected = [entry for entry in entries if entry["kernel_name"] == kernel]
    assert selected, "Cannot identify kernel metadata: " + kernel
    assert all(entry == selected[0] for entry in selected), "Ambiguous kernel metadata"
    entry = selected[0]
    assert entry["pattern"] == pattern, "Unexpected fusion metadata pattern"
    names = {str(name).split(":")[0] for name in entry["member_node_names"]}
    assert entry["abi"] == target.attr["abi"].s.decode() == "mlir_ciface"
    assert entry["execution_mode"] == mode
    assert bool(target.attr["shared_lib_path"].s) == (mode == "aot")
    assert set(members) <= names, "Missing fused members: " + str(set(members) - names)
    if pattern == "tensordot_matmul":
        assert {a["tf_name"] for a in selected[0]["args"]} == {
            "weight",
            "input",
        }, "Wrong tensordot input boundary"
        assert (
            len(target.input) == 2 and target.input[0].split(":")[0] == "weight"
        ), "Wrong TF fixed/dynamic input order"
    return [
        {
            "node": target.name,
            "kernel": kernel,
            "pattern": pattern,
            "members": sorted(names),
            "metadata": selected[0],
        }
    ]


def run_fusion(args, data, reject=False):
    graph, feeds, outputs, members, pattern, exact = data
    mode = (
        "jit"
        if (
            pattern == "matmul"
            or args.case.startswith("TF-F11-dynamic")
            or args.case == "TF-F11-unfolded"
        )
        else "aot"
    )
    work = args.work_dir
    (work / "input.pb").write_bytes(graph.SerializeToString())
    (work / "fusion_metadata.jsonl").write_text("")
    pipeline = work / "capture_pipeline.py"
    # Request the documented JSON output without depending on internal IR paths.
    pipeline.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            f"""\
            import json
            from pathlib import Path
            import subprocess
            import sys

            command = [{str(args.build_dir / 'bin/annc-tf-pipeline')!r}]
            result = subprocess.run(command + sys.argv[1:] + ["--dump-fusion-metadata"])
            if result.returncode == 0:
                directory = Path(sys.argv[sys.argv.index("--work_dir") + 1])
                metadata = json.loads((directory / "fusion_metadata.json").read_text())
                with Path({str(work / 'fusion_metadata.jsonl')!r}).open("a") as output:
                    output.write(json.dumps(metadata) + "\\n")
            sys.exit(result.returncode)
            """
        )
    )
    pipeline.chmod(0o755)
    report = {"case": args.case, "outputs": list(outputs), "feeds": [], "fusions": []}
    with open_session(graph, False) as reference, open_session(
        graph, True, pipeline
    ) as optimized:
        for index, feed in enumerate(feeds):
            expected = execute(reference, feed, outputs)
            metadata = tf.RunMetadata()
            actual = execute(optimized, feed, outputs, metadata)
            (work / "run_metadata-{}.pb".format(index)).write_bytes(
                metadata.SerializeToString()
            )
            np.savez(
                str(work / "outputs-{}.npz".format(index)),
                **dict(
                    [("reference_{}".format(i), v) for i, v in enumerate(expected)]
                    + [("actual_{}".format(i), v) for i, v in enumerate(actual)]
                ),
            )
            report["feeds"].append(
                {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in feed.items()
                }
            )
            report["fusions"].append(
                fusion_evidence(metadata, pattern, members, outputs, work, mode, reject)
            )
            compare_outputs(expected, actual, exact)
            compare_outputs(expected, execute(optimized, feed, outputs), exact)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.build_dir = args.build_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    # Import only in the target SDK process; pytest collection never imports TF.
    global np, tf
    import numpy as np
    import tensorflow as tf

    assert tf.__version__.startswith("1.15."), "Requires the configured TF1.15 SDK"
    tf.disable_eager_execution()
    build = args.build_dir
    os.environ.update(
        {
            "ANNC_PIPELINE_PATH": str(build / "bin/annc-tf-pipeline"),
            "ANNC_FUSED_OP_PATH": str(
                build / "lib/tensorflow/1.15/libannc_fused_op.so"
            ),
            "ANNC_WORK_DIR": str(args.work_dir / "pipeline"),
            "ANNC_KEEP_TEMPS": "1",
            "ANNC_GEMM": "1",
            "ANNC_VERBOSE": "1",
            "ANNC_FUSED_PROFILE": "0",
        }
    )
    os.environ["PATH"] = (
        str(build / "bin")
        + ":"
        + str(build / "_deps/llvm-build/bin")
        + ":"
        + os.environ["PATH"]
    )
    sdk = Path(tf.sysconfig.get_lib())
    ctypes.CDLL(str(sdk / "libtensorflow_framework.so.1"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(
        str(sdk / "python/_pywrap_tensorflow_internal.so"), mode=ctypes.RTLD_GLOBAL
    )
    tf.load_library(str(build / "lib/tensorflow/1.15/libannc_optimizer.so"))
    tf.load_op_library(os.environ["ANNC_FUSED_OP_PATH"])
    if args.case.startswith("TF-F"):
        data = build_case(args.case)
        report = run_fusion(args, data)
    elif args.case.startswith("TF-C-reject-"):
        data = build_tensordot_case(args.case[len("TF-C-reject-") :]) + (
            "tensordot_matmul",
            False,
        )
        report = run_fusion(args, data, reject=True)
    else:
        raise ValueError("Unknown acceptance case: " + args.case)
    config = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if (
            line.startswith(
                (
                    "ANNC_TENSORFLOW_",
                    "ANNC_ENABLE_CONSTANT_FOLDING:",
                    "ANNC_KDNN_SOURCE:",
                    "CMAKE_BUILD_TYPE:",
                )
            )
            and "=" in line
        ):
            key, value = line.split("=", 1)
            config[key] = value
    report.update(
        case=args.case,
        status="passed",
        tensorflow=tf.__version__,
        build=str(build),
        configuration=config,
    )
    (args.work_dir / "result.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
