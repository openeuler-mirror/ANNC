#!/usr/bin/env python3
"""Run TensorFlow-to-ANNC E2E checks for all KP embedding patterns.

Each case builds a small graph, runs the external fusion pipeline, loads the
rewritten GraphDef, and compares its outputs with the unfused TensorFlow graph.
Use ``--case all`` (the default) to run every graph in this file.
"""

import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "fusion" / "kp_embedding_patterns.json"


def _first_file(candidates, description):
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = "\n  ".join(os.fspath(path) for path in candidates)
    raise FileNotFoundError(f"cannot find {description}; checked:\n  {rendered}")


def _default_pipeline():
    configured = os.environ.get("ANNC_PIPELINE_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            ROOT / "install" / "bin" / "annc-tf-pipeline",
            ROOT / "build" / "bin" / "annc-tf-pipeline",
            ROOT / "build-v2" / "bin" / "annc-tf-pipeline",
        ]
    )
    return _first_file(candidates, "annc-tf-pipeline")


def _default_fused_op():
    configured = os.environ.get("ANNC_FUSED_OP_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            ROOT / "install" / "lib" / "libannc_fused_op.so",
            ROOT / "build" / "tensorflow_addons" / "libannc_fused_op.so",
            ROOT / "build-v2" / "tensorflow_addons" / "libannc_fused_op.so",
            ROOT / "tensorflow_addons" / "build" / "libannc_fused_op.so",
        ]
    )
    return _first_file(candidates, "libannc_fused_op.so")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, help="annc-tf-pipeline path")
    parser.add_argument("--fused-op", type=Path, help="libannc_fused_op.so path")
    parser.add_argument("--fusion-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="directory for GraphDefs, MLIR and the generated kernel library",
    )
    parser.add_argument(
        "--case",
        choices=("all", "sparse_select", "padding", "padding_fast",
                 "sparse_reshape", "sparse_reshape_dynamic_limit",
                 "sparse_segment_mean", "sparse_segment_sum",
                 "action_id_gather", "target_behavior_interaction",
                 "sparse_dynamic_stitch_n2"),
        default="all",
        help="embedding pattern to verify (default: all)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _build_graph_def():
    graph = tf.Graph()
    with graph.as_default():
        # Keep the leading dimension dynamic so tf.shape survives tf2atir.
        # The config pattern requires separate Shape(b_column)/Shape(c_column)
        # nodes instead of frontend-folded shape constants.
        a = tf.compat.v1.placeholder(tf.int32, shape=[None], name="a")
        b = tf.compat.v1.placeholder(tf.int32, shape=[None], name="b")
        c = tf.compat.v1.placeholder(tf.int32, shape=[None], name="c")

        gt = tf.constant(1, dtype=tf.int32, name="gt")
        eq1 = tf.constant(7, dtype=tf.int32, name="eq1")
        eq2 = tf.constant(8, dtype=tf.int32, name="eq2")
        eq3 = tf.constant(3, dtype=tf.int32, name="eq3")
        reshape_shape = tf.constant([-1, 1], tf.int32, name="reshape_shape")

        x = tf.reshape(a, reshape_shape, name="x")
        b_column = tf.reshape(b, reshape_shape, name="b_column")
        c_column = tf.reshape(c, reshape_shape, name="c_column")

        greater = tf.greater(x, gt, name="greater")
        greater_value = tf.cast(greater, tf.float32, name="greater_value")

        one = tf.constant(1.0, tf.float32, name="one")
        b_fill_0 = tf.fill(tf.shape(b_column), one, name="b_fill_0")
        equal_0 = tf.equal(b_column, eq1, name="equal_0")
        select_1 = tf.where(equal_0, b_fill_0, greater_value, name="select_1")

        b_fill_1 = tf.fill(tf.shape(b_column), one, name="b_fill_1")
        equal_1 = tf.equal(b_column, eq2, name="equal_1")
        y = tf.where(equal_1, b_fill_1, select_1, name="y")

        c_fill_0 = tf.fill(tf.shape(c_column), one, name="c_fill_0")
        c_fill_1 = tf.fill(tf.shape(c_column), one, name="c_fill_1")
        equal_2 = tf.equal(c_column, eq3, name="equal_2")
        select_2 = tf.where(equal_2, c_fill_1, c_fill_0, name="select_2")

        empty = tf.fill(
            tf.constant([4, 0], tf.int32, name="empty_shape"),
            tf.constant(0.0, tf.float32, name="zero"),
            name="empty",
        )
        w = tf.concat([y, select_2, empty], axis=1, name="w")

        tf.identity(x, name="output_x")
        tf.identity(y, name="output_y")
        tf.identity(w, name="output_w")

    return graph.as_graph_def(add_shapes=True)


def _build_padding_case(fast=False):
    graph = tf.Graph()
    with graph.as_default():
        origin_shape = tf.compat.v1.placeholder(tf.int64, [2], name="origin_shape")
        input_rows = tf.compat.v1.placeholder(tf.int32, [], name="input_rows")
        pack = tf.compat.v1.placeholder(tf.int32, [], name="pack")
        data = tf.compat.v1.placeholder(tf.float32, [4, 3], name="data")
        reshape_sizes = tf.compat.v1.placeholder(tf.int32, [2], name="reshape_sizes")
        cast = tf.cast(origin_shape, tf.int32, name="origin_cast")
        rows = tf.strided_slice(cast, [0], [1], [1], shrink_axis_mask=1,
                                name="origin_rows")
        padding_rows = tf.subtract(rows, input_rows, name="padding_rows")
        padding_shape = tf.stack([padding_rows, pack], axis=0, name="padding_shape")
        padding = tf.fill(padding_shape, 0.0, name="padding")
        padded = tf.concat([data, padding], axis=0, name="padded")
        out_data = tf.reshape(padded, reshape_sizes, name="out_data")
        if fast:
            out_shape = tf.shape(out_data, name="reshaped_shape")
            reshape_rows = tf.strided_slice(out_shape, [0], [1], [1],
                                            shrink_axis_mask=1,
                                            name="reshape_rows")
        tf.identity(padding_rows, name="output_0")
        if fast:
            tf.identity(reshape_rows, name="output_1")
        else:
            tf.identity(out_data, name="output_1")
    feeds = {
        "origin_shape": np.asarray([6, 3], np.int64),
        "input_rows": np.asarray(4, np.int32),
        "pack": np.asarray(3, np.int32),
        "data": np.arange(12, dtype=np.float32).reshape(4, 3),
        "reshape_sizes": np.asarray([6, 3], np.int32),
    }
    return graph.as_graph_def(add_shapes=True), feeds, ("output_0", "output_1")


def _build_sparse_reshape_case(dynamic_limit=False):
    graph = tf.Graph()
    with graph.as_default():
        # Keep the row count dynamic so Shape(keys) remains an ATIR operation.
        keys = tf.compat.v1.placeholder(tf.int64, [None, 3], name="keys")
        begin = tf.compat.v1.placeholder(tf.int32, [2], name="begin")
        pack_const = tf.compat.v1.placeholder(tf.int32, [], name="pack_const")
        new_shape = tf.compat.v1.placeholder(tf.int64, [2], name="new_shape")
        keys_shape = tf.shape(keys, out_type=tf.int32, name="keys_shape")
        rows = tf.strided_slice(keys_shape, [0], [1], [1], shrink_axis_mask=1,
                                name="rows")
        packed_shape = tf.stack([rows, pack_const], axis=0, name="packed_shape")
        input_shape = tf.cast(packed_shape, tf.int64, name="input_shape")
        keys_slice = tf.strided_slice(keys, begin, [0, 0], [1, 1],
                                      begin_mask=1, end_mask=1,
                                      shrink_axis_mask=2, name="keys_slice")
        keys_column = tf.reshape(keys_slice, [-1, 1], name="keys_column")
        if dynamic_limit:
            range_limit = tf.compat.v1.placeholder(tf.int32, [], name="range_limit")
            range_value = tf.range(range_limit, dtype=tf.int32, name="range")
        else:
            range_value = tf.range(4, dtype=tf.int32, name="range")
        range_column = tf.reshape(range_value, [-1, 1], name="range_column")
        range_i64 = tf.cast(range_column, tf.int64, name="range_i64")
        indices = tf.concat([range_i64, tf.cast(keys_column, tf.int64)], axis=-1,
                            name="indices")
        out_indices, out_shape = tf.raw_ops.SparseReshape(
            input_indices=indices, input_shape=input_shape,
            new_shape=new_shape, name="sparse_reshape")
        tf.identity(out_indices, name="output_0")
        tf.identity(out_shape, name="output_1")
    feeds = {
        "keys": np.asarray([[0, 2, 1], [1, 0, 2], [2, 1, 0], [0, 1, 2]], np.int64),
        "begin": np.asarray([0, 0], np.int32),
        "pack_const": np.asarray(3, np.int32),
        "new_shape": np.asarray([2, 6], np.int64),
    }
    if dynamic_limit:
        feeds["range_limit"] = np.asarray(4, np.int32)
    return graph.as_graph_def(add_shapes=True), feeds, ("output_0", "output_1")


def _build_segment_case(combiner):
    graph = tf.Graph()
    with graph.as_default():
        keys = tf.compat.v1.placeholder(tf.int64, [4, 2], name="keys")
        begin = tf.compat.v1.placeholder(tf.int32, [2], name="begin")
        data = tf.compat.v1.placeholder(tf.float32, [6, 3], name="data")
        indices = tf.compat.v1.placeholder(tf.int64, [4], name="indices")
        begin_1 = tf.compat.v1.placeholder(tf.int32, [1], name="begin_1")
        segment_ids = tf.strided_slice(keys, begin, [0, 0], [1, 1],
                                       begin_mask=1, end_mask=1,
                                       shrink_axis_mask=2, name="segment_ids")
        if combiner == "mean":
            reduced = tf.raw_ops.SparseSegmentMean(
                data=data, indices=indices, segment_ids=segment_ids,
                name="output")
        else:
            reduced = tf.raw_ops.SparseSegmentSum(
                data=data, indices=indices, segment_ids=segment_ids,
                name="output")
        shape = tf.shape(reduced, out_type=tf.int32, name="output_shape")
        sliced = tf.strided_slice(shape, begin_1, [1], [1],
                                   shrink_axis_mask=1, name="slice_output")
        tf.identity(reduced, name="output_0")
        tf.identity(sliced, name="output_1")
    feeds = {
        "keys": np.asarray([[0, 0], [1, 0], [1, 0], [2, 0]], np.int64),
        "begin": np.asarray([0, 0], np.int32),
        "data": np.arange(18, dtype=np.float32).reshape(6, 3),
        "indices": np.asarray([0, 1, 2, 3], np.int64),
        "begin_1": np.asarray([1], np.int32),
    }
    return graph.as_graph_def(add_shapes=True), feeds, ("output_0", "output_1")


def _build_dynamic_stitch_case():
    graph = tf.Graph()
    with graph.as_default():
        # Keep the length dynamic so Size(x) remains an explicit Range input.
        x = tf.compat.v1.placeholder(tf.int64, [None], name="x")
        variables = [tf.compat.v1.placeholder(tf.float32, [2, 3], name=f"variable_{i}")
                     for i in range(2)]
        size = tf.size(x, out_type=tf.int32, name="size")
        range_value = tf.range(0, size, 1, name="range")
        partition_id = tf.math.floormod(x, 2, name="partition_id")
        partition_id_i32 = tf.cast(partition_id, tf.int32, name="partition_id_i32")
        index_parts = tf.dynamic_partition(range_value, partition_id_i32, 2,
                                           name="index_partition")
        row_id = tf.math.floordiv(x, 2, name="row_id")
        row_parts = tf.dynamic_partition(row_id, partition_id_i32, 2,
                                         name="row_partition")
        data_parts = [tf.gather(v, row_parts[i], axis=0, name=f"data_{i}")
                      for i, v in enumerate(variables)]
        output = tf.dynamic_stitch(index_parts, data_parts, name="output")
        tf.identity(output, name="output_0")
    feeds = {
        "x": np.asarray([0, 1, 2, 3], np.int64),
        "variable_0": np.asarray([[1, 2, 3], [4, 5, 6]], np.float32),
        "variable_1": np.asarray([[10, 20, 30], [40, 50, 60]], np.float32),
    }
    graph_def = graph.as_graph_def(add_shapes=True)
    # ParallelDynamicStitch is an ANNC frontend op rather than a TensorFlow
    # runtime op. Build the equivalent DynamicStitch graph first, then retag
    # only that node so tf2atir lowers it to atir.ParallelDynamicStitch.
    for node in graph_def.node:
        if node.name == "output":
            node.op = "ParallelDynamicStitch"
            break
    return graph_def, feeds, ("output_0",)


def _build_action_gather_case():
    graph = tf.Graph()
    with graph.as_default():
        params = tf.compat.v1.placeholder(tf.float32, [10, 4], name="params")
        indices1 = tf.compat.v1.placeholder(tf.int64, [2, 1], name="indices1")
        indices2 = tf.compat.v1.placeholder(tf.int32, [1, 2], name="indices2")
        pack_dim = tf.compat.v1.placeholder(tf.int32, [], name="pack_dim")
        pack = tf.compat.v1.placeholder(tf.int32, [], name="pack")
        inner = tf.gather(params, indices1, axis=0, name="inner")
        outer = tf.gather(inner, indices2, axis=0, name="outer")
        reshape_shape = tf.stack([pack_dim, -1], axis=0, name="reshape_shape")
        reshaped = tf.reshape(outer, reshape_shape, name="reshaped")
        padding_shape = tf.stack([pack_dim, pack], axis=0, name="padding_shape")
        padding = tf.fill(padding_shape, 0.0, name="padding")
        output = tf.concat([reshaped, padding], axis=-1, name="output")
        tf.identity(output, name="output_0")
    feeds = {
        "params": np.arange(40, dtype=np.float32).reshape(10, 4),
        "indices1": np.asarray([[0], [2]], np.int64),
        "indices2": np.asarray([[1, 0]], np.int32),
        "pack_dim": np.asarray(2, np.int32),
        "pack": np.asarray(1, np.int32),
    }
    return graph.as_graph_def(add_shapes=True), feeds, ("output_0",)


def _build_target_behavior_case():
    graph = tf.Graph()
    with graph.as_default():
        batch_input = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], name="batch_input")
        weight = tf.compat.v1.placeholder(tf.float32, [3, 2], name="weight")
        bias = tf.compat.v1.placeholder(tf.float32, [2], name="bias")
        tile_input = tf.compat.v1.placeholder(tf.float32, [1, 2, 2], name="tile_input")
        batch_matmul = tf.matmul(batch_input, weight, name="batch_matmul")
        biased = tf.add(batch_matmul, bias, name="biased")
        absolute = tf.abs(biased, name="absolute")
        positive_twice = tf.add(absolute, biased, name="positive_twice")
        relu = tf.multiply(positive_twice, 0.5, name="relu")
        tiled = tf.tile(tile_input, [1, 1, 1], name="tiled")
        difference = tf.subtract(relu, tiled, name="difference")
        product = tf.multiply(relu, tiled, name="product")
        output = tf.concat([relu, tiled, difference, product], axis=-1, name="output")
        tf.identity(output, name="output_0")
    feeds = {
        "batch_input": np.arange(6, dtype=np.float32).reshape(1, 2, 3),
        "weight": np.asarray([[1, 0], [0, 1], [1, -1]], np.float32),
        "bias": np.asarray([1, -1], np.float32),
        "tile_input": np.ones((1, 2, 2), np.float32),
    }
    return graph.as_graph_def(add_shapes=True), feeds, ("output_0",)


def _read_graph_def(path):
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(path.read_bytes())
    return graph_def


def _run_graph(graph_def, feeds, output_names):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")

    config = tf.compat.v1.ConfigProto()
    config.graph_options.rewrite_options.disable_meta_optimizer = True
    with tf.compat.v1.Session(graph=graph, config=config) as session:
        feed_dict = {
            graph.get_tensor_by_name(f"{name}:0"): value
            for name, value in feeds.items()
        }
        outputs = [graph.get_tensor_by_name(f"{name}:0") for name in output_names]
        return session.run(outputs, feed_dict=feed_dict)


def _run_pipeline(
    pipeline, config, input_graph, output_graph, work_dir, output_names, verbose
):
    command = [
        os.fspath(pipeline),
        "--input_graphdef",
        os.fspath(input_graph),
        "--output_graphdef",
        os.fspath(output_graph),
        "--work_dir",
        os.fspath(work_dir),
        "--fusion-config",
        os.fspath(config),
        "--keep_temps",
        "--dump-fusion-metadata",
    ]
    for output_name in output_names:
        command.extend(("--output_tensor", f"{output_name}:0"))
    if verbose:
        command.append("--verbose")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"annc-tf-pipeline failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if verbose:
        print(completed.stdout, end="")
        print(completed.stderr, end="")


_FUSED_OP_LOADED = False


def _load_fused_op(path):
    global _FUSED_OP_LOADED
    if _FUSED_OP_LOADED:
        return
    # Preload-mode addon symbols are resolved against TensorFlow's host
    # libraries, which Python normally loads with local visibility.
    tf_lib_dir = Path(tf.sysconfig.get_lib())
    global_mode = getattr(ctypes, "RTLD_GLOBAL", os.RTLD_GLOBAL)
    ctypes.CDLL(
        os.fspath(tf_lib_dir / "libtensorflow_framework.so.2"),
        mode=global_mode,
    )
    ctypes.CDLL(
        os.fspath(tf_lib_dir / "libtensorflow_cc.so.2"),
        mode=global_mode,
    )
    tf.load_op_library(os.fspath(path))
    _FUSED_OP_LOADED = True


CASE_BUILDERS = {
    "sparse_select": lambda: (
        _build_graph_def(),
        {
            "a": np.asarray([-1, 2, -3, 5], dtype=np.int32),
            "b": np.asarray([9, 7, 9, 8], dtype=np.int32),
            "c": np.asarray([3, 0, 3, 4], dtype=np.int32),
        },
        ("output_x", "output_y", "output_w"),
        "kp_sparse_select",
    ),
    "padding": lambda: (*_build_padding_case(False), "kp_embedding_padding"),
    "padding_fast": lambda: (
        *_build_padding_case(True),
        "kp_embedding_padding_fast",
    ),
    "sparse_reshape": lambda: (
        *_build_sparse_reshape_case(False),
        "kp_sparse_reshape",
    ),
    "sparse_reshape_dynamic_limit": lambda: (
        *_build_sparse_reshape_case(True),
        "kp_sparse_reshape_dynamic_limit",
    ),
    "sparse_segment_mean": lambda: (
        *_build_segment_case("mean"),
        "kp_sparse_segment_reduce",
    ),
    "sparse_segment_sum": lambda: (
        *_build_segment_case("sum"),
        "kp_sparse_segment_reduce",
    ),
    "action_id_gather": lambda: (
        *_build_action_gather_case(),
        "kp_embedding_action_id_gather",
    ),
    "target_behavior_interaction": lambda: (
        *_build_target_behavior_case(),
        "kp_target_behavior_interaction",
    ),
    "sparse_dynamic_stitch_n2": lambda: (
        *_build_dynamic_stitch_case(),
        "kp_sparse_dynamic_stitch",
    ),
}


def _run_case(
    case_name, builder, pipeline, fused_op, fusion_config, root_work_dir, verbose
):
    graph_def, feeds, output_names, expected_pattern = builder()
    work_dir = root_work_dir / case_name
    work_dir.mkdir(parents=True, exist_ok=True)
    input_graph = work_dir / f"{case_name}.pb"
    rewritten_graph = work_dir / f"{case_name}_annc.pb"
    input_graph.write_bytes(graph_def.SerializeToString())
    if case_name == "sparse_dynamic_stitch_n2":
        x = feeds["x"]
        variables = [feeds["variable_0"], feeds["variable_1"]]
        expected = [np.asarray([variables[int(index) % 2][int(index) // 2]
                                for index in x])]
    else:
        expected = _run_graph(graph_def, feeds, output_names)
    _run_pipeline(
        pipeline,
        fusion_config,
        input_graph,
        rewritten_graph,
        work_dir,
        output_names,
        verbose,
    )
    rewritten = _read_graph_def(rewritten_graph)
    fused_nodes = [node for node in rewritten.node if node.op == "ANNCFused"]
    if len(fused_nodes) != 1:
        raise AssertionError(
            f"{case_name}: expected exactly one ANNCFused node, found "
            f"{len(fused_nodes)}"
        )
    fused = fused_nodes[0]
    pattern = fused.attr["fusion_pattern"].s.decode()
    if pattern != expected_pattern:
        raise AssertionError(
            f"{case_name}: unexpected fusion pattern {pattern!r}, "
            f"expected {expected_pattern!r}"
        )
    abi = fused.attr["abi"].s.decode()
    kernel_library = Path(fused.attr["shared_lib_path"].s.decode())
    if not kernel_library.is_file():
        raise AssertionError(f"{case_name}: generated kernel library is missing: {kernel_library}")
    _load_fused_op(fused_op)
    actual = _run_graph(rewritten, feeds, output_names)
    for output_name, expected_value, actual_value in zip(output_names, expected, actual):
        np.testing.assert_allclose(actual_value, expected_value, rtol=0.0, atol=0.0)
        print(f"{case_name}/{output_name}: shape={actual_value.shape}")
    print(f"PASS: {case_name} ({pattern}, {abi})")
    print(f"kernel: {kernel_library}")
    print(f"artifacts: {work_dir}")


def main():
    args = _parse_args()
    pipeline = args.pipeline.resolve() if args.pipeline else _default_pipeline()
    fused_op = args.fused_op.resolve() if args.fused_op else _default_fused_op()
    fusion_config = args.fusion_config.resolve()
    if not pipeline.is_file():
        raise FileNotFoundError(pipeline)
    if not fused_op.is_file():
        raise FileNotFoundError(fused_op)
    if not fusion_config.is_file():
        raise FileNotFoundError(fusion_config)

    if args.work_dir:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="annc_kp_embedding_patterns_e2e_"))
    case_names = list(CASE_BUILDERS) if args.case == "all" else [args.case]
    failures = []
    for case_name in case_names:
        try:
            _run_case(
                case_name,
                CASE_BUILDERS[case_name],
                pipeline,
                fused_op,
                fusion_config,
                work_dir,
                args.verbose,
            )
        except Exception as error:  # Keep all case diagnostics in one run.
            failures.append((case_name, error))
            print(f"FAIL: {case_name}: {error}")
    if failures:
        raise RuntimeError(
            "embedding pattern E2E failures: "
            + ", ".join(name for name, _ in failures)
        )
    print(f"ALL PASS: {len(case_names)} embedding pattern cases")


if __name__ == "__main__":
    main()
