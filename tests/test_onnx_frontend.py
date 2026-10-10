"""End-to-end tests for the annc-onnx2atir frontend.

These drive the built binary over small generated models and check the CLI
contract: the ATIR that comes out, the coverage report beside it, and the exit
codes. The stage-level behaviour (decoder, parser, resolver, adapter fallback
decisions) is covered by annc-onnx2atir-frontend-test.

The models are generated with the `onnx` Python package; the test skips when it
is not installed, the same way the other binary-driving tests skip when their
binary has not been built.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("onnx")
import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _build_dir():
    configured = os.environ.get("ANNC_BUILD_DIR")
    if configured:
        return Path(configured)
    for candidate in (ROOT / "build-v2", ROOT / "build"):
        if candidate.is_dir():
            return candidate
    return ROOT / "build"


def _binary(build_dir, name):
    path = build_dir / "bin" / name
    if not path.is_file():
        pytest.skip(f"{name} is not built: {path}")
    return path


def _run(command):
    return subprocess.run([os.fspath(arg) for arg in command],
                          text=True, capture_output=True, check=False)


def _make_model(graph, opset=13, check=True):
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="annc-test",
    )
    model.ir_version = 7
    if check:
        onnx.checker.check_model(model)
    return model


def _reshape_chain_model():
    """A small slice of the FusedTensordotMatMul shape chain."""
    shape = helper.make_tensor("shape", TensorProto.INT64, [2], [2, 3])
    graph = helper.make_graph(
        name="graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["x_shape"], name="shape"),
            helper.make_node("Cast", ["x_shape"], ["x_shape_i32"], name="cast",
                             to=TensorProto.INT32),
            helper.make_node("Gather", ["x_shape_i32", "gather_idx"],
                             ["free_axes"], name="gather", axis=0),
            helper.make_node("ReduceProd", ["free_axes"], ["prod"],
                             name="prod", keepdims=0),
            helper.make_node("Unsqueeze", ["prod", "unsq_axes"], ["unsq"],
                             name="unsqueeze"),
            helper.make_node("Concat", ["unsq", "unsq"], ["target"],
                             name="concat", axis=0),
            helper.make_node("Cast", ["target"], ["target_i64"], name="cast2",
                             to=TensorProto.INT64),
            helper.make_node("Reshape", ["x", "target_i64"], ["reshaped"],
                             name="reshape"),
            helper.make_node("MatMul", ["reshaped", "w"], ["matmul"],
                             name="matmul"),
            helper.make_node("Reshape", ["matmul", "shape"], ["y"],
                             name="final_reshape"),
        ],
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [6, 1])],
        initializer=[
            shape,
            numpy_helper.from_array(np.array([0], dtype=np.int64), "gather_idx"),
            numpy_helper.from_array(np.array([0], dtype=np.int64), "unsq_axes"),
            numpy_helper.from_array(np.random.rand(4, 1).astype(np.float32), "w"),
        ],
    )
    return _make_model(graph)


def _unsupported_op_model():
    """`Or` has no ATIR op: it must be reported, not dropped."""
    graph = helper.make_graph(
        name="graph",
        nodes=[
            helper.make_node("Equal", ["x", "y"], ["eq"], name="eq"),
            helper.make_node("Or", ["eq", "eq"], ["or_out"], name="or_node"),
        ],
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3]),
        ],
        outputs=[helper.make_tensor_value_info("or_out", TensorProto.BOOL, [2, 3])],
    )
    return _make_model(graph)


def _control_flow_model():
    then_graph = helper.make_graph([], "then", [],
                                   [helper.make_tensor_value_info(
                                       "branch_out", TensorProto.FLOAT, [2])])
    graph = helper.make_graph(
        name="graph",
        nodes=[helper.make_node(
            "If", ["cond"], ["out"], name="if_node",
            then_branch=then_graph, else_branch=then_graph)],
        inputs=[helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
    )
    # Not run through the ONNX checker: the point of this model is the
    # GRAPH-typed attribute, and the checker rejects a subgraph whose declared
    # output no node produces before the frontend ever sees it.
    return _make_model(graph, check=False)


def _import(build_dir, model_path, tmp_path, *extra):
    binary = _binary(build_dir, "annc-onnx2atir")
    stats = tmp_path / "stats.json"
    atir = tmp_path / "model.mlir"
    result = _run([binary, model_path, "--dump-stats", stats, "-o", atir,
                   *extra])
    return result, stats, atir


def test_shape_chain_imports_to_the_matching_atir_ops(tmp_path):
    build_dir = _build_dir()
    model_path = tmp_path / "chain.onnx"
    onnx.save(_reshape_chain_model(), model_path)
    result, stats, atir = _import(build_dir, model_path, tmp_path)
    assert result.returncode == 0, result.stderr + result.stdout

    text = atir.read_text()
    for op in ("atir.Shape", "atir.Cast", "atir.Gather", "atir.Prod",
               "atir.ExpandDims", "atir.Concat", "atir.Reshape", "atir.MatMul"):
        assert op in text, f"{op} missing from the imported module"

    report = json.loads(stats.read_text())
    assert report["import_coverage"]["fallback_nodes"] == 0
    assert report["conservation"]["nodes_accounted"] is True
    assert report["conservation"]["opaque_nodes_match"] is True
    assert report["contract_version"] == "1.0.0"


def test_unsupported_op_is_reported_and_exits_two(tmp_path):
    build_dir = _build_dir()
    model_path = tmp_path / "unsupported.onnx"
    onnx.save(_unsupported_op_model(), model_path)
    result, stats, atir = _import(build_dir, model_path, tmp_path)

    # Exit 2 means "completed, but with fallbacks" -- never a silent success.
    assert result.returncode == 2, result.stderr + result.stdout
    report = json.loads(stats.read_text())
    coverage = report["import_coverage"]
    assert coverage["fallback_nodes"] == 1
    assert coverage["fallback_reasons"] == {"NO_SPEC": 1}
    entry = [p for p in report["provenance"] if p["node"] == "or_node"][0]
    assert entry["decision"] == "opaque"
    assert entry["reason"] == "NO_SPEC"
    # Every model node still has a provenance row: nothing was dropped.
    assert report["conservation"]["model_nodes"] == 2
    assert report["conservation"]["provenance_entries"] == 2
    assert "atir.opaque" in atir.read_text()


def test_strict_turns_a_fallback_into_a_failure(tmp_path):
    build_dir = _build_dir()
    model_path = tmp_path / "unsupported.onnx"
    onnx.save(_unsupported_op_model(), model_path)
    result, _, _ = _import(build_dir, model_path, tmp_path, "--strict")
    assert result.returncode == 1
    assert "--strict" in result.stderr


def test_control_flow_is_refused_model_wide(tmp_path):
    build_dir = _build_dir()
    model_path = tmp_path / "control_flow.onnx"
    onnx.save(_control_flow_model(), model_path)
    result, _, _ = _import(build_dir, model_path, tmp_path)
    assert result.returncode == 1
    assert "control flow" in result.stderr
    assert "branch" in result.stderr


def test_opset_other_than_13_is_refused_unless_relaxed(tmp_path):
    build_dir = _build_dir()
    model_path = tmp_path / "opset18.onnx"
    onnx.save(_make_model(_reshape_chain_model().graph, opset=18), model_path)

    result, _, _ = _import(build_dir, model_path, tmp_path)
    assert result.returncode == 1
    assert "OP_SET_MISMATCH" in result.stderr

    relaxed, stats, _ = _import(build_dir, model_path, tmp_path,
                                "--relaxed-opset")
    assert relaxed.returncode == 2, relaxed.stderr + relaxed.stdout
    report = json.loads(stats.read_text())
    reasons = report["import_coverage"]["fallback_reasons"]
    assert reasons.get("OP_SET_MISMATCH", 0) > 0


def test_inference_report_is_written_per_tensor(tmp_path):
    build_dir = _build_dir()
    model_path = tmp_path / "chain.onnx"
    onnx.save(_reshape_chain_model(), model_path)
    binary = _binary(build_dir, "annc-onnx2atir")
    inference = tmp_path / "inference.json"
    result = _run([binary, model_path, "--dump-inference", inference,
                   "--quiet", "-o", tmp_path / "model.mlir"])
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(inference.read_text())
    assert {"graph_inputs", "initializers", "tensors"} <= set(report)
    entry = report["tensors"]["reshaped"]
    assert entry["dtype"] == "float32"
    assert entry["rank_known"] is True
