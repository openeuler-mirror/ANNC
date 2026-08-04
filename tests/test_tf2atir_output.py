import os
import shutil
import subprocess
from pathlib import Path

import pytest


GRAPHDEF_PBTXT = r'''
node {
  name: "input"
  op: "Placeholder"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr {
    key: "_output_shapes"
    value { list { shape { dim { size: -1 } dim { size: 1 } } } }
  }
}
node {
  name: "left_const"
  op: "Const"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape { dim { size: 1 } }
        float_val: 1.0
      }
    }
  }
  attr {
    key: "_output_shapes"
    value { list { shape { dim { size: 1 } } } }
  }
}
node {
  name: "right_const"
  op: "Const"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape { dim { size: 1 } }
        float_val: 2.0
      }
    }
  }
  attr {
    key: "_output_shapes"
    value { list { shape { dim { size: 1 } } } }
  }
}
node {
  name: "out_a"
  op: "AddV2"
  input: "input"
  input: "left_const"
  attr { key: "T" value { type: DT_FLOAT } }
  attr {
    key: "_output_shapes"
    value { list { shape { dim { size: -1 } dim { size: 1 } } } }
  }
}
node {
  name: "out_b"
  op: "Mul"
  input: "input"
  input: "right_const"
  attr { key: "T" value { type: DT_FLOAT } }
  attr {
    key: "_output_shapes"
    value { list { shape { dim { size: -1 } dim { size: 1 } } } }
  }
}
'''


def tf2atir_binary() -> Path:
    configured = os.environ.get("ANNC_TF2ATIR_BIN")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "build" / "bin" / "annc-tf2atir"


def run_tf2atir(binary: Path, graph: Path, outputs: list[str], result: Path):
    command = [str(binary), str(graph)]
    for output in outputs:
        command.extend(["--output_tensor", output])
    command.extend(["-o", str(result)])
    return subprocess.run(command, check=False, text=True, capture_output=True)


def test_repeated_output_tensor_arguments_control_graph_outputs(tmp_path: Path):
    binary = tf2atir_binary()
    if not binary.exists():
        pytest.skip(f"annc-tf2atir not found at {binary}")

    graph = tmp_path / "graph.pbtxt"
    graph.write_text(GRAPHDEF_PBTXT)

    both_mlir = tmp_path / "both.mlir"
    both = run_tf2atir(binary, graph, ["out_a", "out_b"], both_mlir)
    assert both.returncode == 0, both.stderr
    both_text = both_mlir.read_text()
    assert 'name = "out_a"' in both_text
    assert 'name = "out_b"' in both_text

    one_mlir = tmp_path / "one.mlir"
    one = run_tf2atir(binary, graph, ["out_a"], one_mlir)
    assert one.returncode == 0, one.stderr
    one_text = one_mlir.read_text()
    assert 'name = "out_a"' in one_text
    assert 'name = "out_b"' not in one_text


def test_pipeline_forwards_repeated_output_tensor_arguments(tmp_path: Path):
    pipeline = Path(__file__).resolve().parents[1] / "build" / "bin" / "annc-tf-pipeline"
    if not pipeline.exists():
        pytest.skip(f"annc-tf-pipeline not found at {pipeline}")

    tool_dir = tmp_path / "tools"
    tool_dir.mkdir()
    pipeline_copy = tool_dir / "annc-tf-pipeline"
    shutil.copy2(pipeline, pipeline_copy)
    pipeline_copy.chmod(0o755)

    fake_tool = r'''#!/usr/bin/env python3
import os
import pathlib
import sys

tool = pathlib.Path(sys.argv[0]).name
arguments = sys.argv[1:]
if tool == "annc-tf2atir":
    pathlib.Path(os.environ["ANNC_CAPTURE_ARGS"]).write_text("\n".join(arguments))
for index, argument in enumerate(arguments[:-1]):
    if argument == "-o":
        pathlib.Path(arguments[index + 1]).touch()
    if argument == "--output_graphdef":
        pathlib.Path(arguments[index + 1]).touch()
'''
    for tool in (
        "annc-tf2atir",
        "annc-opt",
        "annc-asm",
        "annc",
        "annc-converter",
    ):
        path = tool_dir / tool
        path.write_text(fake_tool)
        path.chmod(0o755)

    captured = tmp_path / "tf2atir.args"
    graph = tmp_path / "input.pb"
    graph.write_bytes(b"graph")
    output = tmp_path / "output.pb"
    result = subprocess.run(
        [
            str(pipeline_copy),
            "--input_graphdef",
            str(graph),
            "--output_graphdef",
            str(output),
            "--work_dir",
            str(tmp_path / "pipeline-work"),
            "--keep_temps",
            "--output_tensor",
            "out_a",
            "--output_tensor",
            "out_b",
        ],
        env={**os.environ, "ANNC_CAPTURE_ARGS": str(captured)},
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert captured.read_text().splitlines()[-4:] == [
        "--output_tensor",
        "out_a",
        "--output_tensor",
        "out_b",
    ]
