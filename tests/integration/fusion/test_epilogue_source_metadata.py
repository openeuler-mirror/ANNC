"""Exercise epilogue kernel extraction and its template identity."""

import os
from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def tools():
    build = Path(os.environ.get("ANNC_BUILD_DIR", ROOT / "build"))
    binaries = {name: build / "bin" / name for name in ("annc-opt", "annc-asm")}
    for binary in binaries.values():
        if not binary.is_file():
            pytest.skip(f"tool is not built: {binary}")
    return binaries


def _run(binary, text, *passes):
    # Epilogue fusion only consumes the candidate when GEMM epilogue mode is on.
    env = dict(os.environ, ANNC_GEMM=os.environ.get("ANNC_GEMM", "2"))
    result = subprocess.run(
        [str(binary), *passes], input=text, text=True, capture_output=True,
        check=False, env=env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _source(prefix="source", view=False):
    # Only source annotations change between equivalent inputs. The shape
    # constant's declared symbol and payload remain part of its definition.
    graph = '''
module {
  func.func @unfused() ->
      !atir.tensor<1xi64, data = dense<7> : tensor<1xi64>> {
    %c = atir.constant "private" @untouched ->
        <1xi64, data = dense<7> : tensor<1xi64>>
        {metadata = {"tf.name" = "source/untouched"}}
    return %c : !atir.tensor<1xi64, data = dense<7> : tensor<1xi64>>
  }
  func.func @main(%lhs: !atir.tensor<4x4xf32>,
                  %rhs: !atir.tensor<4x4xf32>,
                  %bias: !atir.tensor<4xf32>) -> @OUTPUT_TYPE@ {
    %mm_out = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
    %add_out = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
    %relu_out = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
    %mm = "atir.MatMul"(%mm_out, %lhs, %rhs) <{
      withBias = false, left_transpose = false, right_transpose = false,
      output_transpose = false
    }> {metadata = {"tf.name" = "source/matmul", "tf.op" = "MatMul",
                    "tf.input.0" = "source/lhs:0",
                    "tf.input.1" = "source/rhs:0",
                    "tf.control_input.0" = "source/control"},
        tf.name = "source/matmul", tf_name = "source/matmul"} :
        (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
         !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32>
    %add = "atir.Add"(%add_out, %mm, %bias) <{
      do_relu = false, relu_limit = -1.0 : f32
    }> {metadata = {"tf.name" = "source/add", "tf.op" = "AddV2",
                    "tf.input.0" = "source/matmul:0",
                    "tf.input.1" = "source/bias:0"}} :
        (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
         !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32>
    %relu = "atir.Relu"(%relu_out, %add) <{relu_limit = -1.0 : f32}>
        {metadata = {"tf.name" = "source/relu", "tf.op" = "Relu",
                     "tf.input.0" = "source/add:0",
                     "tf.output_tensors" = ["source/result:0"]}} :
        (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>)
        -> !atir.tensor<4x4xf32>
    @TAIL@
  }
}
'''
    output_type = "!atir.tensor<4x4xf32>"
    tail = "return %relu : !atir.tensor<4x4xf32>"
    if view:
        output_type = "!atir.tensor<4x1x4xf32>"
        tail = '''
    %shape = atir.constant "private" @shape ->
        <3xi64, data = dense<[4, 1, 4]> : tensor<3xi64>>
        {metadata = {"tf.name" = "source/shape", "tf.op" = "Const"},
         tf_name = "source/shape"}
    %view_out = "atir.buffer"() : () -> !atir.tensor<4x1x4xf32>
    %view = "atir.Reshape"(%view_out, %relu, %shape)
        {metadata = {"tf.name" = "source/view", "tf.op" = "Reshape",
                     "tf.input.0" = "source/relu:0",
                     "tf.input.1" = "source/shape:0",
                     "tf.output_tensors" = ["source/result:0"]}} :
        (!atir.tensor<4x1x4xf32>, !atir.tensor<4x4xf32>,
         !atir.tensor<3xi64, data = dense<[4, 1, 4]> : tensor<3xi64>>)
        -> !atir.tensor<4x1x4xf32>
    return %view : !atir.tensor<4x1x4xf32>
'''
    return (graph.replace("@OUTPUT_TYPE@", output_type).replace("@TAIL@", tail)
            .replace("source/", f"{prefix}/"))


def _extract(tools, source):
    text = _run(tools["annc-opt"], source,
                "--atir-gemm-epilogue-fusion", "--atir-op-fusion")
    kernels = text.split("func.func private @")
    assert len(kernels) == 2, text
    header, body = re.split(r"\}\s*\{\n", kernels[1], maxsplit=1)
    fingerprints = re.findall(r'template_fingerprint = "([0-9a-f]{64})"', header)
    assert len(fingerprints) == 1, header
    return text, header, body, fingerprints[0]


@pytest.mark.parametrize("view", [False, True], ids=["chain", "terminal-view"])
def test_equivalent_epilogues_ignore_source_metadata(tools, view):
    left = _extract(tools, _source("left", view))
    right = _extract(tools, _source("right", view))
    assert left[3] == right[3]
    assert left[2] == right[2]
    for prefix, (text, header, body, _) in (("left", left), ("right", right)):
        assert "metadata" not in body
        assert "tf.name" not in body and "tf_name" not in body
        assert "annc.gemm.epilogue.candidate" in body
        assert 'fusion.pattern = "matmul_add_relu"' in header
        assert f'tf_name = "{prefix}/result:0"' in header
        # Source metadata outside the extracted kernel remains available.
        assert f'metadata = {{tf.name = "{prefix}/untouched"}}' in text
        if view:
            assert "atir.Reshape" in body
            assert "dense<[4, 1, 4]>" in body


@pytest.mark.parametrize("change", ["relu-limit", "binary-op", "shape-value"])
def test_computational_differences_keep_distinct_fingerprints(tools, change):
    source = _source(view=True)
    if change == "relu-limit":
        different = source.replace('"atir.Relu"(%relu_out, %add) <{relu_limit = -1.0',
                                   '"atir.Relu"(%relu_out, %add) <{relu_limit = 6.0')
    elif change == "binary-op":
        different = re.sub(r'"atir.Add"(\([^)]*\)) <\{[^}]*\}>',
                           r'"atir.Mul"\1', source)
    else:
        # The equivalent inferred dimension must still preserve its literal
        # constant payload in the template.
        different = source.replace("dense<[4, 1, 4]>", "dense<[-1, 1, 4]>")
    assert _extract(tools, source)[3] != _extract(tools, different)[3]


@pytest.mark.parametrize("view", [False, True], ids=["chain", "terminal-view"])
def test_cleaned_epilogue_can_be_committed_by_backend(tools, view):
    source = _source(view=view).replace("relu_limit = -1.0", "relu_limit = 6.0")
    fused, _, _, _ = _extract(tools, source)
    committed = _run(tools["annc-asm"], fused, "--atir-prune-func",
                     "--aarch64-atir-gemm-epilogue-commit")
    assert "annc.gemm.epilogue =" in committed
    assert "annc.gemm.epilogue.candidate" not in committed
    assert 'opcode = "add"' in committed and 'opcode = "relu"' in committed
    assert "limit = 6.000000e+00 : f32" in committed
    assert '"atir.Add"' not in committed and '"atir.Relu"' not in committed
