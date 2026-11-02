"""Strings cross the framework boundary through typed StableHLO custom calls."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPT = Path(os.environ.get('ANNC_FRAMEWORK_OPT', ROOT / 'build/bin/annc-onnx-framework-opt'))
ONNX_MLIR_OPT = os.environ.get('ANNC_ONNX_MLIR_OPT')
pytestmark = pytest.mark.skipif(not OPT.is_file(), reason='build the framework pass runner first')


def test_string_constant_expand_preserve_data_shape_and_source(tmp_path):
    source = tmp_path / 'input.mlir'
    output = tmp_path / 'output.mlir'
    source.write_text('''module {
  func.func @main(%shape: tensor<1xi64>) -> tensor<?x!onnx.String> {
    %c = "onnx.Constant"() {value = dense<["one", "two"]> : tensor<2x!onnx.String>, onnx_node_name = "labels"} : () -> tensor<2x!onnx.String>
    %r = "onnx.Expand"(%c, %shape) {onnx_node_name = "expand_labels"} : (tensor<2x!onnx.String>, tensor<1xi64>) -> tensor<?x!onnx.String>
    return %r : tensor<?x!onnx.String>
  }
}
''')
    assert OPT.is_file(), 'build the independent framework optimizer'
    result = subprocess.run([str(OPT), str(source), '--onnx-string-to-framework-custom-call',
                             '-o', str(output)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    text = output.read_text()
    assert 'atir.' not in text
    assert '!onnx.String' not in text
    assert '"annc_framework.custom_call"' in text
    assert 'stablehlo.custom_call' not in text
    assert 'annc.onnx.string.constant' in text
    assert 'annc.onnx.string.expand' in text
    assert '6f6e65' in text and '74776f' in text
    assert 'tensor<2xi64>' in text and 'tensor<?xi64>' in text
    assert 'expand_labels' in text
    assert 'annc.original_element_type = "string"' in text

    lowered = tmp_path / 'lowered.mlir'
    result = subprocess.run([
        str(OPT), str(output), '--framework-custom-call-to-stablehlo',
        '-o', str(lowered)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'stablehlo.custom_call' in lowered.read_text()
    assert 'annc_framework.custom_call' not in lowered.read_text()

    if ONNX_MLIR_OPT:
        converted = tmp_path / 'converted.mlir'
        result = subprocess.run([
            ONNX_MLIR_OPT, '--allow-unregistered-dialect',
            '--convert-onnx-to-stablehlo', '--mlir-print-op-generic',
            str(output), '-o', str(converted)], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert '"annc_framework.custom_call"' in converted.read_text()


@pytest.mark.parametrize('passes', [
    ['--onnx-string-to-framework-custom-call'],
    ['--verify-framework-stablehlo'],
])
def test_unranked_string_signature_is_rejected(tmp_path, passes):
    source = tmp_path / 'unranked.mlir'
    source.write_text('module { func.func private @f(tensor<*x!onnx.String>) }')
    result = subprocess.run([str(OPT), str(source), *passes],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert 'framework' in result.stderr


def test_pure_gate_checks_nested_signature_types(tmp_path):
    source = tmp_path / 'nested.mlir'
    source.write_text('module { func.func private @f(tuple<tensor<2x!onnx.String>>) }')
    result = subprocess.run([str(OPT), str(source), '--verify-framework-stablehlo'],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert 'framework' in result.stderr
