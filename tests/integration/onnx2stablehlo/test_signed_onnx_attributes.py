"""Native ONNX-MLIR emits signed attributes, even for dimensions and axes."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPT = Path(os.environ.get('ANNC_FRAMEWORK_OPT', ROOT / 'build/bin/annc-onnx-framework-opt'))
pytestmark = pytest.mark.skipif(not OPT.is_file(), reason='build the framework pass runner first')


@pytest.mark.parametrize('operation, signature', [
    ('%r = "onnx.ReduceProdV13"(%a) {axes = [0], keepdims = 1 : si64} : (tensor<2xi32>) -> tensor<1xi32>\nreturn %r : tensor<1xi32>',
     '%a: tensor<2xi32>) -> tensor<1xi32>'),
    ('%s, %r = "onnx.ConcatShapeTranspose"(%a, %b) {axis = -1 : si64, start = 0 : si64, perm = [0, 1, 2]} : (tensor<?x50x32xf32>, tensor<?x50x32xf32>) -> (tensor<3xi64>, tensor<?x50x64xf32>)\nreturn %r : tensor<?x50x64xf32>',
     '%a: tensor<?x50x32xf32>, %b: tensor<?x50x32xf32>) -> tensor<?x50x64xf32>'),
])
def test_signed_attributes_from_native_importer(tmp_path, operation, signature):
    source = tmp_path / 'native.mlir'
    source.write_text(f'module {{\nfunc.func @main({signature} {{\n{operation}\n}}\n}}\n')
    result = subprocess.run([str(OPT), str(source), '--onnx-external-to-stablehlo',
                             '-o', str(tmp_path / 'output.mlir')],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
