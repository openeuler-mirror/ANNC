"""Boundary shape computations must become portable StableHLO operations."""
import os
from pathlib import Path
import re
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[3]
OPT = Path(os.environ.get('ANNC_FRAMEWORK_OPT', ROOT / 'build/bin/annc-onnx-framework-opt'))
pytestmark = pytest.mark.skipif(not OPT.is_file(), reason='build the framework pass runner first')


def test_shape_broadcast_affine_and_signed_division_are_legalized(tmp_path):
    source = tmp_path / 'shapes.mlir'
    output = tmp_path / 'stablehlo.mlir'
    source.write_text('''module {
  func.func @main(%a: tensor<?x1xf32>, %b: tensor<1x?xf32>) -> tensor<2xi64> {
    %sa = shape.shape_of %a : tensor<?x1xf32> -> tensor<2xindex>
    %sb = shape.shape_of %b : tensor<1x?xf32> -> tensor<2xindex>
    %s = shape.broadcast %sa, %sb : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %d = shape.get_extent %s, %zero : tensor<2xindex>, index -> index
    %e = affine.apply affine_map<(d0) -> (d0 * 3 + 1)>(%d)
    %q = arith.floordivsi %e, %one : index
    %r = "shape.from_extents"(%q, %d) : (index, index) -> !shape.shape
    %rt = "shape.to_extent_tensor"(%r) : (!shape.shape) -> tensor<2xindex>
    %o = arith.index_cast %rt : tensor<2xindex> to tensor<2xi64>
    return %o : tensor<2xi64>
  }
}
''')
    result = subprocess.run([str(OPT), str(source), '--framework-shape-to-stablehlo',
                             '--mlir-print-op-generic', '-o', str(output)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    text = output.read_text()
    operations = re.findall(r'"([\w.]+)"\(', text)
    assert set(op.split('.')[0] for op in operations) <= {'builtin', 'func', 'stablehlo'}
    assert 'index' not in text
    assert 'stablehlo.get_dimension_size' in text
    assert 'stablehlo.select' in text
    assert 'stablehlo.divide' in text


def test_zero_dimension_broadcast_and_signed_floor_division_values(tmp_path):
    translate = os.environ.get('ANNC_STABLEHLO_TRANSLATE')
    if not translate:
        pytest.skip('set ANNC_STABLEHLO_TRANSLATE for the upstream reference interpreter')
    source = tmp_path / 'shape_values.mlir'
    output = tmp_path / 'shape_values.stablehlo.mlir'
    source.write_text('''module {
  func.func @main() -> (tensor<2xi64>, tensor<i64>, tensor<i64>) {
    %a = arith.constant dense<[0, 1]> : tensor<2xindex>
    %b = arith.constant dense<[1, 3]> : tensor<2xindex>
    %s = shape.broadcast %a, %b : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
    %si = arith.index_cast %s : tensor<2xindex> to tensor<2xi64>
    %n = arith.constant -5 : index
    %p = arith.constant 2 : index
    %q = arith.floordivsi %n, %p : index
    %qi = arith.index_cast %q : index to i64
    %t = arith.floordivsi %p, %n : index
    %ti = arith.index_cast %t : index to i64
    %qt = "tensor.from_elements"(%qi) : (i64) -> tensor<1xi64>
    %tt = "tensor.from_elements"(%ti) : (i64) -> tensor<1xi64>
    %qr = "stablehlo.reshape"(%qt) : (tensor<1xi64>) -> tensor<i64>
    %tr = "stablehlo.reshape"(%tt) : (tensor<1xi64>) -> tensor<i64>
    return %si, %qr, %tr : tensor<2xi64>, tensor<i64>, tensor<i64>
  }
}
''')
    result = subprocess.run([str(OPT), str(source), '--framework-shape-to-stablehlo',
                             '-o', str(output)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([translate, '--interpret', str(output)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == 'tensor<2xi64> {\n  [0, 3]\n}\ntensor<i64> {-3}\ntensor<i64> {-1}\n'
