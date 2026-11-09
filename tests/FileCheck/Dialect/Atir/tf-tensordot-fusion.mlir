// RUN: env ANNC_GEMM=1 annc-opt --split-input-file %s --atir-op-fusion | FileCheck %s
// RUN: env ANNC_GEMM=0 annc-opt --split-input-file %s --atir-op-fusion | FileCheck %s --check-prefix=OFF --implicit-check-not='fusion.pattern = "tensordot_matmul"'
// OFF: func.func @main
// TF has no ONNX contract marker. A plain MatMul subfusion is insufficient.

// folded
// CHECK-LABEL: func.func @main
// CHECK-LABEL: func.func private @{{[^ (]+}}(
// CHECK-SAME: %[[X:[^:]+]]: !atir.tensor<2x3x8xf32{{.*}}>, %[[W:[^:]+]]: !atir.tensor<8x4xf32{{.*}}>, %[[Y:[^:]+]]: !atir.tensor<2x3x4xf32{{.*}}>) attributes
// CHECK-SAME: abi = "mlir_ciface"
// CHECK-SAME: execution_mode = "aot"
// CHECK-SAME: fusion.pattern = "tensordot_matmul"
// CHECK: atir.Dot %[[Y]], %[[X]], %[[W]]
// CHECK: return
module attributes {module.state = "atir"} {
  func.func @main(%x: !atir.tensor<2x3x8xf32, name = "input">) -> !atir.tensor<2x3x4xf32, name = "restore"> {
    %w = atir.constant "public" @"weight" -> <8x4xf32, name = "weight", data = dense<0.25> : tensor<8x4xf32>>
    %fs = atir.constant "public" @"flat_shape" -> <2xi32, name = "flat_shape", data = dense<[6, 8]> : tensor<2xi32>>
    %fb = "atir.buffer"() : () -> !atir.tensor<6x8xf32, name = "flatten">
    %flat = atir.Reshape %fb, %x, %fs : <6x8xf32, name = "flatten">, <2x3x8xf32, name = "input">, <2xi32, name = "flat_shape", data = dense<[6, 8]> : tensor<2xi32>> -> <6x8xf32, name = "flatten">
    %mb = "atir.buffer"() : () -> !atir.tensor<6x4xf32, name = "matmul">
    %mm = "atir.MatMul"(%mb, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<6x4xf32, name = "matmul">, !atir.tensor<6x8xf32, name = "flatten">, !atir.tensor<8x4xf32, name = "weight", data = dense<0.25> : tensor<8x4xf32>>) -> !atir.tensor<6x4xf32, name = "matmul">
    %rs = atir.constant "public" @"restore_shape" -> <3xi32, name = "restore_shape", data = dense<[2, 3, 4]> : tensor<3xi32>>
    %rb = "atir.buffer"() : () -> !atir.tensor<2x3x4xf32, name = "restore">
    %result = atir.Reshape %rb, %mm, %rs : <2x3x4xf32, name = "restore">, <6x4xf32, name = "matmul">, <3xi32, name = "restore_shape", data = dense<[2, 3, 4]> : tensor<3xi32>> -> <2x3x4xf32, name = "restore">
    return %result : !atir.tensor<2x3x4xf32, name = "restore">
  }
}

// -----

// wrong_shape
// CHECK-LABEL: func.func @main
// CHECK-NOT: fusion.pattern = "tensordot_matmul"
module attributes {module.state = "atir"} {
  func.func @main(%x: !atir.tensor<2x3x8xf32, name = "input">) -> !atir.tensor<3x2x4xf32, name = "restore"> {
    %w = atir.constant "public" @"weight" -> <8x4xf32, name = "weight", data = dense<0.25> : tensor<8x4xf32>>
    %fs = atir.constant "public" @"flat_shape" -> <2xi32, name = "flat_shape", data = dense<[6, 8]> : tensor<2xi32>>
    %fb = "atir.buffer"() : () -> !atir.tensor<6x8xf32, name = "flatten">
    %flat = atir.Reshape %fb, %x, %fs : <6x8xf32, name = "flatten">, <2x3x8xf32, name = "input">, <2xi32, name = "flat_shape", data = dense<[6, 8]> : tensor<2xi32>> -> <6x8xf32, name = "flatten">
    %mb = "atir.buffer"() : () -> !atir.tensor<6x4xf32, name = "matmul">
    %mm = "atir.MatMul"(%mb, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<6x4xf32, name = "matmul">, !atir.tensor<6x8xf32, name = "flatten">, !atir.tensor<8x4xf32, name = "weight", data = dense<0.25> : tensor<8x4xf32>>) -> !atir.tensor<6x4xf32, name = "matmul">
    %rs = atir.constant "public" @"restore_shape" -> <3xi32, name = "restore_shape", data = dense<[3, 2, 4]> : tensor<3xi32>>
    %rb = "atir.buffer"() : () -> !atir.tensor<3x2x4xf32, name = "restore">
    %result = atir.Reshape %rb, %mm, %rs : <3x2x4xf32, name = "restore">, <6x4xf32, name = "matmul">, <3xi32, name = "restore_shape", data = dense<[3, 2, 4]> : tensor<3xi32>> -> <3x2x4xf32, name = "restore">
    return %result : !atir.tensor<3x2x4xf32, name = "restore">
  }
}

// -----

// external_use
// CHECK-LABEL: func.func @main
// CHECK-NOT: fusion.pattern = "tensordot_matmul"
module attributes {module.state = "atir"} {
  func.func @main(%x: !atir.tensor<2x3x8xf32, name = "input">) -> (!atir.tensor<2x3x4xf32, name = "restore">, !atir.tensor<6x8xf32, name = "flatten">) {
    %w = atir.constant "public" @"weight" -> <8x4xf32, name = "weight", data = dense<0.25> : tensor<8x4xf32>>
    %fs = atir.constant "public" @"flat_shape" -> <2xi32, name = "flat_shape", data = dense<[6, 8]> : tensor<2xi32>>
    %fb = "atir.buffer"() : () -> !atir.tensor<6x8xf32, name = "flatten">
    %flat = atir.Reshape %fb, %x, %fs : <6x8xf32, name = "flatten">, <2x3x8xf32, name = "input">, <2xi32, name = "flat_shape", data = dense<[6, 8]> : tensor<2xi32>> -> <6x8xf32, name = "flatten">
    %mb = "atir.buffer"() : () -> !atir.tensor<6x4xf32, name = "matmul">
    %mm = "atir.MatMul"(%mb, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<6x4xf32, name = "matmul">, !atir.tensor<6x8xf32, name = "flatten">, !atir.tensor<8x4xf32, name = "weight", data = dense<0.25> : tensor<8x4xf32>>) -> !atir.tensor<6x4xf32, name = "matmul">
    %rs = atir.constant "public" @"restore_shape" -> <3xi32, name = "restore_shape", data = dense<[2, 3, 4]> : tensor<3xi32>>
    %rb = "atir.buffer"() : () -> !atir.tensor<2x3x4xf32, name = "restore">
    %result = atir.Reshape %rb, %mm, %rs : <2x3x4xf32, name = "restore">, <6x4xf32, name = "matmul">, <3xi32, name = "restore_shape", data = dense<[2, 3, 4]> : tensor<3xi32>> -> <2x3x4xf32, name = "restore">
    return %result, %flat : !atir.tensor<2x3x4xf32, name = "restore">, !atir.tensor<6x8xf32, name = "flatten">
  }
}

// -----

// transpose
// CHECK-LABEL: func.func @main
// CHECK-NOT: fusion.pattern = "tensordot_matmul"
module attributes {module.state = "atir"} {
  func.func @main(%x: !atir.tensor<2x3x8xf32, name = "input">) -> !atir.tensor<2x3x4xf32, name = "restore"> {
    %w = atir.constant "public" @"weight" -> <8x4xf32, name = "weight", data = dense<0.25> : tensor<8x4xf32>>
    %perm = atir.constant "public" @"perm" -> <3xi32, name = "perm", data = dense<[1, 0, 2]> : tensor<3xi32>>
    %tbuf = "atir.buffer"() : () -> !atir.tensor<3x2x8xf32, name = "transpose">
    %t = atir.Transpose %tbuf, %x, %perm {permutation = [1, 0, 2]} : <3x2x8xf32, name = "transpose">, <2x3x8xf32, name = "input">, <3xi32, name = "perm", data = dense<[1, 0, 2]> : tensor<3xi32>> -> <3x2x8xf32, name = "transpose">
    %fs = atir.constant "public" @"flat_shape" -> <2xi32, name = "flat_shape", data = dense<[6, 8]> : tensor<2xi32>>
    %fb = "atir.buffer"() : () -> !atir.tensor<6x8xf32, name = "flatten">
    %flat = atir.Reshape %fb, %t, %fs : <6x8xf32, name = "flatten">, <3x2x8xf32, name = "transpose">, <2xi32, name = "flat_shape", data = dense<[6, 8]> : tensor<2xi32>> -> <6x8xf32, name = "flatten">
    %mb = "atir.buffer"() : () -> !atir.tensor<6x4xf32, name = "matmul">
    %mm = "atir.MatMul"(%mb, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<6x4xf32, name = "matmul">, !atir.tensor<6x8xf32, name = "flatten">, !atir.tensor<8x4xf32, name = "weight", data = dense<0.25> : tensor<8x4xf32>>) -> !atir.tensor<6x4xf32, name = "matmul">
    %rs = atir.constant "public" @"restore_shape" -> <3xi32, name = "restore_shape", data = dense<[2, 3, 4]> : tensor<3xi32>>
    %rb = "atir.buffer"() : () -> !atir.tensor<2x3x4xf32, name = "restore">
    %result = atir.Reshape %rb, %mm, %rs : <2x3x4xf32, name = "restore">, <6x4xf32, name = "matmul">, <3xi32, name = "restore_shape", data = dense<[2, 3, 4]> : tensor<3xi32>> -> <2x3x4xf32, name = "restore">
    return %result : !atir.tensor<2x3x4xf32, name = "restore">
  }
}
