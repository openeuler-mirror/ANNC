// RUN: annc-stablehlo-sink --help >/dev/null && annc-asm %s --shape-legalize-to-stablehlo --reconcile-unrealized-casts --annc-select-kernel='kernel-name=fused_matmul_add' --annc-specialize-shapes='shape-spec=%S/Inputs/annc-runtime-shapes.json' --verify-each | FileCheck %s

module {
  func.func private @fused_matmul_add(
      %arg0: tensor<?x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>)
      -> tensor<?x3xf32> attributes {annc.kernel} {
    %0 = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT]
        : (tensor<?x4xf32>, tensor<4x3xf32>) -> tensor<?x3xf32>
    %1 = shape.shape_of %0 : tensor<?x3xf32> -> tensor<2xindex>
    %2 = shape.const_shape [3] : tensor<1xindex>
    %3 = shape.broadcast %1, %2 : tensor<2xindex>, tensor<1xindex>
        -> tensor<2xindex>
    %4 = arith.index_cast %3 : tensor<2xindex> to tensor<2xi32>
    %5 = stablehlo.dynamic_broadcast_in_dim %arg2, %4, dims = [1]
        : (tensor<3xf32>, tensor<2xi32>) -> tensor<?x3xf32>
    %6 = stablehlo.add %0, %5 : tensor<?x3xf32>
    return %6 : tensor<?x3xf32>
  }
}

// CHECK: func.func private @fused_matmul_add(%arg0: tensor<5x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>) -> tensor<5x3xf32>
// CHECK: stablehlo.dot {{.*}} -> tensor<5x3xf32>
// CHECK: stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<3xf32>) -> tensor<5x3xf32>
// CHECK: stablehlo.add {{.*}} : tensor<5x3xf32>
// CHECK-NOT: stablehlo.dynamic_broadcast_in_dim
