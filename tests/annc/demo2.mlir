module @demo attributes {module.state = "atir"} {
  func.func private @fused_matmul_add_relu_A0732AD9DB33D09F(%arg0: memref<4x8xf32>, %arg1: memref<8x2xf32>, %arg2: memref<4x2xf32>, %arg3: memref<2xf32>) -> memref<4x2xf32> attributes {fusion.pattern = "matmul_add_relu", llvm.emit_c_interface} {
    %subview = memref.subview %arg3[0] [2] [1] : memref<2xf32> to memref<2xf32, strided<[1]>>
    %subview_0 = memref.subview %arg0[0, 0] [4, 8] [1, 1] : memref<4x8xf32> to memref<4x8xf32, strided<[8, 1]>>
    %subview_1 = memref.subview %arg1[0, 0] [8, 2] [1, 1] : memref<8x2xf32> to memref<8x2xf32, strided<[2, 1]>>
    %subview_2 = memref.subview %arg2[0, 0] [4, 2] [1, 1] : memref<4x2xf32> to memref<4x2xf32, strided<[2, 1]>>
    affine.for %arg4 = 0 to 4 {
      affine.for %arg5 = 0 to 2 {
        %0 = affine.load %subview[%arg5] : memref<2xf32, strided<[1]>>
        affine.for %arg6 = 0 to 8 {
          %6 = affine.load %subview_0[%arg4, %arg6] : memref<4x8xf32, strided<[8, 1]>>
          %7 = affine.load %subview_1[%arg6, %arg5] : memref<8x2xf32, strided<[2, 1]>>
          %8 = arith.mulf %6, %7 : f32
          %9 = affine.load %subview_2[%arg4, %arg5] : memref<4x2xf32, strided<[2, 1]>>
          %10 = arith.addf %9, %8 : f32
          affine.store %10, %subview_2[%arg4, %arg5] : memref<4x2xf32, strided<[2, 1]>>
        }
        %1 = affine.load %subview_2[%arg4, %arg5] : memref<4x2xf32, strided<[2, 1]>>
        %2 = arith.addf %1, %0 : f32
        affine.store %2, %subview_2[%arg4, %arg5] : memref<4x2xf32, strided<[2, 1]>>
        %3 = affine.load %subview_2[%arg4, %arg5] : memref<4x2xf32, strided<[2, 1]>>
        %cst = arith.constant -1.000000e+00 : f32
        %4 = arith.cmpf olt, %3, %cst : f32
        %5 = arith.select %4, %cst, %3 : f32
        affine.store %5, %subview_2[%arg4, %arg5] : memref<4x2xf32, strided<[2, 1]>>
      }
    }
    return %arg2 : memref<4x2xf32>
  }
}
