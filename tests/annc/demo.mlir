module @demo {
  func.func private @fused_matmul(%arg0: memref<1536x1344xf32>, %arg1: memref<1344x1152xf32>, %arg2: memref<1536x1152xf32>, %arg3: memref<1152xf32>) -> memref<1536x1152xf32> attributes {llvm.emit_c_interface} {
    %subview = memref.subview %arg3[0] [1152] [1] : memref<1152xf32> to memref<1152xf32, strided<[1]>>
    %subview_0 = memref.subview %arg0[0, 0] [1536, 1344] [1, 1] : memref<1536x1344xf32> to memref<1536x1344xf32, strided<[1344, 1]>>
    %subview_1 = memref.subview %arg1[0, 0] [1344, 1152] [1, 1] : memref<1344x1152xf32> to memref<1344x1152xf32, strided<[1152, 1]>>
    %subview_2 = memref.subview %arg2[0, 0] [1536, 1152] [1, 1] : memref<1536x1152xf32> to memref<1536x1152xf32, strided<[1152, 1]>>
    affine.for %arg4 = 0 to 1536 {
      affine.for %arg5 = 0 to 1152 {
        %0 = affine.load %subview[%arg5] : memref<1152xf32, strided<[1]>>
        affine.for %arg6 = 0 to 1344 {
          %6 = affine.load %subview_0[%arg4, %arg6] : memref<1536x1344xf32, strided<[1344, 1]>>
          %7 = affine.load %subview_1[%arg6, %arg5] : memref<1344x1152xf32, strided<[1152, 1]>>
          %8 = arith.mulf %6, %7 : f32
          %9 = affine.load %subview_2[%arg4, %arg5] : memref<1536x1152xf32, strided<[1152, 1]>>
          %10 = arith.addf %9, %8 : f32
          affine.store %10, %subview_2[%arg4, %arg5] : memref<1536x1152xf32, strided<[1152, 1]>>
        }
        %1 = affine.load %subview_2[%arg4, %arg5] : memref<1536x1152xf32, strided<[1152, 1]>>
        %2 = arith.addf %1, %0 : f32
        affine.store %2, %subview_2[%arg4, %arg5] : memref<1536x1152xf32, strided<[1152, 1]>>
        %3 = affine.load %subview_2[%arg4, %arg5] : memref<1536x1152xf32, strided<[1152, 1]>>
        %cst = arith.constant 0.000000e+00 : f32
        %4 = arith.cmpf olt, %3, %cst : f32
        %5 = arith.select %4, %cst, %3 : f32
        affine.store %5, %subview_2[%arg4, %arg5] : memref<1536x1152xf32, strided<[1152, 1]>>
      }
    }
    return %arg2 : memref<1536x1152xf32>
  }
}