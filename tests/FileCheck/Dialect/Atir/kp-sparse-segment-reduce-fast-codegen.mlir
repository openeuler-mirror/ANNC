// RUN: annc-asm --split-input-file %s -atir-fast-codegen | FileCheck %s
// KP-style sparse segment reduce fusion, second level, Execution V2: inside
// the kernel func extracted by OpFusion (signature !llvm.ptr + 5 inputs,
// 2 results), the StridedSlice->SparseSegmentMean->Shape->StridedSlice
// subgraph folds into a single KPFusedSparseSegmentReduceI64Mean CustomizeOp
// whose two results map to execution slots 0/1.

// CHECK-LABEL: func.func private @fused_kp_sparse_segment_reduce
// CHECK-SAME: %arg0: !llvm.ptr,
// CHECK-SAME: %arg1: !atir.tensor<4x2xi64, name = "keys">, %arg2: !atir.tensor<2xi32, name = "begin">, %arg3: !atir.tensor<6x3xf32, name = "data">, %arg4: !atir.tensor<4xi64, name = "indices">, %arg5: !atir.tensor<1xi32, name = "begin_1">
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "KPFusedSparseSegmentReduceI64Mean"
// CHECK-SAME: custom.result_names = ["output", "slice_output"]
// CHECK: return {{%.*}}, {{%.*}} : !atir.tensor<?x3xf32, name = "output">, !atir.tensor<i32, name = "slice_output">
// CHECK-NOT: atir.SparseSegmentMean
// CHECK-NOT: atir.Shape
// CHECK-NOT: atir.StridedSlice
module attributes {module.state = "atir"} {
  func.func private @fused_kp_sparse_segment_reduce(
      %arg0: !llvm.ptr,
      %arg1: !atir.tensor<4x2xi64, name = "keys">,
      %arg2: !atir.tensor<2xi32, name = "begin">,
      %arg3: !atir.tensor<6x3xf32, name = "data">,
      %arg4: !atir.tensor<4xi64, name = "indices">,
      %arg5: !atir.tensor<1xi32, name = "begin_1">)
      -> (!atir.tensor<?x3xf32, name = "output">, !atir.tensor<i32, name = "slice_output">)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_execution_v2"}} {
    %0 = "atir.buffer"() : () -> !atir.tensor<?xi64, name = "seg_ids">
    %1 = atir.constant "public" @"ss_end" -> <2xi32, name = "ss_end", data = dense<[0, 2]> : tensor<2xi32>>
    %2 = atir.constant "public" @"ss_strides" -> <2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>
    %3 = "atir.StridedSlice"(%0, %arg1, %arg2, %1, %2) <{beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32}> : (!atir.tensor<?xi64, name = "seg_ids">, !atir.tensor<4x2xi64, name = "keys">, !atir.tensor<2xi32, name = "begin">, !atir.tensor<2xi32, name = "ss_end", data = dense<[0, 2]> : tensor<2xi32>>, !atir.tensor<2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>) -> !atir.tensor<?xi64, name = "seg_ids">
    %4 = atir.constant "public" @"num_segments" -> <i32, name = "num_segments", data = dense<0> : tensor<i32>>
    %13 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "output_b">
    %5 = "atir.SparseSegmentMean"(%13, %arg3, %arg4, %3, %4) : (!atir.tensor<?x3xf32, name = "output_b">, !atir.tensor<6x3xf32, name = "data">, !atir.tensor<4xi64, name = "indices">, !atir.tensor<?xi64, name = "seg_ids">, !atir.tensor<i32, name = "num_segments", data = dense<0> : tensor<i32>>) -> !atir.tensor<?x3xf32, name = "output">
    %6 = atir.Shape %5 : (!atir.tensor<?x3xf32, name = "output">) -> !atir.tensor<2xi32, name = "output_shape">
    %7 = atir.constant "public" @"anchor_end" -> <1xi32, name = "anchor_end", data = dense<1> : tensor<1xi32>>
    %8 = atir.constant "public" @"anchor_strides" -> <1xi32, name = "anchor_strides", data = dense<1> : tensor<1xi32>>
    %14 = "atir.buffer"() : () -> !atir.tensor<i32, name = "slice_output_b">
    %9 = "atir.StridedSlice"(%14, %6, %arg5, %7, %8) <{beginMask = 0 : i32, endMask = 0 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 1 : i32}> : (!atir.tensor<i32, name = "slice_output_b">, !atir.tensor<2xi32, name = "output_shape">, !atir.tensor<1xi32, name = "begin_1">, !atir.tensor<1xi32, name = "anchor_end", data = dense<1> : tensor<1xi32>>, !atir.tensor<1xi32, name = "anchor_strides", data = dense<1> : tensor<1xi32>>) -> !atir.tensor<i32, name = "slice_output">
    return %5, %9 : !atir.tensor<?x3xf32, name = "output">, !atir.tensor<i32, name = "slice_output">
  }
}

// -----

// Negative case: keys StridedSlice shrink_axis_mask=1 (not 2) — reject.
// CHECK-LABEL: func.func private @bad_mask
// CHECK-NOT: atir.Customize
// CHECK: atir.SparseSegmentMean
module attributes {module.state = "atir"} {
  func.func private @bad_mask(
      %arg0: !llvm.ptr,
      %arg1: !atir.tensor<4x2xi64, name = "keys_t">,
      %arg2: !atir.tensor<2xi32, name = "begin_t">,
      %arg3: !atir.tensor<6x3xf32, name = "data_t">,
      %arg4: !atir.tensor<4xi64, name = "indices_t">,
      %arg5: !atir.tensor<1xi32, name = "begin_1_t">)
      -> (!atir.tensor<?x3xf32, name = "output_t">, !atir.tensor<i32, name = "slice_output_t">)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_execution_v2"}} {
    %0 = "atir.buffer"() : () -> !atir.tensor<?xi64, name = "seg_ids_t">
    %1 = atir.constant "public" @"ss_end_t" -> <2xi32, name = "ss_end_t", data = dense<[0, 2]> : tensor<2xi32>>
    %2 = atir.constant "public" @"ss_strides_t" -> <2xi32, name = "ss_strides_t", data = dense<[1, 1]> : tensor<2xi32>>
    %3 = "atir.StridedSlice"(%0, %arg1, %arg2, %1, %2) <{beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 1 : i32}> : (!atir.tensor<?xi64, name = "seg_ids_t">, !atir.tensor<4x2xi64, name = "keys_t">, !atir.tensor<2xi32, name = "begin_t">, !atir.tensor<2xi32, name = "ss_end_t", data = dense<[0, 2]> : tensor<2xi32>>, !atir.tensor<2xi32, name = "ss_strides_t", data = dense<[1, 1]> : tensor<2xi32>>) -> !atir.tensor<?xi64, name = "seg_ids_t">
    %4 = atir.constant "public" @"num_segments_t" -> <i32, name = "num_segments_t", data = dense<0> : tensor<i32>>
    %13 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "output_tb">
    %5 = "atir.SparseSegmentMean"(%13, %arg3, %arg4, %3, %4) : (!atir.tensor<?x3xf32, name = "output_tb">, !atir.tensor<6x3xf32, name = "data_t">, !atir.tensor<4xi64, name = "indices_t">, !atir.tensor<?xi64, name = "seg_ids_t">, !atir.tensor<i32, name = "num_segments_t", data = dense<0> : tensor<i32>>) -> !atir.tensor<?x3xf32, name = "output_t">
    %6 = atir.Shape %5 : (!atir.tensor<?x3xf32, name = "output_t">) -> !atir.tensor<2xi32, name = "output_shape_t">
    %7 = atir.constant "public" @"anchor_end_t" -> <1xi32, name = "anchor_end_t", data = dense<1> : tensor<1xi32>>
    %8 = atir.constant "public" @"anchor_strides_t" -> <1xi32, name = "anchor_strides_t", data = dense<1> : tensor<1xi32>>
    %14 = "atir.buffer"() : () -> !atir.tensor<i32, name = "slice_output_tb">
    %9 = "atir.StridedSlice"(%14, %6, %arg5, %7, %8) <{beginMask = 0 : i32, endMask = 0 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 1 : i32}> : (!atir.tensor<i32, name = "slice_output_tb">, !atir.tensor<2xi32, name = "output_shape_t">, !atir.tensor<1xi32, name = "begin_1_t">, !atir.tensor<1xi32, name = "anchor_end_t", data = dense<1> : tensor<1xi32>>, !atir.tensor<1xi32, name = "anchor_strides_t", data = dense<1> : tensor<1xi32>>) -> !atir.tensor<i32, name = "slice_output_t">
    return %5, %9 : !atir.tensor<?x3xf32, name = "output_t">, !atir.tensor<i32, name = "slice_output_t">
  }
}

// -----
// Cast wrapper: Cast(i32) ← StridedSlice(i64) ← keys, Cast inside the kernel
// func.  The fast-codegen rewrite absorbs the Cast into the CustomizeOp.
// CHECK-LABEL: func.func private @fused_ssr_cast
// CHECK-SAME: %arg0: !llvm.ptr,
// CHECK-SAME: %arg1: !atir.tensor<4x2xi64, name = "keys">, %arg2: !atir.tensor<2xi32, name = "begin">, %arg3: !atir.tensor<6x3xf32, name = "data">, %arg4: !atir.tensor<4xi64, name = "indices">, %arg5: !atir.tensor<1xi32, name = "begin_1">
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "KPFusedSparseSegmentReduceI64Mean"
// CHECK-SAME: custom.result_names = ["output", "slice_output"]
// CHECK: return {{%.*}}, {{%.*}} : !atir.tensor<?x3xf32, name = "output">, !atir.tensor<i32, name = "slice_output">
// CHECK-NOT: atir.SparseSegmentMean
// CHECK-NOT: atir.Shape
// CHECK-NOT: atir.StridedSlice
module attributes {module.state = "atir"} {
  func.func private @fused_ssr_cast(
      %arg0: !llvm.ptr,
      %arg1: !atir.tensor<4x2xi64, name = "keys">,
      %arg2: !atir.tensor<2xi32, name = "begin">,
      %arg3: !atir.tensor<6x3xf32, name = "data">,
      %arg4: !atir.tensor<4xi64, name = "indices">,
      %arg5: !atir.tensor<1xi32, name = "begin_1">)
      -> (!atir.tensor<?x3xf32, name = "output">, !atir.tensor<i32, name = "slice_output">)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_execution_v2"}} {
    %0 = "atir.buffer"() : () -> !atir.tensor<?xi64, name = "seg_ids">
    %1 = atir.constant "public" @"ss_end" -> <2xi32, name = "ss_end", data = dense<[0, 2]> : tensor<2xi32>>
    %2 = atir.constant "public" @"ss_strides" -> <2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>
    %3 = "atir.StridedSlice"(%0, %arg1, %arg2, %1, %2) <{beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32}> : (!atir.tensor<?xi64, name = "seg_ids">, !atir.tensor<4x2xi64, name = "keys">, !atir.tensor<2xi32, name = "begin">, !atir.tensor<2xi32, name = "ss_end", data = dense<[0, 2]> : tensor<2xi32>>, !atir.tensor<2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>) -> !atir.tensor<?xi64, name = "seg_ids">
    %cast_buf = "atir.buffer"() : () -> !atir.tensor<?xi32, name = "cast_seg_ids">
    %cast = "atir.Cast"(%cast_buf, %3) : (!atir.tensor<?xi32, name = "cast_seg_ids">, !atir.tensor<?xi64, name = "seg_ids">) -> !atir.tensor<?xi32, name = "cast_seg_ids">
    %4 = atir.constant "public" @"num_segments" -> <i32, name = "num_segments", data = dense<0> : tensor<i32>>
    %13 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "output_b">
    %5 = "atir.SparseSegmentMean"(%13, %arg3, %arg4, %cast, %4) : (!atir.tensor<?x3xf32, name = "output_b">, !atir.tensor<6x3xf32, name = "data">, !atir.tensor<4xi64, name = "indices">, !atir.tensor<?xi32, name = "cast_seg_ids">, !atir.tensor<i32, name = "num_segments", data = dense<0> : tensor<i32>>) -> !atir.tensor<?x3xf32, name = "output">
    %6 = atir.Shape %5 : (!atir.tensor<?x3xf32, name = "output">) -> !atir.tensor<2xi32, name = "output_shape">
    %7 = atir.constant "public" @"anchor_end" -> <1xi32, name = "anchor_end", data = dense<1> : tensor<1xi32>>
    %8 = atir.constant "public" @"anchor_strides" -> <1xi32, name = "anchor_strides", data = dense<1> : tensor<1xi32>>
    %14 = "atir.buffer"() : () -> !atir.tensor<i32, name = "slice_output_b">
    %9 = "atir.StridedSlice"(%14, %6, %arg5, %7, %8) <{beginMask = 0 : i32, endMask = 0 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 1 : i32}> : (!atir.tensor<i32, name = "slice_output_b">, !atir.tensor<2xi32, name = "output_shape">, !atir.tensor<1xi32, name = "begin_1">, !atir.tensor<1xi32, name = "anchor_end", data = dense<1> : tensor<1xi32>>, !atir.tensor<1xi32, name = "anchor_strides", data = dense<1> : tensor<1xi32>>) -> !atir.tensor<i32, name = "slice_output">
    return %5, %9 : !atir.tensor<?x3xf32, name = "output">, !atir.tensor<i32, name = "slice_output">
  }
}
