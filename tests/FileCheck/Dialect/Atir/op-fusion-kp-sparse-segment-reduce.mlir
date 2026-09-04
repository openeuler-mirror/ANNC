// RUN: annc-opt --split-input-file %s -atir-config-fusion="config=%S/../../../../configs/fusion/kp_embedding_patterns.json run-builtin-after=false warn-limit=0" | FileCheck %s --check-prefix=CONFIG
// Config fusion first level for sparse segment reduce, Execution V2: the
// subgraph in main is extracted into a private kernel func with signature
// (!llvm.ptr, keys, begin, data, indices, begin_1) -> (output, slice_output),
// annc.kernel + fusion.metadata abi = "annc_execution_v2".

// CONFIG: func.func private @fused_kp_sparse_segment_reduce_
// CONFIG-SAME: %arg0: !llvm.ptr
// CONFIG-SAME: annc.kernel
// CONFIG-SAME: abi = "annc_execution_v2"
// CONFIG: func.func @main
// CONFIG: annc.fusion_materialized

// CHECK-LABEL: func.func @main
// CHECK: atir.StridedSlice
// CHECK: atir.StridedSlice
// CHECK-SAME: annc.fusion_materialized
// CHECK-LABEL: func.func private @fused_kp_sparse_segment_reduce_
// CHECK-SAME: %arg0: !llvm.ptr,
// CHECK-SAME: %arg1: !atir.tensor<4x2xi64, name = "keys">, %arg2: !atir.tensor<2xi32, name = "begin">, %arg3: !atir.tensor<6x3xf32, name = "data">, %arg4: !atir.tensor<4xi64, name = "indices">, %arg5: !atir.tensor<1xi32, name = "begin_1">
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_execution_v2"
// CHECK-SAME: kernel_name = "fused_kp_sparse_segment_reduce_
// CHECK-SAME: llvm.emit_c_interface
module attributes {module.state = "atir"} {
func.func @main(
    %arg0: !atir.tensor<6x3xf32, name = "data">,
    %arg1: !atir.tensor<4xi64, name = "indices">,
    %arg2: !atir.tensor<4x2xi64, name = "keys">,
    %arg3: !atir.tensor<2xi32, name = "begin">,
    %arg4: !atir.tensor<1xi32, name = "begin_1">)
    -> (!atir.tensor<?x3xf32, name = "output">, !atir.tensor<i32, name = "slice_output">) {
    %0 = "atir.buffer"() : () -> !atir.tensor<?xi64, name = "seg_ids">
    %1 = atir.constant "public" @"ss_end" -> <2xi32, name = "ss_end", data = dense<[0, 2]> : tensor<2xi32>>
    %2 = atir.constant "public" @"ss_strides" -> <2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>
    %3 = "atir.StridedSlice"(%0, %arg2, %arg3, %1, %2) <{beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32}> : (!atir.tensor<?xi64, name = "seg_ids">, !atir.tensor<4x2xi64, name = "keys">, !atir.tensor<2xi32, name = "begin">, !atir.tensor<2xi32, name = "ss_end", data = dense<[0, 2]> : tensor<2xi32>>, !atir.tensor<2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>) -> !atir.tensor<?xi64, name = "seg_ids">
    %4 = atir.constant "public" @"num_segments" -> <i32, name = "num_segments", data = dense<0> : tensor<i32>>
    %10 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "output">
    %5 = "atir.SparseSegmentMean"(%10, %arg0, %arg1, %3, %4) : (!atir.tensor<?x3xf32, name = "output">, !atir.tensor<6x3xf32, name = "data">, !atir.tensor<4xi64, name = "indices">, !atir.tensor<?xi64, name = "seg_ids">, !atir.tensor<i32, name = "num_segments", data = dense<0> : tensor<i32>>) -> !atir.tensor<?x3xf32, name = "output">
    %6 = atir.Shape %5 : (!atir.tensor<?x3xf32, name = "output">) -> !atir.tensor<2xi32, name = "output_shape">
    %7 = atir.constant "public" @"anchor_end" -> <1xi32, name = "anchor_end", data = dense<1> : tensor<1xi32>>
    %8 = atir.constant "public" @"anchor_strides" -> <1xi32, name = "anchor_strides", data = dense<1> : tensor<1xi32>>
    %11 = "atir.buffer"() : () -> !atir.tensor<i32, name = "slice_output">
    %9 = "atir.StridedSlice"(%11, %6, %arg4, %7, %8) <{beginMask = 0 : i32, endMask = 0 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 1 : i32}> : (!atir.tensor<i32, name = "slice_output">, !atir.tensor<2xi32, name = "output_shape">, !atir.tensor<1xi32, name = "begin_1">, !atir.tensor<1xi32, name = "anchor_end", data = dense<1> : tensor<1xi32>>, !atir.tensor<1xi32, name = "anchor_strides", data = dense<1> : tensor<1xi32>>) -> !atir.tensor<i32, name = "slice_output">
    return %5, %9 : !atir.tensor<?x3xf32, name = "output">, !atir.tensor<i32, name = "slice_output">
  }
}
