// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 annc-asm %s --annc-one-shot-bufferize --verify-each | FileCheck %s --implicit-check-not='memref.alloc('
// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 annc-asm %s --annc-one-shot-bufferize --verify-each | mlir-opt --buffer-deallocation-pipeline --verify-each | FileCheck %s --check-prefix=PIPELINE --implicit-check-not=memref.dealloc

module {
  // PIPELINE-LABEL: func.func @context_outputs(
  // PIPELINE: call @annc_context_alloc_output_memref
  // PIPELINE: return
  // PIPELINE-LABEL: func.func @dynamic_output(
  // PIPELINE: call @annc_context_alloc_output_memref
  // PIPELINE: return
  // CHECK: func.func private @annc_context_alloc_output_memref(!llvm.ptr, i32, i32, !llvm.ptr) -> !llvm.ptr
  func.func private @external_custom(!llvm.ptr) -> i32

  // CHECK-LABEL: func.func @context_outputs(
  // CHECK-SAME: !llvm.ptr) attributes
  // CHECK: llvm.mlir.constant(0 : i64)
  // CHECK: call @annc_context_alloc_output_memref(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, i32, i32, !llvm.ptr) -> !llvm.ptr
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<{{.*}}> to memref<2x3xf32>
  // CHECK: %[[TEMP:.*]] = memref.alloca()
  // CHECK: llvm.mlir.constant(1 : i64)
  // CHECK: call @annc_context_alloc_output_memref(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, i32, i32, !llvm.ptr) -> !llvm.ptr
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<{{.*}}> to memref<2x3xf32>
  // CHECK-NOT: memref.alloc()
  // CHECK: linalg.generic
  // CHECK: outs(%{{.*}} : memref<2x3xf32>)
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.cast
  // CHECK-NOT: strided<
  // CHECK-NOT: cf.cond_br
  // CHECK: return{{$}}
  func.func @context_outputs(%context: !llvm.ptr)
      -> (memref<2x3xf32>, memref<2x3xf32>)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_context_v3"}} {
    %zero = arith.constant 0.0 : f32
    %empty0 = tensor.empty() : tensor<2x3xf32>
    %result0 = linalg.fill ins(%zero : f32) outs(%empty0 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %tmp = tensor.empty() : tensor<2x3xf32>
    %intermediate = linalg.fill ins(%zero : f32) outs(%tmp : tensor<2x3xf32>) -> tensor<2x3xf32>
    %empty1 = tensor.empty() : tensor<2x3xf32>
    %result1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%intermediate : tensor<2x3xf32>) outs(%empty1 : tensor<2x3xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    } -> tensor<2x3xf32>
    %out0 = bufferization.to_buffer %result0 : tensor<2x3xf32> to memref<2x3xf32>
    %out1 = bufferization.to_buffer %result1 : tensor<2x3xf32> to memref<2x3xf32>
    return %out0, %out1 : memref<2x3xf32>, memref<2x3xf32>
  }

  // CHECK-LABEL: func.func @dynamic_output(
  // CHECK: %[[DIM:.*]] = arith.index_cast %{{.*}} : index to i64
  // CHECK: llvm.store %[[DIM]], %{{.*}} : i64, !llvm.ptr
  // CHECK: call @annc_context_alloc_output_memref(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, i32, i32, !llvm.ptr) -> !llvm.ptr
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<{{.*}}> to memref<?x3xf32>
  // CHECK: return{{$}}
  func.func @dynamic_output(%context: !llvm.ptr) -> memref<?x3xf32>
      attributes {annc.kernel, fusion.metadata = {abi = "annc_context_v3"}} {
    %n = arith.constant 2 : index
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty(%n) : tensor<?x3xf32>
    %result = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x3xf32>) -> tensor<?x3xf32>
    %out = bufferization.to_buffer %result : tensor<?x3xf32> to memref<?x3xf32>
    return %out : memref<?x3xf32>
  }

  // CHECK-LABEL: func.func @void_with_temporary(
  // CHECK-SAME: !llvm.ptr) attributes
  // CHECK: memref.alloca()
  // CHECK: return{{$}}
  func.func @void_with_temporary(%context: !llvm.ptr)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_context_v3"}} {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<2x3xf32>
    %unused = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x3xf32>) -> tensor<2x3xf32>
    return
  }

  // CHECK-LABEL: func.func @nonreturned_alloc(
  // CHECK: memref.alloca() : memref<2x3xf32>
  // CHECK: return{{$}}
  func.func @nonreturned_alloc(%context: !llvm.ptr)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_context_v3"}} {
    %tmp = memref.alloc() : memref<2x3xf32>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%tmp : memref<2x3xf32>)
    return
  }

  // CHECK-LABEL: func.func @customize_status(
  // CHECK-SAME: !llvm.ptr) -> i32
  // CHECK: call @external_custom
  // CHECK: return %{{.*}} : i32
  func.func @customize_status(%context: !llvm.ptr) -> i32
      attributes {annc.kernel, fusion.metadata = {abi = "annc_context_v3"}} {
    %status = call @external_custom(%context) : (!llvm.ptr) -> i32
    return %status : i32
  }
}
