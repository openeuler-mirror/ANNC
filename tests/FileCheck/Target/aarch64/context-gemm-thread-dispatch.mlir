// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 annc-asm %s --aarch64-gemm-thread-tiling --verify-each | FileCheck %s --check-prefix=CONTEXT --implicit-check-not=annc_threadpool_parallel_for_gemm
// RUN: env -u ANNC_ENABLE_LOWERING_CONTEXT_ABI annc-asm %s --aarch64-gemm-thread-tiling --verify-each | FileCheck %s --check-prefix=LEGACY --implicit-check-not=annc_context_parallel_for_gemm
// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 annc-asm %s --aarch64-gemm-thread-tiling --verify-each | mlir-opt --convert-linalg-to-loops --lower-affine --convert-scf-to-cf --expand-strided-metadata --lower-affine --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not=builtin.unrealized_conversion_cast
// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 not annc-asm %S/gemm-thread-task-id.mlir --aarch64-gemm-thread-tiling 2>&1 | FileCheck %s --check-prefix=INVALID
// INVALID: context GEMM dispatch requires one context pointer argument

// CONTEXT-DAG: func.func private @annc_context_parallel_for_gemm(!llvm.ptr, i64, (i64, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> (), memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) attributes {llvm.emit_c_interface}
// CONTEXT-DAG: func.func private @annc_context_parallel_for_gemm_epilogue(!llvm.ptr, i64,
// CONTEXT-LABEL: func.func @context_gemm(
// CONTEXT-SAME: %[[CONTEXT:.*]]: !llvm.ptr)
// CONTEXT: %[[TASK:.*]] = constant @__annc_gemm_thread_task_0 : (i64, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
// CONTEXT: %[[COUNT:.*]] = arith.constant 4 : i64
// CONTEXT: call @annc_context_parallel_for_gemm(%[[CONTEXT]], %[[COUNT]], %[[TASK]],
// CONTEXT-LABEL: func.func @context_epilogue(
// CONTEXT-SAME: %[[EP_CONTEXT:.*]]: !llvm.ptr)
// CONTEXT: %[[EP_TASK:.*]] = constant @__annc_gemm_thread_task_1
// CONTEXT: %[[EP_COUNT:.*]] = arith.constant 4 : i64
// CONTEXT: call @annc_context_parallel_for_gemm_epilogue(%[[EP_CONTEXT]], %[[EP_COUNT]], %[[EP_TASK]],
// CONTEXT-LABEL: func.func private @__annc_gemm_thread_task_0(
// CONTEXT-SAME: %{{.*}}: i64, %{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>)
// CONTEXT-LABEL: func.func private @__annc_gemm_thread_task_1(
// CONTEXT-SAME: %{{.*}}: i64, %{{.*}}: memref<?x?xf32>
// CONTEXT-SAME: memref<*xf32>
// LEGACY-LABEL: func.func @context_gemm(
// LEGACY: call @annc_threadpool_parallel_for_gemm(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64,
// LEGACY-LABEL: func.func @context_epilogue(
// LEGACY: call @annc_threadpool_parallel_for_gemm_epilogue(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}},
// LLVM-DAG: llvm.func @_mlir_ciface_annc_context_parallel_for_gemm(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// LLVM-DAG: llvm.func @_mlir_ciface_annc_context_parallel_for_gemm_epilogue(!llvm.ptr, i64,
// LLVM-LABEL: llvm.func @context_gemm(
// LLVM-SAME: %[[LLVM_CONTEXT:.*]]: !llvm.ptr)
// LLVM: llvm.call @annc_context_parallel_for_gemm(%[[LLVM_CONTEXT]],
// LLVM-LABEL: llvm.func @context_epilogue(
// LLVM-SAME: %[[LLVM_EP_CONTEXT:.*]]: !llvm.ptr)
// LLVM: llvm.call @annc_context_parallel_for_gemm_epilogue(%[[LLVM_EP_CONTEXT]],

func.func @context_gemm(
    %context: !llvm.ptr) attributes {annc.kernel, fusion.metadata = {abi = "annc_context_v3"}, annc.aarch64.gemm_strategies = {s0 = {version = 4 : i64, target_arch = "hip12", isa = "sve", data_type = "f32", kernel_family = "annc-sve-f32-v1", cache_strategy = "fixed", kernel_strategy = "fixed", cache_parameters = {mc = 12 : i64, nc = 40 : i64, kc = 8 : i64}, kernel_parameters = {mr = 3 : i64, panel_lanes = 2 : i64}, execution_kind = "gemm", rhs_packing = "packed"}}} {
  %a = memref.alloca() : memref<13x8xf32>
  %b = memref.alloca() : memref<8x40xf32>
  %c = memref.alloca() : memref<13x40xf32>
  linalg.matmul {
    annc.aarch64.gemm_problem = {version = 1 : i64, m = 13 : i64, n = 40 : i64, k = 8 : i64, lda = 8 : i64, ldb = 40 : i64, ldc = 40 : i64}, annc.aarch64.gemm_strategy = "s0", annc.aarch64.gemm_candidate = {version = 4 : i64, planning_state = "planned", strategy = "s0", max_thread_count = 4 : i64, thread_count = 4 : i64, tasks_m = 2 : i64, tasks_n = 2 : i64, shard_direction = "rows", thread_partition = "static-2d"}
  } ins(%a, %b : memref<13x8xf32>, memref<8x40xf32>)
    outs(%c : memref<13x40xf32>)
  return
}

func.func @context_epilogue(%context: !llvm.ptr) attributes {annc.kernel, fusion.metadata = {abi = "annc_context_v3"}, annc.aarch64.gemm_strategies = {s0 = {version = 4 : i64, target_arch = "hip12", isa = "sve", data_type = "f32", kernel_family = "annc-sve-f32-v1", cache_strategy = "fixed", kernel_strategy = "fixed", cache_parameters = {mc = 12 : i64, nc = 40 : i64, kc = 8 : i64}, kernel_parameters = {mr = 3 : i64, panel_lanes = 2 : i64}, execution_kind = "gemm", rhs_packing = "packed"}}} {
  %a = memref.alloca() : memref<13x8xf32>
  %b = memref.alloca() : memref<8x40xf32>
  %c = memref.alloca() : memref<13x40xf32>
  %bias = memref.alloca() : memref<40xf32>
  linalg.generic {
    annc.gemm,
    annc.aarch64.gemm_problem = {version = 1 : i64, m = 13 : i64, n = 40 : i64, k = 8 : i64, lda = 8 : i64, ldb = 40 : i64, ldc = 40 : i64}, annc.aarch64.gemm_strategy = "s0", annc.aarch64.gemm_candidate = {version = 4 : i64, planning_state = "planned", strategy = "s0", max_thread_count = 4 : i64, thread_count = 4 : i64, tasks_m = 2 : i64, tasks_n = 2 : i64, shard_direction = "rows", thread_partition = "static-2d"},
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (n)>, affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%a, %b, %bias : memref<13x8xf32>, memref<8x40xf32>, memref<40xf32>) outs(%c : memref<13x40xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %bias_value: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      %result = arith.addf %sum, %bias_value : f32
      linalg.yield %result : f32
  }
  return
}
