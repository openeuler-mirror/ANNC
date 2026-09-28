#ifndef ANNC_AARCH64_GEMM_DISPATCH_H
#define ANNC_AARCH64_GEMM_DISPATCH_H

#include <stdint.h>

#include "Kernel/MemRefTypes.h"

// This is the default FuncToLLVM ABI of a private task with signature:
//   (i64, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
typedef void (*AnncAArch64GemmTask)(
    int64_t task_id, float* lhs_allocated, float* lhs_aligned,
    int64_t lhs_offset, int64_t lhs_size0, int64_t lhs_size1,
    int64_t lhs_stride0, int64_t lhs_stride1, float* rhs_allocated,
    float* rhs_aligned, int64_t rhs_offset, int64_t rhs_size0,
    int64_t rhs_size1, int64_t rhs_stride0, int64_t rhs_stride1,
    float* out_allocated, float* out_aligned, int64_t out_offset,
    int64_t out_size0, int64_t out_size1, int64_t out_stride0,
    int64_t out_stride1);

// C representation of MLIR's unranked memref descriptor. The descriptor
// storage remains owned by the synchronous caller for the duration of the
// thread-pool dispatch.
typedef struct {
  int64_t rank;
  void* descriptor;
} AnncUnrankedMemRefF32;

typedef void (*AnncAArch64GemmEpilogueTask)(
    int64_t task_id, float* lhs_allocated, float* lhs_aligned,
    int64_t lhs_offset, int64_t lhs_size0, int64_t lhs_size1,
    int64_t lhs_stride0, int64_t lhs_stride1, float* rhs_allocated,
    float* rhs_aligned, int64_t rhs_offset, int64_t rhs_size0,
    int64_t rhs_size1, int64_t rhs_stride0, int64_t rhs_stride1,
    float* out_allocated, float* out_aligned, int64_t out_offset,
    int64_t out_size0, int64_t out_size1, int64_t out_stride0,
    int64_t out_stride1, int64_t epilogue0_rank, void* epilogue0_descriptor,
    int64_t epilogue1_rank, void* epilogue1_descriptor,
    int64_t epilogue2_rank, void* epilogue2_descriptor,
    int64_t epilogue3_rank, void* epilogue3_descriptor,
    int64_t epilogue4_rank, void* epilogue4_descriptor,
    int64_t epilogue5_rank, void* epilogue5_descriptor,
    int64_t epilogue6_rank, void* epilogue6_descriptor,
    int64_t epilogue7_rank, void* epilogue7_descriptor);

#ifdef __cplusplus
extern "C" {
#endif

// C-interface entry emitted for the private MLIR declaration
// `@annc_threadpool_parallel_for_gemm` with `llvm.emit_c_interface`. `total`
// is the planner's frozen static task count and effective worker budget.
void _mlir_ciface_annc_threadpool_parallel_for_gemm(
    int64_t total, AnncAArch64GemmTask task, const AnncMemRef2DF32* lhs,
    const AnncMemRef2DF32* rhs,
    const AnncMemRef2DF32* out);

void _mlir_ciface_annc_threadpool_parallel_for_gemm_epilogue(
    int64_t total, AnncAArch64GemmEpilogueTask task,
    const AnncMemRef2DF32* lhs, const AnncMemRef2DF32* rhs,
    const AnncMemRef2DF32* out, const AnncUnrankedMemRefF32* epilogue0,
    const AnncUnrankedMemRefF32* epilogue1,
    const AnncUnrankedMemRefF32* epilogue2,
    const AnncUnrankedMemRefF32* epilogue3,
    const AnncUnrankedMemRefF32* epilogue4,
    const AnncUnrankedMemRefF32* epilogue5,
    const AnncUnrankedMemRefF32* epilogue6,
    const AnncUnrankedMemRefF32* epilogue7);
#ifdef __cplusplus
}
#endif

#endif  // ANNC_AARCH64_GEMM_DISPATCH_H
