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
#ifdef __cplusplus
}
#endif

#endif  // ANNC_AARCH64_GEMM_DISPATCH_H
