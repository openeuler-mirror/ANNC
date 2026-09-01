#include <cstdint>
#include <exception>

#include "Kernel/ExecutionContextUtils.h"
#include "Kernel/KernelStatus.h"
#include "Kernel/MemRefTypes.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/ThreadPool.h"

namespace {

// Fused sparse select, ported from the 812 KPFusedSparseSelect TF kernel,
// adapted to the Execution V2 ABI (outputs allocated through the execution
// context callbacks).  Semantics (per element i):
//
//   a_greater   = (a[i] > gt)   ? 1.0f : 0.0f        // Greater+Cast
//   res_equal1  = (b[i] == eq1) ? 1.0f : a_greater   // inner Where
//   res_equal2  = (b[i] == eq2) ? 1.0f : res_equal1  // outer Where
//   x[i] = a[i];  y[i] = res_equal2;
//   w[i,0] = res_equal2;  w[i,1] = 1.0f              // concat [?,0] 空列
//
// Outputs: slot 0 = out_x (i32 [N,1]), slot 1 = out_y (f32 [N,1]),
//          slot 2 = out_w (f32 [N,2]); N is value-dependent.
// 812 reads equal3_val but never uses it; the port keeps the read for
// semantic fidelity.
annc::kernels::KernelStatus kpFusedSparseSelectImpl(
    annc::threadpool::AnncThreadPool* thread_pool,
    const AnncExecutionContext* execution, AnncMemRef1DI32* a,
    AnncMemRef1DI32* b, AnncMemRef1DI32* c, AnncMemRef0DI32* gt,
    AnncMemRef0DI32* eq1, AnncMemRef0DI32* eq2, AnncMemRef0DI32* eq3) {
  (void)thread_pool;
  if (!execution || !a || !b || !c || !gt || !eq1 || !eq2 || !eq3) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int64_t n = a->sizes[0];
  if (n < 0 || b->sizes[0] != n || c->sizes[0] != n) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if (a->offset < 0 || b->offset < 0 || c->offset < 0 || gt->offset < 0 ||
      eq1->offset < 0 || eq2->offset < 0 || eq3->offset < 0) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if ((n > 0 && (!a->aligned || !b->aligned || !c->aligned)) || !gt->aligned ||
      !eq1->aligned || !eq2->aligned || !eq3->aligned) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if (a->strides[0] != 1 || b->strides[0] != 1 || c->strides[0] != 1) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  try {
    const int32_t gt_v = *ANNC_MEMREF_DATA(*gt);
    const int32_t eq1_v = *ANNC_MEMREF_DATA(*eq1);
    const int32_t eq2_v = *ANNC_MEMREF_DATA(*eq2);
    // 812: equal3_val 读入但从不使用 (right-branch select 双分支都是
    // Fill(1.0), 结果恒 1.0 — kernel 直接写 w 第 1 列)。
    const int32_t eq3_v = *ANNC_MEMREF_DATA(*eq3);
    (void)eq3_v;

    // V2: value-dependent [N,1] / [N,2] outputs.
    auto out_x = annc::kernels::execution::allocateOutput2DI32(
        execution, 0, n, 1, ANNC_ALLOCATION_NONE);
    if (!out_x) {
      llvm::consumeError(out_x.takeError());
      return annc::kernels::KernelStatus::RuntimeError;
    }
    auto out_y = annc::kernels::execution::allocateOutput2DF32(
        execution, 1, n, 1, ANNC_ALLOCATION_NONE);
    if (!out_y) {
      llvm::consumeError(out_y.takeError());
      return annc::kernels::KernelStatus::RuntimeError;
    }
    auto out_w = annc::kernels::execution::allocateOutput2DF32(
        execution, 2, n, 2, ANNC_ALLOCATION_NONE);
    if (!out_w) {
      llvm::consumeError(out_w.takeError());
      return annc::kernels::KernelStatus::RuntimeError;
    }

    const int32_t* a_data = ANNC_MEMREF_DATA(*a);
    const int32_t* b_data = ANNC_MEMREF_DATA(*b);
    int32_t* x_data = ANNC_MEMREF_DATA(*out_x);
    float* y_data = ANNC_MEMREF_DATA(*out_y);
    float* w_data = ANNC_MEMREF_DATA(*out_w);

    for (int64_t i = 0; i < n; ++i) {
      const float a_greater = (a_data[i] > gt_v) ? 1.0f : 0.0f;
      const float res_equal1 = (b_data[i] == eq1_v) ? 1.0f : a_greater;
      const float res_equal2 = (b_data[i] == eq2_v) ? 1.0f : res_equal1;
      x_data[i] = a_data[i];
      y_data[i] = res_equal2;
      w_data[i * 2 + 0] = res_equal2;
      w_data[i * 2 + 1] = 1.0f;
    }
    return annc::kernels::KernelStatus::Success;
  } catch (const std::exception&) {
    return annc::kernels::KernelStatus::RuntimeError;
  } catch (...) {
    return annc::kernels::KernelStatus::UnknownError;
  }
}

// Non-template dispatch entry point (macro-safe: all commas in parens).
annc::kernels::KernelStatus kpFusedSparseSelect(
    annc::threadpool::AnncThreadPool* tp, const AnncExecutionContext* execution,
    AnncMemRef1DI32* a, AnncMemRef1DI32* b, AnncMemRef1DI32* c,
    AnncMemRef0DI32* gt, AnncMemRef0DI32* eq1, AnncMemRef0DI32* eq2,
    AnncMemRef0DI32* eq3) {
  return kpFusedSparseSelectImpl(tp, execution, a, b, c, gt, eq1, eq2, eq3);
}

}  // namespace
