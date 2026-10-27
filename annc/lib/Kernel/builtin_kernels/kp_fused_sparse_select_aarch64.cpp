#include <cstdint>
#include <exception>

#include "Context/KernelContextUtils.h"

namespace {

// Fused sparse select, ported from the 812 KPFusedSparseSelect TF kernel,
// adapted to the context ABI (outputs allocated through the execution
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
AnncStatusCode kpFusedSparseSelectImpl(annc::kernels::KernelContext& context) {
  const auto* a = context.getInput(0);
  const auto* b = context.getInput(1);
  const auto* c = context.getInput(2);
  const auto* gt = context.getInput(3);
  const auto* eq1 = context.getInput(4);
  const auto* eq2 = context.getInput(5);
  const auto* eq3 = context.getInput(6);
  if (!a || !b || !c || !gt || !eq1 || !eq2 || !eq3) return context.status();

  if (!a || !b || !c || !gt || !eq1 || !eq2 || !eq3) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }

  const int64_t n = a->sizes[0];
  if (n < 0 || b->sizes[0] != n || c->sizes[0] != n) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  if (a->offset < 0 || b->offset < 0 || c->offset < 0 || gt->offset < 0 ||
      eq1->offset < 0 || eq2->offset < 0 || eq3->offset < 0) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  if ((n > 0 && (!a->aligned || !b->aligned || !c->aligned)) || !gt->aligned ||
      !eq1->aligned || !eq2->aligned || !eq3->aligned) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  if (a->strides[0] != 1 || b->strides[0] != 1 || c->strides[0] != 1) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }

  try {
    const int32_t gt_v = *annc::kernels::contextData<int32_t>(*gt);
    const int32_t eq1_v = *annc::kernels::contextData<int32_t>(*eq1);
    const int32_t eq2_v = *annc::kernels::contextData<int32_t>(*eq2);
    // 812: equal3_val 读入但从不使用 (right-branch select 双分支都是
    // Fill(1.0), 结果恒 1.0 — kernel 直接写 w 第 1 列)。
    const int32_t eq3_v = *annc::kernels::contextData<int32_t>(*eq3);
    (void)eq3_v;

    // Context: value-dependent [N,1] / [N,2] outputs.
    auto* out_x = context.allocateOutput<int32_t>(0, {n, 1});
    if (!out_x) return context.status();
    auto* out_y = context.allocateOutput<float>(1, {n, 1});
    if (!out_y) return context.status();
    auto* out_w = context.allocateOutput<float>(2, {n, 2});
    if (!out_w) return context.status();

    const int32_t* a_data = annc::kernels::contextData<int32_t>(*a);
    const int32_t* b_data = annc::kernels::contextData<int32_t>(*b);
    int32_t* x_data = annc::kernels::contextData<int32_t>(*out_x);
    float* y_data = annc::kernels::contextData<float>(*out_y);
    float* w_data = annc::kernels::contextData<float>(*out_w);

    for (int64_t i = 0; i < n; ++i) {
      const float a_greater = (a_data[i] > gt_v) ? 1.0f : 0.0f;
      const float res_equal1 = (b_data[i] == eq1_v) ? 1.0f : a_greater;
      const float res_equal2 = (b_data[i] == eq2_v) ? 1.0f : res_equal1;
      x_data[i] = a_data[i];
      y_data[i] = res_equal2;
      w_data[i * 2 + 0] = res_equal2;
      w_data[i * 2 + 1] = 1.0f;
    }
    return context.status();
  } catch (const std::exception&) {
    return ANNC_STATUS_INTERNAL;
  } catch (...) {
    return ANNC_STATUS_INTERNAL;
  }
}

}  // namespace
