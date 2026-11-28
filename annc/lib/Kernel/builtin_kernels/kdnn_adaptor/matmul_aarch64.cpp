#include <exception>

#include "Context/KernelContextUtils.h"
#include "kdnn.hpp"
#include "kdnn_adaptor/KDNNTensorInfoAdaptor.h"
#include "kdnn_adaptor/KDNNThreadPoolAdaptor.h"

namespace annc::kernels::kdnn_adaptor {

AnncStatusCode matmul_kdnn_impl(KernelContext& context) {
  const auto* lhs = context.getStridedInput(0);
  const auto* rhs = context.getStridedInput(1);
  if (!lhs || !rhs) return context.status();
  if (lhs->element_type != ANNC_ELEMENT_TYPE_F32 ||
      rhs->element_type != ANNC_ELEMENT_TYPE_F32 || lhs->rank != 2 ||
      rhs->rank != 2)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "KDNN MatMul requires rank-2 F32 inputs");
  if (lhs->sizes[1] != rhs->sizes[0]) return ANNC_STATUS_INVALID_ARGUMENT;
  auto* output =
      context.allocateOutput<float>(0, {lhs->sizes[0], rhs->sizes[1]});
  if (!output) return context.status();

  try {
    ScopedKDNNThreadPoolActivation scoped_thread_pool(context);

    const auto lhs_ref = makeDenseTensorRef(*lhs, KDNN::Layout::AB);
    const auto rhs_ref = makeDenseTensorRef(*rhs, KDNN::Layout::AB);
    auto output_ref = makeMutableDenseTensorRef(*output, KDNN::Layout::AB);

    KDNN::Gemm gemm(lhs_ref.info, rhs_ref.info, output_ref.info);
    gemm.Run(lhs_ref.data, rhs_ref.data, output_ref.data);
    return context.status();
  } catch (const std::exception& ex) {
    return context.error(ANNC_STATUS_INTERNAL, ex.what());
  } catch (...) {
    return context.error(ANNC_STATUS_INTERNAL,
                         "[ANNC KDNN] MatMul failed: unknown exception\n");
  }
}

}  // namespace annc::kernels::kdnn_adaptor
