#include <exception>

#include "Context/KernelContextUtils.h"
#include "kdnn.hpp"
#include "kdnn_adaptor/KDNNTensorInfoAdaptor.h"
#include "kdnn_adaptor/KDNNThreadPoolAdaptor.h"

namespace annc::kernels::kdnn_adaptor {
namespace {

KDNN::TensorInfo makeBiasTensorInfo(const AnncTensorView& bias) {
  int64_t bias_sizes[2] = {1, bias.sizes[0]};
  return KDNN::TensorInfo(makeShape(bias_sizes), KDNN::Element::TypeT::F32,
                          KDNN::Layout::AB);
}

}  // namespace

AnncStatusCode matmul_add_kdnn_packed_impl(KernelContext& context) {
  const auto* lhs = context.getStridedInput(0);
  const auto* rhs = context.getStridedInput(1);
  const auto* bias = context.getInput(2);
  if (!lhs || !rhs || !bias) return context.status();
  if (lhs->element_type != ANNC_ELEMENT_TYPE_F32 ||
      rhs->element_type != ANNC_ELEMENT_TYPE_F32 || lhs->rank != 2 ||
      rhs->rank != 2)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "KDNN MatMulAdd requires rank-2 F32 inputs");
  if (lhs->sizes[1] != rhs->sizes[0] || bias->sizes[0] != rhs->sizes[1])
    return ANNC_STATUS_INVALID_ARGUMENT;
  auto* output =
      context.allocateOutput<float>(0, {lhs->sizes[0], rhs->sizes[1]});
  if (!output) return context.status();

  try {
    ScopedKDNNThreadPoolActivation scoped_thread_pool(context);

    const auto lhs_ref = makeDenseTensorRef(*lhs, KDNN::Layout::AB);
    const auto rhs_ref = makeDenseTensorRef(*rhs, KDNN::Layout::AB);
    auto output_ref = makeMutableDenseTensorRef(*output, KDNN::Layout::AB);
    const auto bias_info = makeBiasTensorInfo(*bias);
    const float* bias_data = contextData<float>(*bias);

    KDNN::Gemm gemm(lhs_ref.info, rhs_ref.info, output_ref.info, bias_info);
    // RunWithPackedB consumes the pre-packed rhs layout and matches the
    // kdnn_packed contract advertised to the constant-folding pass.
    gemm.RunWithPackedB(lhs_ref.data, rhs_ref.data, output_ref.data, bias_data);
    return context.status();
  } catch (const std::exception& ex) {
    return context.error(ANNC_STATUS_INTERNAL, ex.what());
  } catch (...) {
    return context.error(
        ANNC_STATUS_INTERNAL,
        "[ANNC KDNN] Packed MatMulAdd failed: unknown exception\n");
  }
}

}  // namespace annc::kernels::kdnn_adaptor
