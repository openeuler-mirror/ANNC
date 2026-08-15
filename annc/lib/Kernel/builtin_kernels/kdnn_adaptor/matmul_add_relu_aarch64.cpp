#include <exception>

#include "Kernel/KernelStatus.h"
#include "Support/ThreadPool/ThreadPool.h"
#include "kdnn.hpp"
#include "kdnn_adaptor/KDNNTensorInfoAdaptor.h"
#include "kdnn_adaptor/KDNNThreadPoolAdaptor.h"
#include "llvm/Support/raw_ostream.h"

namespace annc::kernels::kdnn_adaptor {
namespace {

KDNN::TensorInfo makeBiasTensorInfo(const AnncMemRef1DF32& bias) {
    int64_t bias_sizes[2] = {1, bias.sizes[0]};
    return KDNN::TensorInfo(makeShape(bias_sizes),
                            KDNN::Element::TypeT::F32,
                            KDNN::Layout::AB);
}

} // namespace

KernelStatus matmul_add_relu_kdnn_impl(
    annc::threadpool::AnncThreadPool* thread_pool,
    AnncMemRef2DF32* output,
    AnncMemRef2DF32* lhs,
    AnncMemRef2DF32* rhs,
    AnncMemRef1DF32* bias) {
    try {
        ScopedKDNNThreadPoolActivation scoped_thread_pool(thread_pool);

        const auto lhs_ref = makeDenseTensorRef(*lhs, KDNN::Layout::AB);
        const auto rhs_ref = makeDenseTensorRef(*rhs, KDNN::Layout::AB);
        auto output_ref = makeMutableDenseTensorRef(*output, KDNN::Layout::AB);
        const auto bias_info = makeBiasTensorInfo(*bias);
        const float* bias_data = ANNC_MEMREF_DATA(*bias);

        KDNN::PostOps post_ops;
        post_ops.AppendEltwise(KDNN::ActivationFunction::RELU);
        KDNN::PostOpsDataPtrs post_op_ptrs;
        post_op_ptrs.push_back(&post_ops);

        KDNN::Attributes attr;
        attr.SetPostOps(post_ops);
        KDNN::Gemm gemm(lhs_ref.info, rhs_ref.info, output_ref.info,
                        bias_info, attr);
        gemm.Run(lhs_ref.data, rhs_ref.data, output_ref.data, bias_data,
                 post_op_ptrs);
        return KernelStatus::Success;
    } catch (const std::exception& ex) {
        llvm::errs() << "[ANNC KDNN] MatMulAddRelu failed: " << ex.what() << '\n';
        return KernelStatus::RuntimeError;
    } catch (...) {
        llvm::errs() << "[ANNC KDNN] MatMulAddRelu failed: unknown exception\n";
        return KernelStatus::UnknownError;
    }
}

} // namespace annc::kernels::kdnn_adaptor
