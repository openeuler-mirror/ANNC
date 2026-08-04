#include <exception>

#include "Kernel/KernelStatus.h"
#include "Support/ThreadPool/ThreadPool.h"
#include "kdnn.hpp"
#include "kdnn_adaptor/KDNNTensorInfoAdaptor.h"
#include "kdnn_adaptor/KDNNThreadPoolAdaptor.h"
#include "llvm/Support/raw_ostream.h"

namespace annc::kernels::kdnn_adaptor {

KernelStatus matmul_kdnn_impl(annc::threadpool::AnncThreadPool* thread_pool,
                              AnncMemRef2DF32* output,
                              AnncMemRef2DF32* lhs,
                              AnncMemRef2DF32* rhs) {
    try {
        ScopedKDNNThreadPoolActivation scoped_thread_pool(thread_pool);

        const auto lhs_ref = makeDenseTensorRef(*lhs, KDNN::Layout::AB);
        const auto rhs_ref = makeDenseTensorRef(*rhs, KDNN::Layout::AB);
        auto output_ref = makeMutableDenseTensorRef(*output, KDNN::Layout::AB);

        KDNN::Gemm gemm(lhs_ref.info, rhs_ref.info, output_ref.info);
        gemm.Run(lhs_ref.data, rhs_ref.data, output_ref.data);
        return KernelStatus::Success;
    } catch (const std::exception& ex) {
        llvm::errs() << "[ANNC KDNN] MatMul failed: " << ex.what() << '\n';
        return KernelStatus::RuntimeError;
    } catch (...) {
        llvm::errs() << "[ANNC KDNN] MatMul failed: unknown exception\n";
        return KernelStatus::UnknownError;
    }
}

} // namespace annc::kernels::kdnn_adaptor
