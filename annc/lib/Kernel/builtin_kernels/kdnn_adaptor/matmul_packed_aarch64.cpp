#include <exception>

#include "Kernel/KernelStatus.h"
#include "Kernel/threadpool/ThreadPool.h"
#include "kdnn.hpp"
#include "kdnn_adaptor/KDNNTensorInfoAdaptor.h"
#include "kdnn_adaptor/KDNNThreadPoolAdaptor.h"
#include "llvm/Support/raw_ostream.h"

namespace annc::kernels::kdnn_adaptor {

static void printMemRef2D(const char* name, const AnncMemRef2DF32* memref) {
    if (!memref) {
        llvm::errs() << ' ' << name << "=<null>";
        return;
    }
    llvm::errs() << ' ' << name << "=[sizes " << memref->sizes[0] << 'x'
              << memref->sizes[1] << ", strides " << memref->strides[0]
              << 'x' << memref->strides[1] << ", offset " << memref->offset
              << ']';
}

KernelStatus matmul_kdnn_packed_impl(
    annc::threadpool::AnncThreadPool* thread_pool,
    AnncMemRef2DF32* output,
    AnncMemRef2DF32* lhs,
    AnncMemRef2DF32* rhs) {
    try {
        ScopedKDNNThreadPoolActivation scoped_thread_pool(thread_pool);

        const auto lhs_ref = makeDenseTensorRef(*lhs, KDNN::Layout::AB);
        const auto rhs_ref = makeDenseTensorRef(*rhs, KDNN::Layout::AB);
        auto output_ref = makeMutableDenseTensorRef(*output, KDNN::Layout::AB);

        KDNN::Gemm gemm(lhs_ref.info, rhs_ref.info, output_ref.info);
        gemm.RunWithPackedB(lhs_ref.data, rhs_ref.data, output_ref.data);
        return KernelStatus::Success;
    } catch (const std::exception& ex) {
        llvm::errs() << "[ANNC KDNN] Packed MatMul failed: " << ex.what();
        printMemRef2D("lhs", lhs);
        printMemRef2D("rhs", rhs);
        printMemRef2D("output", output);
        llvm::errs() << '\n';
        return KernelStatus::RuntimeError;
    } catch (...) {
        llvm::errs() << "[ANNC KDNN] Packed MatMul failed: unknown exception";
        printMemRef2D("lhs", lhs);
        printMemRef2D("rhs", rhs);
        printMemRef2D("output", output);
        llvm::errs() << '\n';
        return KernelStatus::UnknownError;
    }
}

} // namespace annc::kernels::kdnn_adaptor
