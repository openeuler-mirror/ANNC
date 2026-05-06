#include <cstdlib>
#include <iostream>

#include "Kernel/threadpool/ThreadPool.h"
#include "kdnn.hpp"
#include "kdnn_adaptor/KDNNTensorInfoAdaptor.h"
#include "kdnn_adaptor/KDNNThreadPoolAdaptor.h"

namespace annc::kernels::kdnn_adaptor {

void matmul_kdnn_impl(annc::threadpool::AnncThreadPool* thread_pool,
                      AnncMemRef2DF32* lhs,
                      AnncMemRef2DF32* rhs,
                      AnncMemRef2DF32* output) {
    try {
        ScopedKDNNThreadPoolActivation scoped_thread_pool(thread_pool);

        const auto lhs_ref = makeDenseTensorRef(*lhs, KDNN::Layout::AB);
        const auto rhs_ref = makeDenseTensorRef(*rhs, KDNN::Layout::AB);
        auto output_ref = makeMutableDenseTensorRef(*output, KDNN::Layout::AB);

        KDNN::Gemm gemm(lhs_ref.info, rhs_ref.info, output_ref.info);
        gemm.Run(lhs_ref.data, rhs_ref.data, output_ref.data);
    } catch (const std::exception& ex) {
        std::cerr << "[ANNC KDNN] MatMul failed: " << ex.what() << '\n';
        std::abort();
    }
}

} // namespace annc::kernels::kdnn_adaptor
