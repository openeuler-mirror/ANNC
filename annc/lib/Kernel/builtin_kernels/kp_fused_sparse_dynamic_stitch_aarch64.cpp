#include "Kernel/MemRefTypes.h"
#include "Kernel/KernelStatus.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/ThreadPool.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>

#include "llvm/Support/raw_ostream.h"

namespace {

// Fused sparse dynamic stitch, ported from the 812 KPFusedSparseDynamicStitch
// TF kernel (tf_custom_ops/kernels/embedding_fused_sparse_dynamic_stitch.cc).
//
//   output[i] = variables[x[i] % N][x[i] / N]   (row-major copy)
//
// x [L] int64, variables [N] of [rows_i, stride] float32 (equal stride),
// output [L, stride].
template <int N>
annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_impl(
    annc::threadpool::AnncThreadPool* thread_pool,
    AnncMemRef2DF32* output,
    AnncMemRef1DI64* x,
    const std::array<AnncMemRef2DF32*, N>& variables) {
    static_assert(N >= 2, "N must be >= 2");

    if (!output || !x) {
        llvm::errs() << "[ANNC Kernel] KPFusedSparseDynamicStitch failed: "
                     "null memref argument\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    for (int i = 0; i < N; ++i) {
        if (!variables[i]) {
            llvm::errs() << "[ANNC Kernel] KPFusedSparseDynamicStitch failed: "
                         "null variable memref\n";
            return annc::kernels::KernelStatus::InvalidArgument;
        }
    }

    const int64_t L = x->sizes[0];
    const int64_t out_rows = output->sizes[0];
    const int64_t out_cols = output->sizes[1];

    if (L < 0 || out_rows < 0 || out_cols <= 0) {
        llvm::errs() << "[ANNC Kernel] KPFusedSparseDynamicStitch failed: "
                     "invalid shape\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if (L != out_rows) {
        llvm::errs() << "[ANNC Kernel] KPFusedSparseDynamicStitch failed: "
                     "output row count mismatch\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    for (int i = 0; i < N; ++i) {
        const auto* v = variables[i];
        if (v->sizes[0] < 0 || v->sizes[1] != out_cols) {
            llvm::errs() << "[ANNC Kernel] KPFusedSparseDynamicStitch failed: "
                         "variable shape mismatch\n";
            return annc::kernels::KernelStatus::InvalidArgument;
        }
    }
    if (output->offset < 0 || x->offset < 0) {
        llvm::errs() << "[ANNC Kernel] KPFusedSparseDynamicStitch failed: "
                     "negative memref offset\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if ((L > 0 && !x->aligned) || (out_rows > 0 && !output->aligned)) {
        llvm::errs() << "[ANNC Kernel] KPFusedSparseDynamicStitch failed: "
                     "null memref data\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if (x->strides[0] != 1 ||
        output->strides[0] != out_cols || output->strides[1] != 1) {
        llvm::errs() << "[ANNC Kernel] KPFusedSparseDynamicStitch failed: "
                     "non-contiguous memref is unsupported\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    for (int i = 0; i < N; ++i) {
        const auto* v = variables[i];
        if (v->sizes[0] > 0 &&
            (v->strides[0] != out_cols || v->strides[1] != 1 ||
             !v->aligned)) {
            llvm::errs() << "[ANNC Kernel] KPFusedSparseDynamicStitch failed: "
                         "non-contiguous variable memref\n";
            return annc::kernels::KernelStatus::InvalidArgument;
        }
    }

    const int64_t* x_data = ANNC_MEMREF_DATA(*x);
    float* out = ANNC_MEMREF_DATA(*output);
    std::array<const float*, N> var_data;
    std::array<int64_t, N> var_rows;
    for (int i = 0; i < N; ++i) {
        var_data[i] = ANNC_MEMREF_DATA(*variables[i]);
        var_rows[i] = variables[i]->sizes[0];
    }

    if (L == 0) {
        return annc::kernels::KernelStatus::Success;
    }

    try {
        const size_t copy_size =
            static_cast<size_t>(out_cols) * sizeof(float);
        std::atomic<bool> failed{false};
        annc::threadpool::parallel_for(
            thread_pool, L, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    const int64_t global_id = x_data[i];
                    const int64_t table_id = global_id % N;
                    const int64_t row_id = global_id / N;
                    if (row_id < 0 || row_id >= var_rows[table_id]) {
                        failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                    std::memcpy(out + i * out_cols,
                                var_data[table_id] + row_id * out_cols,
                                copy_size);
                }
            });
        if (failed.load(std::memory_order_relaxed)) {
            llvm::errs() << "[ANNC Kernel] KPFusedSparseDynamicStitch failed: "
                         "row_id out of range\n";
            return annc::kernels::KernelStatus::InvalidArgument;
        }
        return annc::kernels::KernelStatus::Success;
    } catch (const std::exception& ex) {
        llvm::errs() << "[ANNC Kernel] KPFusedSparseDynamicStitch failed: "
                     << ex.what() << '\n';
        return annc::kernels::KernelStatus::RuntimeError;
    } catch (...) {
        llvm::errs() << "[ANNC Kernel] KPFusedSparseDynamicStitch failed: "
                     "unknown exception\n";
        return annc::kernels::KernelStatus::UnknownError;
    }
}

// Non-template dispatch entry points.  The ANNC_KERNEL macro body is
// comma-split at the preprocessor level, so neither template
// argument lists nor brace-initialized aggregates may appear
// there.  These helpers accept the N variable pointers as
// separate arguments (all commas inside parens) and select the
// N specialization.
annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n2(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1) {
  const std::array<AnncMemRef2DF32*, 2> vars{v0, v1};
  return kp_fused_sparse_dynamic_stitch_impl<2>(tp, output, x, vars);
}

annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n3(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1, AnncMemRef2DF32* v2) {
  const std::array<AnncMemRef2DF32*, 3> vars{v0, v1, v2};
  return kp_fused_sparse_dynamic_stitch_impl<3>(tp, output, x, vars);
}

annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n4(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1, AnncMemRef2DF32* v2, AnncMemRef2DF32* v3) {
  const std::array<AnncMemRef2DF32*, 4> vars{v0, v1, v2, v3};
  return kp_fused_sparse_dynamic_stitch_impl<4>(tp, output, x, vars);
}

annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n5(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1, AnncMemRef2DF32* v2, AnncMemRef2DF32* v3, AnncMemRef2DF32* v4) {
  const std::array<AnncMemRef2DF32*, 5> vars{v0, v1, v2, v3, v4};
  return kp_fused_sparse_dynamic_stitch_impl<5>(tp, output, x, vars);
}

annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n6(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1, AnncMemRef2DF32* v2, AnncMemRef2DF32* v3, AnncMemRef2DF32* v4, AnncMemRef2DF32* v5) {
  const std::array<AnncMemRef2DF32*, 6> vars{v0, v1, v2, v3, v4, v5};
  return kp_fused_sparse_dynamic_stitch_impl<6>(tp, output, x, vars);
}

annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n7(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1, AnncMemRef2DF32* v2, AnncMemRef2DF32* v3, AnncMemRef2DF32* v4, AnncMemRef2DF32* v5, AnncMemRef2DF32* v6) {
  const std::array<AnncMemRef2DF32*, 7> vars{v0, v1, v2, v3, v4, v5, v6};
  return kp_fused_sparse_dynamic_stitch_impl<7>(tp, output, x, vars);
}

annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n8(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1, AnncMemRef2DF32* v2, AnncMemRef2DF32* v3, AnncMemRef2DF32* v4, AnncMemRef2DF32* v5, AnncMemRef2DF32* v6, AnncMemRef2DF32* v7) {
  const std::array<AnncMemRef2DF32*, 8> vars{v0, v1, v2, v3, v4, v5, v6, v7};
  return kp_fused_sparse_dynamic_stitch_impl<8>(tp, output, x, vars);
}

annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n9(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1, AnncMemRef2DF32* v2, AnncMemRef2DF32* v3, AnncMemRef2DF32* v4, AnncMemRef2DF32* v5, AnncMemRef2DF32* v6, AnncMemRef2DF32* v7, AnncMemRef2DF32* v8) {
  const std::array<AnncMemRef2DF32*, 9> vars{v0, v1, v2, v3, v4, v5, v6, v7, v8};
  return kp_fused_sparse_dynamic_stitch_impl<9>(tp, output, x, vars);
}

annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n10(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1, AnncMemRef2DF32* v2, AnncMemRef2DF32* v3, AnncMemRef2DF32* v4, AnncMemRef2DF32* v5, AnncMemRef2DF32* v6, AnncMemRef2DF32* v7, AnncMemRef2DF32* v8, AnncMemRef2DF32* v9) {
  const std::array<AnncMemRef2DF32*, 10> vars{v0, v1, v2, v3, v4, v5, v6, v7, v8, v9};
  return kp_fused_sparse_dynamic_stitch_impl<10>(tp, output, x, vars);
}

annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n11(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1, AnncMemRef2DF32* v2, AnncMemRef2DF32* v3, AnncMemRef2DF32* v4, AnncMemRef2DF32* v5, AnncMemRef2DF32* v6, AnncMemRef2DF32* v7, AnncMemRef2DF32* v8, AnncMemRef2DF32* v9, AnncMemRef2DF32* v10) {
  const std::array<AnncMemRef2DF32*, 11> vars{v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10};
  return kp_fused_sparse_dynamic_stitch_impl<11>(tp, output, x, vars);
}

annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n12(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1, AnncMemRef2DF32* v2, AnncMemRef2DF32* v3, AnncMemRef2DF32* v4, AnncMemRef2DF32* v5, AnncMemRef2DF32* v6, AnncMemRef2DF32* v7, AnncMemRef2DF32* v8, AnncMemRef2DF32* v9, AnncMemRef2DF32* v10, AnncMemRef2DF32* v11) {
  const std::array<AnncMemRef2DF32*, 12> vars{v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11};
  return kp_fused_sparse_dynamic_stitch_impl<12>(tp, output, x, vars);
}

annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n13(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1, AnncMemRef2DF32* v2, AnncMemRef2DF32* v3, AnncMemRef2DF32* v4, AnncMemRef2DF32* v5, AnncMemRef2DF32* v6, AnncMemRef2DF32* v7, AnncMemRef2DF32* v8, AnncMemRef2DF32* v9, AnncMemRef2DF32* v10, AnncMemRef2DF32* v11, AnncMemRef2DF32* v12) {
  const std::array<AnncMemRef2DF32*, 13> vars{v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12};
  return kp_fused_sparse_dynamic_stitch_impl<13>(tp, output, x, vars);
}

annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n14(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1, AnncMemRef2DF32* v2, AnncMemRef2DF32* v3, AnncMemRef2DF32* v4, AnncMemRef2DF32* v5, AnncMemRef2DF32* v6, AnncMemRef2DF32* v7, AnncMemRef2DF32* v8, AnncMemRef2DF32* v9, AnncMemRef2DF32* v10, AnncMemRef2DF32* v11, AnncMemRef2DF32* v12, AnncMemRef2DF32* v13) {
  const std::array<AnncMemRef2DF32*, 14> vars{v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13};
  return kp_fused_sparse_dynamic_stitch_impl<14>(tp, output, x, vars);
}

annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n15(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1, AnncMemRef2DF32* v2, AnncMemRef2DF32* v3, AnncMemRef2DF32* v4, AnncMemRef2DF32* v5, AnncMemRef2DF32* v6, AnncMemRef2DF32* v7, AnncMemRef2DF32* v8, AnncMemRef2DF32* v9, AnncMemRef2DF32* v10, AnncMemRef2DF32* v11, AnncMemRef2DF32* v12, AnncMemRef2DF32* v13, AnncMemRef2DF32* v14) {
  const std::array<AnncMemRef2DF32*, 15> vars{v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14};
  return kp_fused_sparse_dynamic_stitch_impl<15>(tp, output, x, vars);
}

annc::kernels::KernelStatus kp_fused_sparse_dynamic_stitch_n16(
    annc::threadpool::AnncThreadPool* tp, AnncMemRef2DF32* output, AnncMemRef1DI64* x, AnncMemRef2DF32* v0, AnncMemRef2DF32* v1, AnncMemRef2DF32* v2, AnncMemRef2DF32* v3, AnncMemRef2DF32* v4, AnncMemRef2DF32* v5, AnncMemRef2DF32* v6, AnncMemRef2DF32* v7, AnncMemRef2DF32* v8, AnncMemRef2DF32* v9, AnncMemRef2DF32* v10, AnncMemRef2DF32* v11, AnncMemRef2DF32* v12, AnncMemRef2DF32* v13, AnncMemRef2DF32* v14, AnncMemRef2DF32* v15) {
  const std::array<AnncMemRef2DF32*, 16> vars{v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
  return kp_fused_sparse_dynamic_stitch_impl<16>(tp, output, x, vars);
}

} // namespace
