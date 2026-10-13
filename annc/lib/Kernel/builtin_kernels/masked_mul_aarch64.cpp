#include <algorithm>
#include <cstdint>

#include "Kernel/ExecutionContextUtils.h"
#include "Kernel/KernelStatus.h"
#include "Kernel/MemRefTypes.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/ThreadPool.h"

namespace {

// FusedMaskedMul, execution-v2 form: the ONNX mask chain
//
//   Equal(ids, padding) -> Not -> Cast(f32) -> Unsqueeze(last axis)
//     -> [Tile(broadcast to N)] -> Mul(data, mask)
//
// collapsed into one kernel per *site*.  The mask carries no per-element
// information the Tile could move (it is pure broadcast along the last axis),
// so the fused form reads ids row by row:
//
//   Y[b, t, n] = data[b, t, n] * (ids[b, t] != padding ? 1.0f : 0.0f)
//
//   data_i : rank-3 [d0, d1, N_i] f32, contiguous row-major (prefix [d0, d1]
//            shared with ids)
//   ids    : rank-2 [d0, d1] i64
//   output : rank-3, allocated here through the execution context, one slot
//            per site in the order the pattern lists the sites
//
// One chain feeds up to four Muls, and every one of them has to be replaced in
// the same site (the chain dies with them), so a site carries as many outputs
// as the chain has Mul consumers.  The execution-v2 kernel ABI is a fixed C
// signature per symbol, so the arities are separate kernel symbols
// (MaskedMul1..MaskedMul4) that share the implementation below -- the same
// multi-name scheme the KPFusedSparseDynamicStitchN* kernels use.
//
// Every element is multiplied, including the kept ones (x1.0f) -- no memset
// shortcut -- so the fused result is bit-identical to the source chain on
// +-0 / NaN / inf inputs (inf * 0.0f -> NaN must survive).
//
// `padding` is the constant the frontend folded out of Equal(); the pattern
// records it in the CustomizeOp metadata and the ABI passes it as a scalar.

// Shared descriptor checks: data and ids must agree on the row prefix and all
// operands must be contiguous row-major, which is what the indexing assumes.
annc::kernels::KernelStatus checkMaskedMulOperands(AnncMemRef3DF32* data,
                                                   AnncMemRef2DI64* ids,
                                                   int64_t* rows, int64_t* n) {
  if (!data || !ids) return annc::kernels::KernelStatus::InvalidArgument;
  const int64_t d0 = data->sizes[0];
  const int64_t d1 = data->sizes[1];
  const int64_t last = data->sizes[2];
  if (d0 < 0 || d1 < 0 || last < 0) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  // The pattern only fuses sites whose prefix shapes line up; re-check here so
  // a mis-specialized call fails loudly instead of producing a wrong result.
  if (ids->sizes[0] != d0 || ids->sizes[1] != d1) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if (last != 0 && (data->strides[2] != 1 || data->strides[1] != last ||
                    data->strides[0] != d1 * last)) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if (ids->strides[1] != 1 || ids->strides[0] != d1) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  *rows = d0 * d1;
  *n = last;
  return annc::kernels::KernelStatus::Success;
}

// Allocates one site's output (its slot) and scales every row whose id differs
// from `padding`.  The output shape follows the site's own data descriptor --
// the leading dims are runtime values (the batch dim is dynamic), so the shape
// is only known after reading the input.
annc::kernels::KernelStatus maskedMulSiteImpl(
    const AnncExecutionContext* execution, uint32_t slot, AnncMemRef3DF32* data,
    AnncMemRef2DI64* ids, int64_t padding, int64_t* outRows, int64_t* outN,
    float** outData) {
  int64_t rows = 0;
  int64_t n = 0;
  const auto checked = checkMaskedMulOperands(data, ids, &rows, &n);
  if (checked != annc::kernels::KernelStatus::Success) return checked;

  const int64_t d0 = data->sizes[0];
  const int64_t d1 = data->sizes[1];
  // Allocate in every case -- also when rows or n is zero -- because the host
  // checks that the call allocated every declared output slot.
  auto output = annc::kernels::execution::allocateOutput3DF32(
      execution, slot, d0, d1, n, ANNC_ALLOCATION_NONE);
  if (!output) {
    llvm::consumeError(output.takeError());
    return annc::kernels::KernelStatus::RuntimeError;
  }
  if (rows == 0 || n == 0) return annc::kernels::KernelStatus::Success;
  if (output->strides[2] != 1 || output->strides[1] != n ||
      output->strides[0] != d1 * n) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  *outRows = rows;
  *outN = n;
  *outData = ANNC_MEMREF_DATA(*output);
  return annc::kernels::KernelStatus::Success;
}

// Runs one site over [begin, end) of its flattened [rows, n] view.
void maskedMulRows(AnncMemRef3DF32* data, AnncMemRef2DI64* ids,
                   int64_t padding, float* out, int64_t n, int64_t begin,
                   int64_t end) {
  const float* x = ANNC_MEMREF_DATA(*data);
  const int64_t* idValues = ANNC_MEMREF_DATA(*ids);
  for (int64_t row = begin; row < end; ++row) {
    const float factor = (idValues[row] != padding) ? 1.0f : 0.0f;
    const float* src = x + row * n;
    float* dst = out + row * n;
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = src[i] * factor;
    }
  }
}

// One parallel region over every site's rows: the sites are independent and
// their row counts differ, so the row space is concatenated and each worker
// maps a global row back to its site.  A single region per call keeps the
// dispatch cost off the small per-site row counts these models have.
template <size_t NumSites>
annc::kernels::KernelStatus maskedMulImpl(
    annc::threadpool::AnncThreadPool* thread_pool,
    const AnncExecutionContext* execution,
    AnncMemRef3DF32* const (&data)[NumSites], AnncMemRef2DI64* ids,
    int64_t padding) {
  if (execution == nullptr) return annc::kernels::KernelStatus::InvalidArgument;

  int64_t siteRows[NumSites] = {};
  int64_t siteN[NumSites] = {};
  float* siteOut[NumSites] = {};
  int64_t totalRows = 0;
  for (size_t site = 0; site < NumSites; ++site) {
    const auto status = maskedMulSiteImpl(
        execution, static_cast<uint32_t>(site), data[site], ids, padding,
        &siteRows[site], &siteN[site], &siteOut[site]);
    if (status != annc::kernels::KernelStatus::Success) return status;
    totalRows += siteRows[site];
  }
  if (totalRows == 0) return annc::kernels::KernelStatus::Success;

  ::annc::threadpool::parallel_for(
      thread_pool, totalRows, [&](int64_t begin, int64_t end) {
        // The site holding row `begin`; zero-row sites are skipped because the
        // advance condition holds for them too.
        size_t site = 0;
        int64_t base = 0;
        while (site + 1 < NumSites && begin >= base + siteRows[site]) {
          base += siteRows[site];
          ++site;
        }
        // One site may hold the whole range, or the range may cross into the
        // next ones.
        for (int64_t row = begin; row < end;) {
          const int64_t localEnd = std::min(end - base, siteRows[site]);
          maskedMulRows(data[site], ids, padding, siteOut[site], siteN[site],
                        row - base, localEnd);
          row = base + localEnd;
          if (site + 1 >= NumSites) break;
          base += siteRows[site];
          ++site;
        }
      });

  return annc::kernels::KernelStatus::Success;
}

// Entry points for the arity-specific kernel symbols (annc/lib/Kernel/
// builtin_kernels/masked_mul_kernel_specs.inc).
annc::kernels::KernelStatus maskedMul1Impl(
    annc::threadpool::AnncThreadPool* thread_pool,
    const AnncExecutionContext* execution, AnncMemRef3DF32* data0,
    AnncMemRef2DI64* ids, int64_t padding) {
  AnncMemRef3DF32* data[] = {data0};
  return maskedMulImpl(thread_pool, execution, data, ids, padding);
}

annc::kernels::KernelStatus maskedMul2Impl(
    annc::threadpool::AnncThreadPool* thread_pool,
    const AnncExecutionContext* execution, AnncMemRef3DF32* data0,
    AnncMemRef3DF32* data1, AnncMemRef2DI64* ids, int64_t padding) {
  AnncMemRef3DF32* data[] = {data0, data1};
  return maskedMulImpl(thread_pool, execution, data, ids, padding);
}

annc::kernels::KernelStatus maskedMul3Impl(
    annc::threadpool::AnncThreadPool* thread_pool,
    const AnncExecutionContext* execution, AnncMemRef3DF32* data0,
    AnncMemRef3DF32* data1, AnncMemRef3DF32* data2, AnncMemRef2DI64* ids,
    int64_t padding) {
  AnncMemRef3DF32* data[] = {data0, data1, data2};
  return maskedMulImpl(thread_pool, execution, data, ids, padding);
}

annc::kernels::KernelStatus maskedMul4Impl(
    annc::threadpool::AnncThreadPool* thread_pool,
    const AnncExecutionContext* execution, AnncMemRef3DF32* data0,
    AnncMemRef3DF32* data1, AnncMemRef3DF32* data2, AnncMemRef3DF32* data3,
    AnncMemRef2DI64* ids, int64_t padding) {
  AnncMemRef3DF32* data[] = {data0, data1, data2, data3};
  return maskedMulImpl(thread_pool, execution, data, ids, padding);
}

}  // namespace
