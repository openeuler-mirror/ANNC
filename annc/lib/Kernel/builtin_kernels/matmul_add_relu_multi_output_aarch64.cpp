#include <cstdint>
#include <exception>

#include "Kernel/ExecutionContextUtils.h"
#include "Kernel/KernelStatus.h"
#include "Kernel/MemRefTypes.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/ThreadPool.h"

namespace {

bool isContiguous2D(const AnncMemRef2DF32& memref) {
  return memref.offset >= 0 && memref.sizes[0] >= 0 && memref.sizes[1] >= 0 &&
         memref.strides[0] == memref.sizes[1] && memref.strides[1] == 1;
}

bool isContiguous1D(const AnncMemRef1DF32& memref) {
  return memref.offset >= 0 && memref.sizes[0] >= 0 && memref.strides[0] == 1;
}

annc::kernels::KernelStatus matmulAddReluWithAddOutputImpl(
    annc::threadpool::AnncThreadPool* threadPool,
    const AnncExecutionContext* execution, AnncMemRef2DF32* lhs,
    AnncMemRef2DF32* rhs, AnncMemRef1DF32* bias) {
  if (execution == nullptr || lhs == nullptr || rhs == nullptr ||
      bias == nullptr || !isContiguous2D(*lhs) || !isContiguous2D(*rhs) ||
      !isContiguous1D(*bias)) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int64_t rows = lhs->sizes[0];
  const int64_t depth = lhs->sizes[1];
  const int64_t columns = rhs->sizes[1];
  if (rhs->sizes[0] != depth || bias->sizes[0] != columns ||
      ((rows > 0 && depth > 0) && lhs->aligned == nullptr) ||
      ((depth > 0 && columns > 0) && rhs->aligned == nullptr) ||
      (columns > 0 && bias->aligned == nullptr)) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  auto matmul = annc::kernels::execution::allocateTemp2DF32(
      execution, rows, columns, ANNC_ALLOCATION_NONE);
  if (!matmul) {
    llvm::consumeError(matmul.takeError());
    return annc::kernels::KernelStatus::RuntimeError;
  }
  auto add = annc::kernels::execution::allocateOutput2DF32(
      execution, 0, rows, columns, ANNC_ALLOCATION_NONE);
  if (!add) {
    llvm::consumeError(add.takeError());
    return annc::kernels::KernelStatus::RuntimeError;
  }
  auto relu = annc::kernels::execution::allocateOutput2DF32(
      execution, 1, rows, columns, ANNC_ALLOCATION_NONE);
  if (!relu) {
    llvm::consumeError(relu.takeError());
    return annc::kernels::KernelStatus::RuntimeError;
  }

  if (rows == 0 || columns == 0) {
    return annc::kernels::KernelStatus::Success;
  }

  const float* lhsData = depth == 0 ? nullptr : ANNC_MEMREF_DATA(*lhs);
  const float* rhsData = depth == 0 ? nullptr : ANNC_MEMREF_DATA(*rhs);
  const float* biasData = ANNC_MEMREF_DATA(*bias);
  float* matmulData = ANNC_MEMREF_DATA(*matmul);
  float* addData = ANNC_MEMREF_DATA(*add);
  float* reluData = ANNC_MEMREF_DATA(*relu);

  try {
    annc::threadpool::parallel_for(
        threadPool, rows, [&](int64_t begin, int64_t end) {
          for (int64_t row = begin; row < end; ++row) {
            for (int64_t column = 0; column < columns; ++column) {
              float value = 0.0f;
              for (int64_t index = 0; index < depth; ++index) {
                value += lhsData[row * depth + index] *
                         rhsData[index * columns + column];
              }
              const int64_t offset = row * columns + column;
              matmulData[offset] = value;
              addData[offset] = value + biasData[column];
              reluData[offset] =
                  addData[offset] > 0.0f ? addData[offset] : 0.0f;
            }
          }
        });
  } catch (const std::exception&) {
    return annc::kernels::KernelStatus::RuntimeError;
  } catch (...) {
    return annc::kernels::KernelStatus::UnknownError;
  }
  return annc::kernels::KernelStatus::Success;
}

}  // namespace
