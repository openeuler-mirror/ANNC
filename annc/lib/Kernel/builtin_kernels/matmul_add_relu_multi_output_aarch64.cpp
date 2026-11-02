#include <cstdint>
#include <exception>

#include "Context/KernelContextUtils.h"

namespace {

AnncStatusCode matmulAddReluWithAddOutputImpl(
    annc::kernels::KernelContext& context) {
  const auto* lhs = context.getInput(0);
  const auto* rhs = context.getInput(1);
  const auto* bias = context.getInput(2);
  if (!lhs || !rhs || !bias) return context.status();

  const int64_t rows = lhs->sizes[0];
  const int64_t depth = lhs->sizes[1];
  const int64_t columns = rhs->sizes[1];
  if (rhs->sizes[0] != depth || bias->sizes[0] != columns ||
      ((rows > 0 && depth > 0) && lhs->aligned == nullptr) ||
      ((depth > 0 && columns > 0) && rhs->aligned == nullptr) ||
      (columns > 0 && bias->aligned == nullptr)) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }

  const int64_t tempDims[] = {rows, columns};
  AnncTensorView matmulView{};
  if (context.allocateTemp<float, 2>(tempDims, &matmulView) !=
      ANNC_STATUS_OK)
    return context.status();

  auto* add = context.allocateOutput<float>(0, {rows, columns});
  if (!add) return context.status();
  auto* relu = context.allocateOutput<float>(1, {rows, columns});
  if (!relu) return context.status();

  if (rows == 0 || columns == 0) {
    return context.status();
  }

  const float* lhsData =
      depth == 0 ? nullptr : annc::kernels::contextData<float>(*lhs);
  const float* rhsData =
      depth == 0 ? nullptr : annc::kernels::contextData<float>(*rhs);
  const float* biasData = annc::kernels::contextData<float>(*bias);
  float* matmulData = annc::kernels::contextData<float>(matmulView);
  float* addData = annc::kernels::contextData<float>(*add);
  float* reluData = annc::kernels::contextData<float>(*relu);

  try {
    auto workers = context.getThreadPool();
    workers.parallelFor(rows, [&](int64_t begin, int64_t end) {
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
          reluData[offset] = addData[offset] > 0.0f ? addData[offset] : 0.0f;
        }
      }
    });
  } catch (const std::exception&) {
    return ANNC_STATUS_INTERNAL;
  } catch (...) {
    return ANNC_STATUS_INTERNAL;
  }
  return context.status();
}

}  // namespace
