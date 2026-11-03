#include <cstdint>
#include <limits>

#include "Context/KernelContextUtils.h"

namespace {

bool multiplyFits(int64_t lhs, int64_t rhs, int64_t* product) {
  if (lhs < 0 || rhs < 0 ||
      (lhs && rhs > std::numeric_limits<int64_t>::max() / lhs))
    return false;
  *product = lhs * rhs;
  return true;
}

// This kernel writes two int64 values per input row. Scheduling small calls
// costs more than the transform itself.
constexpr int64_t kParallelRowThreshold = 4096;

template <typename Work>
AnncStatusCode RunByRows(annc::kernels::KernelContext& context,
                         int64_t row_count, int64_t cost_per_row,
                         const Work& work) {
  if (row_count >= kParallelRowThreshold) {
    const auto workers = context.getThreadPool();
    if (workers.numThreads() > 1) {
      return annc::kernels::Shard(workers.numThreads(), workers, row_count,
                                  cost_per_row, work);
    }
  }
  work(0, row_count);
  return context.status();
}

AnncStatusCode WriteSingleColumnIndices(const int64_t* slice_data,
                                        int64_t row_count, int64_t slice_cols,
                                        int64_t selected_col,
                                        int64_t pack_const, int64_t* output,
                                        annc::kernels::KernelContext& context) {
  auto work = [&](int64_t start_row, int64_t end_row) {
    const int64_t* selected =
        slice_data + start_row * slice_cols + selected_col;
    int64_t input_offset = start_row * pack_const;
    int64_t* output_row = output + start_row * 2;
    for (int64_t row = start_row; row < end_row; ++row) {
      output_row[0] = input_offset + *selected;
      output_row[1] = 0;
      selected += slice_cols;
      input_offset += pack_const;
      output_row += 2;
    }
  };
  return RunByRows(context, row_count, 4, work);
}

AnncStatusCode WriteIdentityIndices(const int64_t* slice_data,
                                    int64_t row_count, int64_t slice_cols,
                                    int64_t selected_col, int64_t* output,
                                    annc::kernels::KernelContext& context) {
  auto work = [&](int64_t start_row, int64_t end_row) {
    const int64_t* selected =
        slice_data + start_row * slice_cols + selected_col;
    int64_t* output_row = output + start_row * 2;
    for (int64_t row = start_row; row < end_row; ++row) {
      output_row[0] = row;
      output_row[1] = *selected;
      selected += slice_cols;
      output_row += 2;
    }
  };
  return RunByRows(context, row_count, 4, work);
}

template <int64_t OutputCols>
AnncStatusCode WriteFixedWidthIndices(const int64_t* slice_data,
                                      int64_t row_count, int64_t slice_cols,
                                      int64_t selected_col, int64_t pack_const,
                                      int64_t* output,
                                      annc::kernels::KernelContext& context) {
  static_assert(OutputCols > 1, "single-column output has a cheaper path");
  auto work = [&](int64_t start_row, int64_t end_row) {
    const int64_t* selected =
        slice_data + start_row * slice_cols + selected_col;
    int64_t input_offset = start_row * pack_const;
    int64_t* output_row = output + start_row * 2;
    for (int64_t row = start_row; row < end_row; ++row) {
      const int64_t linear_index = input_offset + *selected;
      const int64_t output_row_index = linear_index / OutputCols;
      output_row[0] = output_row_index;
      output_row[1] = linear_index - output_row_index * OutputCols;
      selected += slice_cols;
      input_offset += pack_const;
      output_row += 2;
    }
  };
  return RunByRows(context, row_count, 8, work);
}

AnncStatusCode WriteGenericIndices(const int64_t* slice_data, int64_t row_count,
                                   int64_t slice_cols, int64_t selected_col,
                                   int64_t pack_const, int64_t output_cols,
                                   int64_t* output,
                                   annc::kernels::KernelContext& context) {
  auto work = [&](int64_t start_row, int64_t end_row) {
    const int64_t* selected =
        slice_data + start_row * slice_cols + selected_col;
    int64_t input_offset = start_row * pack_const;
    int64_t* output_row = output + start_row * 2;
    for (int64_t row = start_row; row < end_row; ++row) {
      const int64_t linear_index = input_offset + *selected;
      const int64_t output_row_index = linear_index / output_cols;
      output_row[0] = output_row_index;
      output_row[1] = linear_index - output_row_index * output_cols;
      selected += slice_cols;
      input_offset += pack_const;
      output_row += 2;
    }
  };
  return RunByRows(context, row_count, 8, work);
}

AnncStatusCode kpFusedSparseReshapeImpl(
    annc::kernels::KernelContext& context) {
  const auto* sliceInput = context.getInput(0);
  const auto* begin = context.getInput(1);
  const auto* packConst = context.getInput(2);
  const auto* newShape = context.getInput(3);
  if (!sliceInput || !begin || !packConst || !newShape) return context.status();
  if (sliceInput->rank != 2 || begin->rank != 1 || begin->sizes[0] != 2 ||
      packConst->rank != 0 || newShape->rank != 1 || newShape->sizes[0] != 2)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "SparseReshape input rank mismatch");

  const int64_t rowCount = sliceInput->sizes[0];
  const int64_t sliceCols = sliceInput->sizes[1];
  const int64_t selectedColumn = annc::kernels::contextData<int32_t>(*begin)[1];
  if (selectedColumn < 0 || selectedColumn >= sliceCols)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "SparseReshape begin column out of range");
  if (packConst->element_type != ANNC_ELEMENT_TYPE_I64)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "SparseReshape pack_const must be i64");
  const int64_t packValue =
      *annc::kernels::contextData<int64_t>(*packConst);
  if (packValue < 0) {
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "SparseReshape pack_const must be non-negative");
  }

  int64_t denseSize = 0;
  if (!multiplyFits(rowCount, packValue, &denseSize))
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "SparseReshape input shape size overflow");

  const int64_t* requested = annc::kernels::contextData<int64_t>(*newShape);
  int64_t outputShape[2];
  int64_t specifiedProduct = 1;
  int unknownIndex = -1;
  for (int i = 0; i < 2; ++i) {
    const int64_t size = requested[i];
    if (size == -1) {
      if (unknownIndex != -1)
        return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                             "SparseReshape has multiple unknown dimensions");
      unknownIndex = i;
      outputShape[i] = 1;
    } else {
      if (size < 0)
        return context.error(
            ANNC_STATUS_INVALID_ARGUMENT,
            "SparseReshape requested dimension must be non-negative");
      if (!multiplyFits(specifiedProduct, size, &specifiedProduct))
        return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                             "SparseReshape requested shape size overflow");
      outputShape[i] = size;
    }
  }
  if (unknownIndex != -1) {
    if (specifiedProduct <= 0)
      return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                           "SparseReshape cannot infer empty dimension");
    const int64_t missing = denseSize / specifiedProduct;
    int64_t reconstructed = 0;
    if (!multiplyFits(specifiedProduct, missing, &reconstructed) ||
        reconstructed != denseSize)
      return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                           "SparseReshape inferred shape mismatch");
    outputShape[unknownIndex] = missing;
  }
  int64_t outputSize = 0;
  if (!multiplyFits(outputShape[0], outputShape[1], &outputSize) ||
      outputSize != denseSize)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "SparseReshape output shape mismatch");

  auto* outputIndices = context.allocateOutput<int64_t>(0, {rowCount, 2});
  if (!outputIndices) return context.status();
  auto* outputShapeView = context.allocateOutput<int64_t>(1, {2});
  if (!outputShapeView) return context.status();

  int64_t* outputShapeData =
      annc::kernels::contextData<int64_t>(*outputShapeView);
  outputShapeData[0] = outputShape[0];
  outputShapeData[1] = outputShape[1];
  if (rowCount == 0) return context.status();

  const int64_t* sliceData = annc::kernels::contextData<int64_t>(*sliceInput);
  int64_t* outputIndexData =
      annc::kernels::contextData<int64_t>(*outputIndices);
  if (outputShape[0] == rowCount && outputShape[1] == packValue) {
    return WriteIdentityIndices(sliceData, rowCount, sliceCols, selectedColumn,
                                outputIndexData, context);
  }
  if (outputShape[1] == 0)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "SparseReshape cannot map to zero-width output");

  switch (outputShape[1]) {
    case 1:
      // CVR, HMV and Presort all use [-1, 1].
      return WriteSingleColumnIndices(sliceData, rowCount, sliceCols,
                                      selectedColumn, packValue,
                                      outputIndexData, context);
    case 492:
      // ADX wide embedding branch.
      return WriteFixedWidthIndices<492>(sliceData, rowCount, sliceCols,
                                         selectedColumn, packValue,
                                         outputIndexData, context);
    case 1668:
      // ADX deep embedding branch.
      return WriteFixedWidthIndices<1668>(sliceData, rowCount, sliceCols,
                                          selectedColumn, packValue,
                                          outputIndexData, context);
    default:
      return WriteGenericIndices(sliceData, rowCount, sliceCols, selectedColumn,
                                 packValue, outputShape[1], outputIndexData,
                                 context);
  }
}


}  // namespace
