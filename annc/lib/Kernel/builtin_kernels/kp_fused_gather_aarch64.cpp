#include <cstdint>
#include <limits>

#include "Kernel/ExecutionContextUtils.h"
#include "Kernel/KernelStatus.h"
#include "Kernel/MemRefTypes.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/ThreadPool.h"

namespace {

bool contiguous1D(const AnncMemRef1DI32 &ref) {
  return ref.offset >= 0 && ref.sizes[0] >= 0 && ref.strides[0] == 1;
}
bool contiguous2D(const AnncMemRef2DF32 &ref) {
  return ref.offset >= 0 && ref.sizes[0] >= 0 && ref.sizes[1] >= 0 &&
         ref.strides[0] == ref.sizes[1] && ref.strides[1] == 1;
}
bool contiguous2D(const AnncMemRef2DI64 &ref) {
  return ref.offset >= 0 && ref.sizes[0] >= 0 && ref.sizes[1] >= 0 &&
         ref.strides[0] == ref.sizes[1] && ref.strides[1] == 1;
}

uint64_t hashKey(int64_t key) {
  uint64_t value = static_cast<uint64_t>(key);
  value ^= value >> 30;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

annc::kernels::KernelStatus kpFusedGatherImpl(
    annc::threadpool::AnncThreadPool *threadPool,
    const AnncExecutionContext *execution, AnncMemRef2DF32 *data,
    AnncMemRef2DI64 *keys, AnncMemRef1DI32 *begin) {
  if (!execution || !data || !keys || !begin || !contiguous2D(*data) ||
      !contiguous2D(*keys) || !contiguous1D(*begin) || begin->sizes[0] != 2)
    return annc::kernels::KernelStatus::InvalidArgument;

  const int64_t rows = keys->sizes[0];
  const int64_t keyColumns = keys->sizes[1];
  const int64_t dataRows = data->sizes[0];
  const int64_t dataColumns = data->sizes[1];
  const int64_t selectedColumn = ANNC_MEMREF_DATA(*begin)[1];
  if (selectedColumn < 0 || selectedColumn >= keyColumns || dataRows < 0 ||
      dataColumns < 0 || rows < 0 ||
      ((rows > 0 && keyColumns > 0) && !keys->aligned) ||
      ((dataRows > 0 && dataColumns > 0) && !data->aligned) ||
      (begin->aligned == nullptr))
    return annc::kernels::KernelStatus::InvalidArgument;

  if (static_cast<uint64_t>(rows) >
      (std::numeric_limits<uint64_t>::max() - 1) / 2)
    return annc::kernels::KernelStatus::InvalidArgument;
  const uint64_t required = static_cast<uint64_t>(rows) * 2 + 1;
  uint64_t capacity = 1;
  while (capacity < required) {
    if (capacity > (std::numeric_limits<uint64_t>::max() >> 1))
      return annc::kernels::KernelStatus::InvalidArgument;
    capacity <<= 1;
  }
  if (capacity > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      static_cast<uint64_t>(rows) >
          (static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) -
           capacity * 2))
    return annc::kernels::KernelStatus::InvalidArgument;
  const int64_t tempElements = static_cast<int64_t>(capacity * 2) + rows;
  auto scratch = annc::kernels::execution::allocateTemp1DI64(
      execution, tempElements, ANNC_ALLOCATION_NONE);
  if (!scratch) {
    llvm::consumeError(scratch.takeError());
    return annc::kernels::KernelStatus::RuntimeError;
  }
  auto inverse = annc::kernels::execution::allocateOutput1DI32(
      execution, 1, rows, ANNC_ALLOCATION_ZERO);
  if (!inverse) {
    llvm::consumeError(inverse.takeError());
    return annc::kernels::KernelStatus::RuntimeError;
  }

  int64_t *tableKeys = ANNC_MEMREF_DATA(*scratch);
  int64_t *tableValues = tableKeys + capacity;
  int64_t *uniqueValues = tableValues + capacity;
  for (uint64_t i = 0; i < capacity; ++i) tableValues[i] = 0;

  const int64_t *keyData = ANNC_MEMREF_DATA(*keys);
  int32_t *inverseData = ANNC_MEMREF_DATA(*inverse);
  int64_t uniqueCount = 0;
  const uint64_t mask = capacity - 1;
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t key = keyData[row * keyColumns + selectedColumn];
    uint64_t position = hashKey(key) & mask;
    while (tableValues[position] != 0 && tableKeys[position] != key) {
      position = (position + 1) & mask;
    }
    if (tableValues[position] == 0) {
      if (key < 0 || key >= dataRows ||
          uniqueCount == std::numeric_limits<int32_t>::max())
        return annc::kernels::KernelStatus::InvalidArgument;
      uniqueValues[uniqueCount] = key;
      tableKeys[position] = key;
      tableValues[position] = uniqueCount + 1;
      ++uniqueCount;
    }
    inverseData[row] = static_cast<int32_t>(tableValues[position] - 1);
  }

  auto values = annc::kernels::execution::allocateOutput1DI64(
      execution, 0, uniqueCount, ANNC_ALLOCATION_NONE);
  if (!values) {
    llvm::consumeError(values.takeError());
    return annc::kernels::KernelStatus::RuntimeError;
  }
  auto gathered = annc::kernels::execution::allocateOutput2DF32(
      execution, 2, uniqueCount, dataColumns, ANNC_ALLOCATION_NONE);
  if (!gathered) {
    llvm::consumeError(gathered.takeError());
    return annc::kernels::KernelStatus::RuntimeError;
  }
  int64_t *valuesData = ANNC_MEMREF_DATA(*values);
  for (int64_t i = 0; i < uniqueCount; ++i) valuesData[i] = uniqueValues[i];

  if (uniqueCount == 0 || dataColumns == 0)
    return annc::kernels::KernelStatus::Success;
  float *gatheredData = ANNC_MEMREF_DATA(*gathered);
  const float *dataData = ANNC_MEMREF_DATA(*data);
  try {
    annc::threadpool::parallel_for(
        threadPool, uniqueCount, [&](int64_t first, int64_t last) {
          for (int64_t i = first; i < last; ++i) {
            const float *source = dataData + uniqueValues[i] * dataColumns;
            float *destination = gatheredData + i * dataColumns;
            for (int64_t column = 0; column < dataColumns; ++column)
              destination[column] = source[column];
          }
        });
  } catch (...) {
    return annc::kernels::KernelStatus::RuntimeError;
  }
  return annc::kernels::KernelStatus::Success;
}

}  // namespace
