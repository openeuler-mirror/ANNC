#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

#include "Context/KernelContextUtils.h"

namespace {

constexpr int64_t kSmallInputThreshold = 64;
constexpr int64_t kParallelWorkThreshold = 16384;

inline uint64_t hashKey(int64_t key) {
  uint64_t value = static_cast<uint64_t>(key);
  value ^= value >> 30;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

AnncStatusCode invalid(annc::kernels::KernelContext &context,
                       const char *message) {
  return context.error(ANNC_STATUS_INVALID_ARGUMENT, message);
}

AnncStatusCode kpFusedGatherImpl(annc::kernels::KernelContext &context) {
  const auto *data = context.getInput(0);
  const auto *keys = context.getInput(1);
  const auto *begin = context.getInput(2);
  if (!data || !keys || !begin) return context.status();
  if (keys->rank != 2)
    return context.error(ANNC_STATUS_INTERNAL, "Input key must be 2D");
  if (data->rank != 2)
    return context.error(ANNC_STATUS_INTERNAL,
                         "Embedding table shape must be 2D");
  int64_t beginElements = 1;
  for (uint32_t i = 0; i < begin->rank; ++i) beginElements *= begin->sizes[i];
  if (beginElements != 2)
    return context.error(ANNC_STATUS_INTERNAL,
                         "begin must be same as keys rank");

  const int64_t rows = keys->sizes[0];
  const int64_t keyColumns = keys->sizes[1];
  const int64_t dataRows = data->sizes[0];
  const int64_t dataColumns = data->sizes[1];
  const int64_t selectedColumn = annc::kernels::contextData<int32_t>(*begin)[1];
  if (selectedColumn < 0 || selectedColumn >= keyColumns)
    return context.error(ANNC_STATUS_INTERNAL, "slice cols out of keys range");
  if (rows > std::numeric_limits<int32_t>::max())
    return invalid(context, "key row count exceeds int32 output range");
  if (dataColumns != 0 &&
      rows > std::numeric_limits<int64_t>::max() / dataColumns)
    return invalid(context, "output shape exceeds int64 range");

  auto *inverse = context.allocateOutput<int32_t>(1, {rows});
  if (!inverse) return context.status();

  const int64_t *keyData = annc::kernels::contextData<int64_t>(*keys);
  const float *dataData = annc::kernels::contextData<float>(*data);
  int32_t *inverseData = annc::kernels::contextData<int32_t>(*inverse);
  int64_t uniqueCount = 0;
  const int64_t *uniqueValuesData = nullptr;
  std::array<int64_t, kSmallInputThreshold> smallUniqueValues;
  AnncTensorView hashKeysView{};
  AnncTensorView hashIndicesView{};

  if (rows <= kSmallInputThreshold) {
    const int64_t *keyPtr = keyData;
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t key = keyPtr[selectedColumn];
      int64_t uniqueIndex = -1;
      for (int64_t i = 0; i < uniqueCount; ++i) {
        if (smallUniqueValues[i] == key) {
          uniqueIndex = i;
          break;
        }
      }
      if (uniqueIndex < 0) {
        if (key < 0 || key >= dataRows)
          return context.error(ANNC_STATUS_INTERNAL, "idx out of table range");
        smallUniqueValues[uniqueCount] = key;
        uniqueIndex = uniqueCount++;
      }
      inverseData[row] = static_cast<int32_t>(uniqueIndex);
      keyPtr += keyColumns;
    }
    uniqueValuesData = smallUniqueValues.data();
  } else {
    const uint64_t rowCount = static_cast<uint64_t>(rows);
    if (rowCount > (std::numeric_limits<uint64_t>::max() - 1) / 2)
      return invalid(context, "key row count is too large");
    const uint64_t requiredSlots = rowCount * 2 + 1;
    uint64_t capacity = 1;
    while (capacity < requiredSlots) {
      if (capacity > (std::numeric_limits<uint64_t>::max() >> 1))
        return invalid(context, "hash table size overflow");
      capacity <<= 1;
    }
    if (capacity > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        capacity > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) -
                       rowCount)
      return invalid(context, "hash table size overflow");

    const int64_t keyTempDims[] = {static_cast<int64_t>(capacity + rowCount)};
    const int64_t indexTempDims[] = {static_cast<int64_t>(capacity)};
    if (context.allocateTemp<int64_t, 1>(keyTempDims, &hashKeysView) !=
            ANNC_STATUS_OK ||
        context.allocateTemp<int32_t, 1>(indexTempDims, &hashIndicesView) !=
            ANNC_STATUS_OK)
      return context.status();
    auto *tableKeys = annc::kernels::contextData<int64_t>(hashKeysView);
    auto *tableIndices = annc::kernels::contextData<int32_t>(hashIndicesView);
    auto *uniqueValues = tableKeys + capacity;
    std::fill(tableIndices, tableIndices + capacity, int32_t(-1));
    const uint64_t mask = capacity - 1;
    const int64_t *keyPtr = keyData;
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t key = keyPtr[selectedColumn];
      uint64_t slot = hashKey(key) & mask;
      while (tableIndices[slot] >= 0 && tableKeys[slot] != key)
        slot = (slot + 1) & mask;
      if (tableIndices[slot] < 0) {
        if (key < 0 || key >= dataRows)
          return context.error(ANNC_STATUS_INTERNAL, "idx out of table range");
        if (uniqueCount >= std::numeric_limits<int32_t>::max())
          return invalid(context,
                         "unique key count exceeds int32 output range");
        tableKeys[slot] = key;
        tableIndices[slot] = static_cast<int32_t>(uniqueCount);
        uniqueValues[uniqueCount++] = key;
      }
      inverseData[row] = tableIndices[slot];
      keyPtr += keyColumns;
    }
    uniqueValuesData = uniqueValues;
  }

  auto *values = context.allocateOutput<int64_t>(0, {uniqueCount});
  if (!values) return context.status();
  if (uniqueCount)
    std::memcpy(annc::kernels::contextData<int64_t>(*values), uniqueValuesData,
                static_cast<size_t>(uniqueCount) * sizeof(int64_t));
  auto *gathered = context.allocateOutput<float>(2, {uniqueCount, dataColumns});
  if (!gathered) return context.status();
  if (!uniqueCount || !dataColumns) return context.status();

  float *gatheredData = annc::kernels::contextData<float>(*gathered);
  const auto work = [&](int64_t first, int64_t last) {
    for (int64_t row = first; row < last; ++row) {
      const float *source = dataData + uniqueValuesData[row] * dataColumns;
      std::memcpy(gatheredData + row * dataColumns, source,
                  static_cast<size_t>(dataColumns) * sizeof(float));
    }
  };
  const bool enoughWork =
      uniqueCount > 1 &&
      (dataColumns >= kParallelWorkThreshold ||
       uniqueCount >= (kParallelWorkThreshold + dataColumns - 1) / dataColumns);
  if (enoughWork) {
    auto workers = context.getThreadPool();
    const int64_t costPerUnit = std::max<int64_t>(dataColumns, 1);
    return annc::kernels::Shard(workers.numThreads(), workers, uniqueCount,
                                costPerUnit, work);
  }
  work(0, uniqueCount);
  return context.status();
}

}  // namespace
