#ifndef ATIR_INTERPRET_COMMON_H
#define ATIR_INTERPRET_COMMON_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "Dialect/Atir/AtirOps.h"
#include "Helper.h"

namespace atir {
namespace interpret {

bool tensorTypeIsBoolAsI32(atir::TensorType resultType);

int64_t getElementCount(ArrayRef<int64_t> shape);

// 将扁平索引转换为多维索引
SmallVector<int64_t> getMultiIndex(ArrayRef<int64_t> shape, int64_t flatIndex);

// 将多维索引转换为扁平索引
int64_t getFlatIndex(ArrayRef<int64_t> shape, ArrayRef<int64_t> index);

// 获取广播后的输入索引
SmallVector<int64_t> getBroadcastIndex(ArrayRef<int64_t> outputShape,
                                       ArrayRef<int64_t> inputShape,
                                       ArrayRef<int64_t> outputIndex);

LogicalResult getTensorTypeAndData(Operation *op, Value value, StringRef name,
                                   atir::TensorType &tensorType,
                                   DenseElementsAttr &attr);

FailureOr<std::vector<float>> getFloatValues(DenseElementsAttr attr);

FailureOr<SmallVector<int64_t>> getIntValues(DenseElementsAttr attr);

SmallVector<int64_t> getIntArrayAttrValues(ArrayAttr attr);

Attribute getZeroElementAttr(Type elementType, MLIRContext *context);

LogicalResult setDenseResult(atir::TensorType resultType,
                             ArrayRef<int64_t> outputShape,
                             ArrayRef<float> values);

LogicalResult setDenseIntResult(atir::TensorType resultType,
                                ArrayRef<int64_t> outputShape,
                                ArrayRef<int64_t> values);

LogicalResult setBooleanLikeResult(atir::TensorType resultType,
                                   ArrayRef<int64_t> outputShape,
                                   ArrayRef<int64_t> values);

/// 从输入和输出形状推断归约轴
std::vector<int64_t> inferReductionAxesFromShapes(ArrayRef<int64_t> inputShape,
                                                  ArrayRef<int64_t> outputShape);

/// 应用 ReLU 限制
float applyReluLimit(float value, bool doRelu, float reluLimit);

} // namespace interpret
} // namespace atir

#endif // ATIR_INTERPRET_COMMON_H