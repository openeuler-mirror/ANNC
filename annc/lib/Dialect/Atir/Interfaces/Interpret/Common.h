#ifndef ATIR_INTERPRET_COMMON_H
#define ATIR_INTERPRET_COMMON_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
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

/// True if the tensor carries string elements (encoding == "string").  The
/// elementType is a `complex<f32>` placeholder in that case; this checks the
/// encoding attribute (the real marker), mirroring the lowering in
/// InputTypeConverter / AtirTypeConverter.
bool isStringTensor(atir::TensorType tensorType);

/// Read a DenseStringElementsAttr as a vector of strings. Returns failure if
/// the attr is not a string elements attr (so non-string-aware ops that call
/// this on a numeric attr fail cleanly).
FailureOr<std::vector<std::string>> getStringValues(DenseElementsAttr attr);

/// Write string values as a DenseStringElementsAttr cacheData on resultType.
/// resultType's elementType must be non-int/float (e.g. the complex<f32>
/// string placeholder) so MLIR allows string elements.
LogicalResult setStringResult(atir::TensorType resultType,
                              ArrayRef<int64_t> outputShape,
                              ArrayRef<std::string> values);

/// Concretize a result's (dynamic) type shape to a static `shape` computed at
/// runtime (e.g. a sparse op's NNZ, which the inferred type leaves as `?`).
/// Creates a static atir::TensorType carrying the result's other params +
/// existing cacheData, binds it to the result via setType, and returns the
/// static type (pass it to the subsequent setDense*Result/setStringResult so
/// cacheData lands on the static type). Returns null if the result isn't an
/// atir::TensorType.
atir::TensorType concretizeResultType(mlir::Value result,
                                      ArrayRef<int64_t> shape);

SmallVector<int64_t> getIntArrayAttrValues(ArrayAttr attr);

/// Resolve dynamic dimensions in a shape using the actual element count.
/// If exactly one dimension is kDynamic, infer it from numElements / product
/// of the static dimensions.
SmallVector<int64_t> resolveDynamicShape(ArrayRef<int64_t> outputShape,
                                         int64_t numElements);

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