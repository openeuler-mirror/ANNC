#ifndef ATIR_DIALECT_H
#define ATIR_DIALECT_H
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Dialect/Traits.h"
// /annc-mlir/LLVM/mlir/include/mlir/Dialect/Traits.h

#include "Dialect/Atir/AtirOpTrait.h"
#include "Dialect/Atir/AtirDialect.h.inc"
#include "Dialect/Atir/AtirEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/Atir/AtirAttr.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Atir/AtirTypes.h.inc"

#include "Dialect/Atir/Interfaces/Interfaces.h"

#define GET_OP_CLASSES
#include "Dialect/Atir/AtirOps.h.inc"

using namespace mlir;
namespace atir {
TensorType getAtirTensorType(MLIRContext *ctx);
ParseResult parseSwitchCases(OpAsmParser &p, DenseI64ArrayAttr &cases,
                 SmallVectorImpl<std::unique_ptr<Region>> &caseRegions);
void printSwitchCases(OpAsmPrinter &p, Operation *op, DenseI64ArrayAttr cases,
                      RegionRange caseRegions);
}  // namespace atir

#endif  // ATIR_DIALECT_H