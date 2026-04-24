#ifndef PIMP_DIALECT_H
#define PIMP_DIALECT_H
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "Dialect/Pimp/PimpOpTrait.h"
#include "Dialect/Pimp/PimpDialect.h.inc"
#include "Dialect/Pimp/PimpEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/Pimp/PimpAttr.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Pimp/PimpTypes.h.inc"

#include "Dialect/Pimp/Interfaces/Interfaces.h"

#define GET_OP_CLASSES
#include "Dialect/Pimp/PimpOps.h.inc"

using namespace mlir;
namespace pimp {
TensorType getPimpTensorType(MLIRContext *ctx);
ParseResult parseSwitchCases(OpAsmParser &p, DenseI64ArrayAttr &cases,
                 SmallVectorImpl<std::unique_ptr<Region>> &caseRegions);
void printSwitchCases(OpAsmPrinter &p, Operation *op, DenseI64ArrayAttr cases,
                      RegionRange caseRegions);
}  // namespace pimp

#endif  // PIMP_DIALECT_H