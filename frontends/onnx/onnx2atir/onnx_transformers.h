#ifndef ANNC_ONNX2ATIR_ONNX_TRANSFORMERS_H
#define ANNC_ONNX2ATIR_ONNX_TRANSFORMERS_H

#include "Builder/OpSpec.h"
#include "llvm/ADT/ArrayRef.h"

namespace annc::onnx2atir {

// Non-1:1 ONNX op transformers. Same contract as the annc::transform* family
// in annc/include/Builder/Transformers.h: build exactly the ATIR op for the
// node, bind every produced value through the context, and report and return
// failure() on every error path.
mlir::LogicalResult onnxTransformTranspose(const NodeInfo&,
                                           llvm::ArrayRef<mlir::Type>,
                                           llvm::ArrayRef<mlir::Value>,
                                           OpContext&);
mlir::LogicalResult onnxTransformReduce(const NodeInfo&,
                                        llvm::ArrayRef<mlir::Type>,
                                        llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult onnxTransformGather(const NodeInfo&,
                                        llvm::ArrayRef<mlir::Type>,
                                        llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult onnxTransformNot(const NodeInfo&, llvm::ArrayRef<mlir::Type>,
                                     llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult onnxTransformSlice(const NodeInfo&,
                                       llvm::ArrayRef<mlir::Type>,
                                       llvm::ArrayRef<mlir::Value>, OpContext&);

}  // namespace annc::onnx2atir

#endif  // ANNC_ONNX2ATIR_ONNX_TRANSFORMERS_H
