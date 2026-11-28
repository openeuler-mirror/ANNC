#ifndef ANNC_TF2ATIR_TF_TRANSFORMERS_H
#define ANNC_TF2ATIR_TF_TRANSFORMERS_H

#include "Builder/OpSpec.h"
#include "llvm/ADT/ArrayRef.h"

namespace annc::tf2atir {

mlir::LogicalResult transformTranspose(const NodeInfo&,
                                       llvm::ArrayRef<mlir::Type>,
                                       llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformPack(const NodeInfo&, llvm::ArrayRef<mlir::Type>,
                                  llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformGather(const NodeInfo&, llvm::ArrayRef<mlir::Type>,
                                    llvm::ArrayRef<mlir::Value>, OpContext&);

}  // namespace annc::tf2atir
#endif
