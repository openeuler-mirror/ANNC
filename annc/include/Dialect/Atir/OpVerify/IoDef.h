#ifndef DIALECT_ATIR_OPVERIFY_IODEF_H
#define DIALECT_ATIR_OPVERIFY_IODEF_H

#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <utility>

namespace atir {

/// @brief IoTensorDef manages input/output tensor definitions
class IoTensorDef {
public:
  using TensorData = std::pair<std::string, mlir::Value>;
  using TensorStorage = llvm::SmallVector<TensorData, 4>;

  void addInput(const std::string &name, mlir::Value value);
  const TensorStorage &getInputs() const;
  const TensorStorage &getOutputs() const;
  void addOutput(const std::string &name, mlir::Value value);
  bool isEmpty() const;
  size_t getInputCount() const;
  size_t getOutputCount() const;

private:
  TensorStorage inputs;
  TensorStorage outputs;
};

// Create I/O tensor definitions from MLIR module
void createIoDef(mlir::ModuleOp root, IoTensorDef *inputs, IoTensorDef *outputs,
                 const std::string &funcName);

// Get function name from library path
std::string getFuncNameFromPath(const std::string &pathValue);

} // namespace atir

#endif // DIALECT_ATIR_OPVERIFY_IODEF_H
