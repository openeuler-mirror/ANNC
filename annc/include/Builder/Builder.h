#ifndef ANNC_BUILDER_H
#define ANNC_BUILDER_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "Builder/MLIROpBuilder.h"
#include "Helper.h"

using namespace mlir;
namespace annc {
class ANNCBuilder {
 public:
  ANNCBuilder(MLIRContext *context) : context_(context), builder_(context) {}
  virtual ~ANNCBuilder() = default;

  ModuleOp module_;
  func::FuncOp func_;

  ModuleOp buildModule(std::string name, const std::vector<NodeInfo>& nodes);

  void save(const std::string& output) {
    outputCode(module_, output);
  }
  void saveBinary(const std::string& output) {
    outputBinary(module_, output);
  }

 private:
  MLIRContext *context_;
  OpBuilder builder_;

};
}  // namespace annc

#endif  // ANNC_BUILDER_H