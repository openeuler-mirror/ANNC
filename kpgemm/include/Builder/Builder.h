#ifndef KPGEMM_BUILDER_H
#define KPGEMM_BUILDER_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "Builder/MLIROpBuilder.h"
#include "Helper.h"

using json = nlohmann::json;
using namespace mlir;
namespace kpgemm {
class KPGEMMBuilder {
 public:
  KPGEMMBuilder(MLIRContext *context) : context_(context), builder_(context) {}
  virtual ~KPGEMMBuilder() = default;

  ModuleOp module_;
  func::FuncOp func_;

  ModuleOp buildModule(std::string name, const json& graph);

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
}  // namespace kpgemm

#endif  // KPGEMM_BUILDER_H