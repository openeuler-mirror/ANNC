#include "Builder/Builder.h"

using namespace mlir;
namespace annc {

ModuleOp ANNCBuilder::buildModule(std::string name, const std::vector<NodeInfo>& nodes) {
  auto unknownLoc = UnknownLoc::get(context_);
  module_ = builder_.create<ModuleOp>(unknownLoc, name);
  module_->setAttr("module.state", builder_.getStringAttr("atir"));
  
  auto cvtr = std::make_shared<MLIRBuilder>(module_);
  cvtr->buildFromNodes(nodes);
  
  return module_;
}

}  // namespace annc