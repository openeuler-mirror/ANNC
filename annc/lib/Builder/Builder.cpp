#include "Builder/Builder.h"

using json = nlohmann::json;
using namespace mlir;
namespace annc {
ModuleOp ANNCBuilder::buildModule(std::string name, const json& graph) {
  auto unknownLoc = UnknownLoc::get(context_);
  module_ = builder_.create<ModuleOp>(unknownLoc, name);
  module_->setAttr("module.state", builder_.getStringAttr("atir"));
  
  auto cvtr = std::make_shared<MLIRBuilder>(module_);
  
  cvtr->jsonConvertor(graph);
  return module_;
}

// ModuleOp ANNCBuilder::buildModule(std::string name, const std::vector<NodeInfo>& nodes) {
//     auto unknownLoc = UnknownLoc::get(context_);
//     auto module = ModuleOp::create(unknownLoc);
//     module->setAttr("module.name", StringAttr::get(context_, name));
//     module->setAttr("module.state", StringAttr::get(context_, "atir"));

//     //  MLIRBuilder module
//     MLIRBuilder cvtr(module); //  shared_ptr

//     printf("222\n");
//     cvtr.Convertor(name, nodes);

//     // 
//     module_ = module;

//     return module;  //  ModuleOp OwningOpRef 
// }
}  // namespace annc