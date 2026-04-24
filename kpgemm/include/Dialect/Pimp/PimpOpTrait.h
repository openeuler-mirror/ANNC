#ifndef PIMP_OP_TRAIT_H
#define PIMP_OP_TRAIT_H

namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class Pimp_OpTrait : public TraitBase<ConcreteType, Pimp_OpTrait> {
 public:
  std::string getNodeName() {
    auto *concrete = static_cast<ConcreteType *>(this);
    auto loc = concrete->getLoc().template dyn_cast<::mlir::NameLoc>();
    if (loc == nullptr) return "unknown";
    return loc.getName().str();
  }
  size_t getOpIndex() {
    auto *concrete = static_cast<ConcreteType *>(this);
    auto *parent = concrete->getParentOp();
    if (parent == nullptr) return 0;
    auto *region = concrete->getParentRegion();
    if (region == nullptr) return 0;
    auto &ops = region->getOps();
    for (size_t i = 0; i < ops.size(); ++i) {
      if (ops[i] == parent) return i;
    }
    return 0;
  }
  uint64_t getOutputElements() {
    auto *concrete = static_cast<ConcreteType *>(this);
    return getNumElements(concrete->getResult());
  }
  uint64_t getInputElements(size_t index) {
    auto *concrete = static_cast<ConcreteType *>(this);
    if (index >= concrete->getNumOperands()) return 0;
    return getNumElements(concrete->getOperand(index));
  }
  uint64_t getNumElements(const Value& val) {
    return 0;
  }
};

}  // namespace OpTrait
}  // namespace mlir

#endif  // PIMP_OP_TRAIT_H