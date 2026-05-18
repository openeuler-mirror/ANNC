#include "Dialect/Atir/Passes/Patterns/PatternRegistry.h"
#include "llvm/Support/Debug.h"

namespace atir {

PatternRegistry &PatternRegistry::instance() {
  static PatternRegistry inst;
  return inst;
}

bool PatternRegistry::addCreator(llvm::StringRef name, PatternCreator creator) {
  std::lock_guard<std::mutex> lock(mutex);
  std::string nameStr = name.str();
  
  if (registeredNames.count(nameStr)) {
    llvm::dbgs() << "ANNC Warning: Pattern '" << name << "' already registered. Skipping.\n";
    return false;
  }
  
  llvm::dbgs() << "ANNC: Registering custom fusion pattern: " << name << "\n";
  creators.push_back(std::move(creator));
  registeredNames.insert(std::move(nameStr));
  return true;
}

void PatternRegistry::populatePatterns(mlir::RewritePatternSet &patterns) const {
  std::lock_guard<std::mutex> lock(mutex);
  llvm::dbgs() << "ANNC: Populating " << creators.size() 
               << " custom fusion patterns\n";
  for (const auto &creator : creators) {
    creator(patterns);
  }
}

} // namespace atir
