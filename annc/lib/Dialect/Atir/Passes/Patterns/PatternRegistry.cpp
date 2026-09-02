#include "Dialect/Atir/Passes/Patterns/PatternRegistry.h"

#include "Support/Log.h"

namespace atir {

PatternRegistry &PatternRegistry::instance() {
  static PatternRegistry inst;
  return inst;
}

bool PatternRegistry::addCreator(llvm::StringRef name, PatternCreator creator) {
  std::lock_guard<std::mutex> lock(mutex);
  std::string nameStr = name.str();

  if (registeredNames.count(nameStr)) {
    ANNC_LOG_WARN("fusion-pattern")
        << "Pattern '" << name.str() << "' already registered. Skipping.\n";
    return false;
  }

  ANNC_LOG_INFO("fusion-pattern")
      << "Registering custom fusion pattern: " << name.str() << "\n";
  creators.push_back(std::move(creator));
  registeredNames.insert(std::move(nameStr));
  return true;
}

void PatternRegistry::populatePatterns(mlir::RewritePatternSet &patterns,
                                       const CustomOpTypeFilter &filter) const {
  std::lock_guard<std::mutex> lock(mutex);
  ANNC_LOG_INFO("fusion-pattern")
      << "Populating " << creators.size() << " custom fusion patterns\n";
  for (const auto &creator : creators) {
    creator(patterns, filter);
  }
}

}  // namespace atir
