#include "Dialect/Atir/Passes/Patterns/PatternRegistry.h"

#include <cstdlib>

#include "Support/Log.h"

namespace atir {

void appendCustomOpTypesFromEnv(const char *envName,
                                llvm::SmallVectorImpl<std::string> &types) {
  const char *value = std::getenv(envName);
  if (!value || !*value) return;

  llvm::SmallVector<llvm::StringRef, 4> entries;
  llvm::StringRef(value).split(entries, ',', /*MaxSplit=*/-1,
                               /*KeepEmpty=*/false);
  for (llvm::StringRef entry : entries) {
    entry = entry.trim();
    if (!entry.empty()) types.push_back(entry.str());
  }
}

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
