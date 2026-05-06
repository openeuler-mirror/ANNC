#ifndef ANNC_PATTERNREGISTRY_H
#define ANNC_PATTERNREGISTRY_H

#include "mlir/IR/PatternMatch.h"
#include <functional>
#include <vector>
#include <string>
#include <unordered_set>
#include <mutex>
#include "llvm/ADT/StringRef.h"

namespace atir {

using PatternCreator = std::function<void(mlir::RewritePatternSet &)>;

class PatternRegistry {
 public:
  static PatternRegistry &instance();

  // Register with name for duplicate detection
  bool addCreator(llvm::StringRef name, PatternCreator creator);

  // Template helper
  template <typename PatternType>
  bool registerPattern(llvm::StringRef name) {
    return addCreator(name, [](mlir::RewritePatternSet &patterns) {
      patterns.add<PatternType>(patterns.getContext());
    });
  }

  void populatePatterns(mlir::RewritePatternSet &patterns) const;

 private:
  PatternRegistry() = default;

  mutable std::mutex mutex;
  std::vector<PatternCreator> creators;
  std::unordered_set<std::string> registeredNames;
};

} // namespace atir

#endif  // ANNC_PATTERNREGISTRY_H
