#ifndef ANNC_PATTERNREGISTRY_H
#define ANNC_PATTERNREGISTRY_H

#include <functional>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"

namespace atir {

// Controls which final custom op types a fusion pattern may materialize.
// An empty allowlist enables all types; the denylist always takes precedence.
class CustomOpTypeFilter {
 public:
  CustomOpTypeFilter() = default;
  CustomOpTypeFilter(llvm::ArrayRef<std::string> enabled,
                     llvm::ArrayRef<std::string> disabled)
      : enabledTypes(enabled.begin(), enabled.end()),
        disabledTypes(disabled.begin(), disabled.end()) {}

  bool isEnabled(llvm::StringRef opType) const {
    if (!disabledTypes.empty() && contains(disabledTypes, opType)) return false;
    return enabledTypes.empty() || contains(enabledTypes, opType);
  }

 private:
  static bool contains(llvm::ArrayRef<std::string> types,
                       llvm::StringRef opType) {
    return llvm::any_of(types, [&](const std::string &type) {
      return llvm::StringRef(type) == opType;
    });
  }

  llvm::SmallVector<std::string, 4> enabledTypes;
  llvm::SmallVector<std::string, 4> disabledTypes;
};

using PatternCreator =
    std::function<void(mlir::RewritePatternSet &, const CustomOpTypeFilter &)>;

class PatternRegistry {
 public:
  static PatternRegistry &instance();

  // Register with name for duplicate detection
  bool addCreator(llvm::StringRef name, PatternCreator creator);

  // Template helper
  template <typename PatternType>
  bool registerPattern(llvm::StringRef name) {
    return addCreator(name, [](mlir::RewritePatternSet &patterns,
                               const CustomOpTypeFilter &filter) {
      patterns.add<PatternType>(patterns.getContext(), filter);
    });
  }

  void populatePatterns(mlir::RewritePatternSet &patterns,
                        const CustomOpTypeFilter &filter) const;

 private:
  PatternRegistry() = default;

  mutable std::mutex mutex;
  std::vector<PatternCreator> creators;
  std::unordered_set<std::string> registeredNames;
};

}  // namespace atir

#endif  // ANNC_PATTERNREGISTRY_H
