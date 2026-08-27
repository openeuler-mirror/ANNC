#ifndef ANNC_DIALECT_ATIR_PASSES_FUSION_CONFIG_FUSION_SUPPORT_H
#define ANNC_DIALECT_ATIR_PASSES_FUSION_CONFIG_FUSION_SUPPORT_H

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "Dialect/Atir/AtirOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace atir::config_fusion {

using llvm::SmallVector;
using mlir::Operation;
using mlir::Value;

// Structured representation of one graph DSL statement, for example:
//   out = MatMul(lhs, rhs)
//   (values, indices) = TopK(input, k)
struct PatternStmt {
  // Output aliases on the left-hand side; "_" means the result is not bound.
  SmallVector<std::string> aliases;
  // Normalized ATIR op simple name, such as MatMul or ConcatV2.
  std::string opName;
  // Input aliases on the right-hand side; they may refer to external inputs
  // or outputs defined by preceding statements.
  SmallVector<std::string> args;
};

// Capture an attribute from a matched op and write it to
// fusion.metadata.pattern_attrs.
struct CaptureSpec {
  // Key written to metadata.
  std::string key;
  // Op output alias defined by the graph DSL.
  std::string alias;
  // MLIR op attribute name, which is case-sensitive.
  std::string attr;
};

// One user-configured fusion rule. The parser only converts JSON/DSL into this
// structure; materialization and matching validate semantics against the IR.
struct PatternRule {
  std::string name;
  std::string pattern;
  std::string kernel;
  // Optional backend ABI for patterns that are consumed by an existing
  // lowering path.  The default preserves the generic mlir_ciface outline.
  std::string abi = "mlir_ciface";
  int64_t priority = 0;
  SmallVector<PatternStmt> stmts;
  // Output alias -> defining graph statement index, used for reverse def-use
  // traversal.
  std::map<std::string, unsigned> aliasToStmt;
  SmallVector<std::string> inputs;
  SmallVector<std::string> outputs;
  std::map<std::string, std::string> attrs;
  SmallVector<CaptureSpec> captures;
  SmallVector<std::string> where;
  int64_t specificity = 0;
};

struct FusionConfig {
  int64_t version = 0;
  SmallVector<PatternRule, 4> patterns;
};

// Bindings collected during one pattern match. Matching traces from outputs
// back to inputs, so aliases, external values, and matched ops are recorded.
struct MatchState {
  // SSA value bound to an internal graph DSL alias.
  std::map<std::string, Value> aliasValues;
  // SSA value bound to an external graph DSL input.
  std::map<std::string, Value> externalValues;
  // Graph statement index -> operation matched in the IR.
  llvm::DenseMap<unsigned, Operation *> stmtOps;
  // First-seen order of external inputs, used when inputs are omitted.
  SmallVector<std::string> externalOrder;
  // Discovery order during matching; reordered by block order before materialize.
  SmallVector<Operation *> matchedOrder;
  // Matched op set, used to identify boundary values and escaping uses.
  std::set<Operation *> matchedSet;
};

enum class SkipReason {
  ConfigurationError,
  StructureMismatch,
  ConstraintMismatch,
  CaptureFailure,
  InputMismatch,
  BoundaryEscape,
  MaterializationFailure,
};

struct SkipSample {
  std::string reason;
  std::string message;
  std::string anchor;
  std::string statement;
  std::string ir;
  std::string mismatchIr;
};

// Per-pattern diagnostics are collected independently from the matcher result
// so observability never changes fusion ordering or rewrite semantics.
struct PatternReport {
  explicit PatternReport(std::string patternName)
      : name(std::move(patternName)) {}

  void recordAnchor(Operation *anchor);
  void recordSkip(SkipReason reason, Operation *anchor, llvm::StringRef message,
                  int64_t sampleLimit, llvm::StringRef statement = {},
                  Operation *mismatch = nullptr);

  std::string name;
  int64_t anchors = 0;
  int64_t matches = 0;
  std::map<std::string, int64_t> skips;
  SmallVector<SkipSample> samples;
  SmallVector<std::string> outlined;
  std::map<std::string, std::string> inferredRoles;

 private:
  std::set<Operation *> seenAnchors;
  std::map<Operation *, std::set<SkipReason>> seenSkips;
};

bool configFusionDebugEnabled();
llvm::StringRef stringifySkipReason(SkipReason reason);
void emitPatternWarnings(mlir::func::FuncOp mainFunc,
                         const std::vector<PatternReport> &reports);
llvm::Error writeFusionReport(llvm::StringRef path, llvm::StringRef configPath,
                              const std::vector<PatternReport> &reports);

template <typename Fn>
void debugLog(Fn &&fn) {
  if (!configFusionDebugEnabled()) return;
  llvm::errs() << "[atir-config-fusion] ";
  fn(llvm::errs());
  llvm::errs() << "\n";
}

std::string trim(std::string value);
std::vector<std::string> splitTopLevel(llvm::StringRef text, char delimiter);
std::string parseQuotedString(llvm::StringRef value);
bool parseInteger(llvm::StringRef value, int64_t *out);
std::string simpleOpName(mlir::Operation *op);
// Normalize configured op names case-insensitively; preserve unknown spellings.
std::string canonicalConfigOpName(llvm::StringRef opName);
std::string sanitizeName(std::string name);
std::string uniquifySymbolName(mlir::ModuleOp module, llvm::StringRef baseName);
std::string getValueName(mlir::Value value);
// Recover a TensorFlow endpoint name for an op result. Multi-result producers
// need the result slot suffix when the source metadata only provides a node
// name (for example, TopK:0 and TopK:1).
std::string getFusionOutputName(mlir::Value value);
std::string getTensorDType(mlir::Type type);
int64_t getRank(mlir::Type type);
std::vector<int64_t> getShape(mlir::Type type);
std::string readAttrAsString(mlir::Attribute attr);
bool parseCaptureEntry(llvm::StringRef entry, std::string *key,
                       std::pair<std::string, std::string> *spec);
llvm::Expected<FusionConfig> parseConfigFile(const std::string &path);
std::vector<std::string> inferOutputs(const PatternRule &rule);
std::vector<std::string> inferInputs(const PatternRule &rule,
                                     const MatchState &state);
bool collectPatternAttrs(const PatternRule &rule, const MatchState &state,
                         std::map<std::string, std::string> *attrs);
bool hasUnlistedEscapingOutput(const PatternRule &rule,
                               const std::vector<std::string> &outputs,
                               const MatchState &state);
mlir::func::FuncOp materializePattern(
    mlir::ModuleOp module, PatternRule rule, const MatchState &state,
    const std::vector<std::string> &inputs,
    const std::vector<std::string> &outputs,
    const std::map<std::string, std::string> &patternAttrs);
bool tryMatchRule(mlir::ModuleOp module, mlir::func::FuncOp mainFunc,
                  const PatternRule &rule, PatternReport *report,
                  int64_t warnLimit);

}  // namespace atir::config_fusion

#endif  // ANNC_DIALECT_ATIR_PASSES_FUSION_CONFIG_FUSION_SUPPORT_H
