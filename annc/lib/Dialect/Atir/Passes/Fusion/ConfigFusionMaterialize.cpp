#include "ConfigFusionSupport.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <set>
#include <unordered_set>

#include "FusionMetadata/FusionMetadata.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

using namespace llvm;
using namespace mlir;
using namespace atir;

namespace atir::config_fusion {

// Serialize an MLIR Attribute into a lightweight string for where checks and
// metadata capture. Attribute type information is intentionally not preserved.
std::string readAttrAsString(mlir::Attribute attr) {
  if (!attr) return "";
  if (auto s = dyn_cast<StringAttr>(attr)) return s.str();
  if (auto i = dyn_cast<IntegerAttr>(attr)) return std::to_string(i.getInt());
  if (auto f = dyn_cast<FloatAttr>(attr)) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << f.getValue();
    return os.str();
  }
  if (auto b = dyn_cast<BoolAttr>(attr)) return b.getValue() ? "true" : "false";
  if (auto array = dyn_cast<ArrayAttr>(attr)) {
    SmallVector<std::string> values;
    for (mlir::Attribute item : array)
      values.push_back(readAttrAsString(item));
    std::string s = "[";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i) s += ",";
      s += values[i];
    }
    s += "]";
    return s;
  }
  std::string printed;
  llvm::raw_string_ostream os(printed);
  attr.print(os);
  return os.str();
}

// The runtime classifies external inputs as constant/fixed/dynamic. Configured
// fusion has no dedicated input-role field, so infer conservatively from the
// defining op and static shape.
static std::string inferInputRole(Value value) {
  if (auto op = value.getDefiningOp()) {
    if (isa<ConstantOp>(op)) return "constant";
  }
  if (auto tensorType = dyn_cast<atir::TensorType>(value.getType())) {
    for (int64_t dim : tensorType.getShape()) {
      if (ShapedType::isDynamic(dim)) return "dynamic";
    }
  }
  return "fixed";
}

// FusionInfo is attached to the outlined kernel func as a DictionaryAttr.
// These helpers encode the C++ metadata contract as MLIR attributes.
static DictionaryAttr makeFusionArgAttr(
    MLIRContext *ctx, const annc::fusion::FusionArg &arg) {
  Builder builder(ctx);
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("role", builder.getStringAttr(arg.role)));
  attrs.push_back(
      builder.getNamedAttr("tf_name", builder.getStringAttr(arg.tfName)));
  SmallVector<mlir::Attribute> shapeAttrs;
  for (int64_t dim : arg.shape) {
    shapeAttrs.push_back(builder.getI64IntegerAttr(dim));
  }
  attrs.push_back(
      builder.getNamedAttr("shape", ArrayAttr::get(ctx, shapeAttrs)));
  attrs.push_back(
      builder.getNamedAttr("rank", builder.getI64IntegerAttr(arg.rank)));
  attrs.push_back(
      builder.getNamedAttr("dtype", builder.getStringAttr(arg.dtype)));
  return DictionaryAttr::get(ctx, attrs);
}

static ArrayAttr makeFusionArgArray(
    MLIRContext *ctx, ArrayRef<annc::fusion::FusionArg> specs) {
  SmallVector<mlir::Attribute> attrs;
  for (const auto &spec : specs) {
    attrs.push_back(makeFusionArgAttr(ctx, spec));
  }
  return ArrayAttr::get(ctx, attrs);
}

static DictionaryAttr buildFusionMetadata(MLIRContext *ctx,
                                          const annc::fusion::FusionInfo &info,
                                          const std::map<std::string, std::string> &patternAttrs) {
  Builder builder(ctx);
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr("tf.name", builder.getStringAttr(info.name)));
  attrs.push_back(builder.getNamedAttr("fusion.pattern",
                                       builder.getStringAttr(info.pattern)));
  attrs.push_back(builder.getNamedAttr("kernel_name",
                                       builder.getStringAttr(info.kernelName)));
  attrs.push_back(builder.getNamedAttr("args", makeFusionArgArray(ctx, info.args)));
  attrs.push_back(
      builder.getNamedAttr("outputs", makeFusionArgArray(ctx, info.outputs)));
  attrs.push_back(
      builder.getNamedAttr("abi", builder.getStringAttr(info.abi)));
  SmallVector<NamedAttribute> patternAttrEntries;
  for (const auto &kv : patternAttrs) {
    patternAttrEntries.push_back(
        builder.getNamedAttr(kv.first, builder.getStringAttr(kv.second)));
  }
  attrs.push_back(builder.getNamedAttr(
      "pattern_attrs", DictionaryAttr::get(ctx, patternAttrEntries)));
  return DictionaryAttr::get(ctx, attrs);
}

static std::string stableHash(StringRef pattern, StringRef kernel,
                              ArrayRef<std::string> inputNames,
                              ArrayRef<std::string> outputNames,
                              ArrayRef<Operation *> ops) {
  SHA256 sha;
  sha.update(pattern);
  sha.update(kernel);
  for (const std::string &name : inputNames) sha.update(name);
  for (const std::string &name : outputNames) sha.update(name);
  for (Operation *op : ops) sha.update(simpleOpName(op));
  auto hash = sha.final();
  SmallString<16> hex;
  for (size_t i = 0; i < 4; ++i) hex += llvm::toHex(hash[i]);
  return hex.str().str();
}

// An external input alias must bind to the same SSA value throughout a match.
// A second binding to a different value indicates an inconsistent data flow.
static bool bindExternalValue(MatchState &state, StringRef name, Value value) {
  auto it = state.externalValues.find(name.str());
  if (it != state.externalValues.end()) return it->second == value;
  state.externalValues[name.str()] = value;
  state.externalOrder.push_back(name.str());
  return true;
}

// Most ATIR ops use operand 0 as an explicit output buffer, which the graph DSL
// omits. This table aligns DSL operands with real operation operands.
static bool hasLeadingOutputOperand(StringRef opName) {
  return llvm::StringSwitch<bool>(opName)
      .Case("Add", true)
      .Case("Concat", true)
      .Case("ConcatV2", true)
      .Case("ExpandDims", true)
      .Case("Identity", true)
      .Case("Cast", true)
      .Case("Where", true)
      .Case("StridedSlice", true)
      .Case("Pack", true)
      .Case("GatherNd", true)
      .Case("StringToHashBucketFast", true)
      .Case("Slice", true)
      .Case("Gather", true)
      .Case("Prod", true)
      .Case("Sum", true)
      .Case("ResourceGather", true)
      .Case("SparseSegmentSum", true)
      .Case("SparseSegmentMin", true)
      .Case("Tile", true)
      .Case("SparseSegmentMean", true)
      .Case("Sub", true)
      .Case("Mul", true)
      .Case("Fill", true)
      .Case("RealDiv", true)
      .Case("SparseToDense", true)
      .Case("BatchMatMul", true)
      .Case("Rsqrt", true)
      .Case("UnsortedSegmentMin", true)
      .Case("Minimum", true)
      .Case("TensorScatterUpdate", true)
      .Case("Pad", true)
      .Case("broadcast", true)
      .Case("Reshape", true)
      .Case("Logistic", true)
      .Case("Maximum", true)
      .Case("Abs", true)
      .Case("Dot", true)
      .Case("Transpose", true)
      .Case("ReduceMean", true)
      .Case("Softmax", true)
      .Case("Pow", true)
      .Case("Relu", true)
      .Case("MatMul", true)
      .Default(false);
}

// These ops expose a variadic input range in ODS. Their configured input
// count must be checked against the candidate selected by graph matching,
// rather than against every op with the same name in the block.
static bool isVariadicConfigOp(StringRef opName) {
  return llvm::StringSwitch<bool>(opName)
      .Case("Add", true)
      .Case("Concat", true)
      .Case("ConcatV2", true)
      .Case("Where", true)
      .Case("Pack", true)
      .Default(false);
}

static bool isImplicitOperand(Value value) {
  Operation *defOp = value.getDefiningOp();
  // Pattern JSON only models dataflow operands. ATIR buffer placeholders and
  // literal constants are implementation details and may be skipped.
  if (!defOp) return false;
  return isa<BufferOp>(defOp) || isa<ConstantOp>(defOp) ||
         simpleOpName(defOp) == "buffer" || simpleOpName(defOp) == "constant";
}

struct StructureMatchFailure {
  unsigned depth = 0;
  unsigned stmtIndex = 0;
  std::string message;
  Operation *operation = nullptr;
  SkipReason reason = SkipReason::StructureMismatch;
  bool valid = false;
};

static std::string formatPatternStmt(const PatternStmt &stmt) {
  std::string text;
  raw_string_ostream os(text);
  if (stmt.aliases.size() == 1) {
    os << stmt.aliases.front();
  } else {
    os << "(";
    llvm::interleaveComma(stmt.aliases, os);
    os << ")";
  }
  os << " = " << stmt.opName << "(";
  llvm::interleaveComma(stmt.args, os);
  os << ")";
  return os.str();
}

struct DisconnectedGraphStmt {
  unsigned stmtIndex;
  std::string alias;
  std::optional<unsigned> suggestedConsumer;
};

// Every configured statement must be reachable by following aliases backward
// from the declared outputs. A disconnected branch usually means that its
// terminal alias was omitted from a downstream operator's argument list.
static std::optional<DisconnectedGraphStmt> findDisconnectedGraphStmt(
    const PatternRule &rule, ArrayRef<std::string> outputs) {
  SmallVector<unsigned> worklist;
  for (const std::string &output : outputs) {
    auto it = rule.aliasToStmt.find(output);
    if (it == rule.aliasToStmt.end()) return std::nullopt;
    worklist.push_back(it->second);
  }

  std::vector<bool> reachable(rule.stmts.size(), false);
  while (!worklist.empty()) {
    unsigned stmtIndex = worklist.pop_back_val();
    if (reachable[stmtIndex]) continue;
    reachable[stmtIndex] = true;
    for (const std::string &arg : rule.stmts[stmtIndex].args) {
      auto it = rule.aliasToStmt.find(arg);
      if (it != rule.aliasToStmt.end()) worklist.push_back(it->second);
    }
  }

  std::set<std::string> consumedAliases;
  for (const PatternStmt &stmt : rule.stmts) {
    for (const std::string &arg : stmt.args) {
      if (rule.aliasToStmt.count(arg)) consumedAliases.insert(arg);
    }
  }

  for (unsigned i = 0; i < rule.stmts.size(); ++i) {
    if (reachable[i]) continue;
    for (const std::string &alias : rule.stmts[i].aliases) {
      if (alias == "_" || consumedAliases.count(alias)) continue;
      std::optional<unsigned> suggestedConsumer;
      for (unsigned j = i + 1; j < rule.stmts.size(); ++j) {
        if (reachable[j]) {
          suggestedConsumer = j;
          break;
        }
      }
      return DisconnectedGraphStmt{i, alias, suggestedConsumer};
    }
  }

  for (unsigned i = 0; i < rule.stmts.size(); ++i) {
    if (!reachable[i] && !rule.stmts[i].aliases.empty()) {
      return DisconnectedGraphStmt{i, rule.stmts[i].aliases.front(),
                                   std::nullopt};
    }
  }
  return std::nullopt;
}

// Fixed-arity ODS operands exposed by the config DSL. Variadic ops are
// checked against the candidate IR after implicit operands are removed.
static std::optional<unsigned> getMaxConfiguredInputs(StringRef opName) {
  int maxInputs = llvm::StringSwitch<int>(opName)
      .Case("MatMul", 3)
      .Case("Reshape", 2)
      .Case("Relu", 1)
      .Case("ExpandDims", 2)
      .Case("Identity", 1)
      .Case("Shape", 1)
      .Case("Cast", 1)
      .Case("StridedSlice", 4)
      .Case("GatherNd", 2)
      .Case("StringToHashBucketFast", 1)
      .Case("SparseReshape", 3)
      .Case("Slice", 3)
      .Case("Gather", 2)
      .Case("Prod", 2)
      .Case("Sum", 2)
      .Case("SparseFillEmptyRows", 4)
      .Case("Unique", 1)
      .Case("SparseSegmentSum", 4)
      .Case("SparseSegmentMin", 4)
      .Case("SparseSegmentMean", 4)
      .Case("Tile", 2)
      .Case("Size", 1)
      .Case("FloorMod", 2)
      .Case("FloorDiv", 2)
      .Case("Range", 3)
      .Case("Sub", 2)
      .Case("Mul", 2)
      .Case("Fill", 2)
      .Case("RealDiv", 2)
      .Case("SparseToDense", 4)
      .Case("BatchMatMul", 2)
      .Case("Rsqrt", 1)
      .Case("TopK", 2)
      .Case("UnsortedSegmentMin", 3)
      .Case("Minimum", 2)
      .Case("TensorScatterUpdate", 3)
      .Case("Pad", 2)
      .Case("Compare", 2)
      .Case("Divide", 2)
      .Case("And", 2)
      .Case("Logistic", 1)
      .Case("Maximum", 2)
      .Case("Abs", 1)
      .Case("Dot", 2)
      .Case("Transpose", 1)
      .Case("ReduceMean", 1)
      .Case("Softmax", 1)
      .Case("Pow", 2)
      .Default(-1);
  if (maxInputs < 0) return std::nullopt;
  return static_cast<unsigned>(maxInputs);
}

static std::string describeOperand(Value operand) {
  if (Operation *defOp = operand.getDefiningOp()) {
    return (Twine(simpleOpName(defOp)) + " result").str();
  }
  return "block argument";
}

static void recordStructureFailure(StructureMatchFailure *failure,
                                   unsigned depth, unsigned stmtIndex,
                                   const Twine &message,
                                   Operation *operation = nullptr) {
  if (!failure || (failure->valid && depth <= failure->depth)) return;
  failure->depth = depth;
  failure->stmtIndex = stmtIndex;
  failure->message = message.str();
  failure->operation = operation;
  failure->valid = true;
}

static bool matchAlias(const PatternRule &rule, unsigned stmtIndex,
                       StringRef alias, Value value, MatchState &state,
                       StructureMatchFailure *failure, unsigned depth);

static bool matchConfiguredOperand(const PatternRule &rule,
                                   StringRef arg, Value operand,
                                   MatchState &state,
                                   StructureMatchFailure *failure,
                                   unsigned stmtIndex, unsigned depth) {
  auto aliasIt = rule.aliasToStmt.find(arg.str());
  if (aliasIt != rule.aliasToStmt.end()) {
    return matchAlias(rule, aliasIt->second, arg, operand, state, failure,
                      depth);
  }
  if (bindExternalValue(state, arg, operand)) return true;
  recordStructureFailure(
      failure, depth, stmtIndex,
      Twine("external input '") + arg +
          "' is already bound to a different SSA value");
  return false;
}

// Compute the real operand index corresponding to the first data operand in the
// graph DSL. Ops with a leading output buffer start matching at index 1.
static std::optional<unsigned> getLogicalOperandStart(Operation *op,
                                                      unsigned argCount) {
  unsigned operandCount = op->getNumOperands();
  if (hasLeadingOutputOperand(simpleOpName(op))) {
    if (operandCount < argCount + 1) return std::nullopt;
    return 1;
  }
  return operandCount < argCount ? std::nullopt : std::optional<unsigned>(0);
}

static unsigned countNonImplicitDataOperands(Operation *op) {
  unsigned start = hasLeadingOutputOperand(simpleOpName(op)) ? 1 : 0;
  if (op->getNumOperands() < start) return 0;
  unsigned count = 0;
  for (unsigned i = start; i < op->getNumOperands(); ++i) {
    if (!isImplicitOperand(op->getOperand(i))) ++count;
  }
  return count;
}

// Match one graph statement to a candidate operation and recursively bind its
// operand dependencies. BufferOp/ConstantOp details omitted by the DSL are
// skipped here.
static bool matchStmt(const PatternRule &rule, unsigned stmtIndex,
                      Operation *op, MatchState &state,
                      StructureMatchFailure *failure, unsigned depth) {
  if (!op) {
    debugLog([&](raw_ostream &os) {
      os << "stmt #" << stmtIndex << " has no candidate op";
    });
    recordStructureFailure(failure, depth, stmtIndex,
                           "expected an operation but the value has no "
                           "defining operation");
    return false;
  }
  std::string opName = simpleOpName(op);
  const PatternStmt &stmt = rule.stmts[stmtIndex];
  if (opName != stmt.opName) {
    recordStructureFailure(failure, depth, stmtIndex,
                           Twine("expected op ") + stmt.opName + ", got " +
                               opName,
                           op);
    return false;
  }
  debugLog([&](raw_ostream &os) {
    os << "try stmt #" << stmtIndex << " " << stmt.opName << " with op ";
    op->print(os, OpPrintingFlags().skipRegions());
  });
  if (op->getNumResults() != stmt.aliases.size()) {
    debugLog([&](raw_ostream &os) {
      os << "reject " << stmt.opName << ": result count "
         << op->getNumResults() << " != alias count " << stmt.aliases.size();
    });
    recordStructureFailure(
        failure, depth, stmtIndex,
        Twine("expected ") + Twine(stmt.aliases.size()) + " result(s), got " +
            Twine(op->getNumResults()));
    return false;
  }

  unsigned actualDataOperands = countNonImplicitDataOperands(op);
  if (!isVariadicConfigOp(opName) && actualDataOperands != stmt.args.size()) {
    std::string message =
        (Twine("configuration statement '") + formatPatternStmt(stmt) +
         "' declares " + Twine(stmt.args.size()) +
         " logical input(s), but IR op " + opName + " has " +
         Twine(actualDataOperands) + " non-implicit input(s); expected " +
         Twine(actualDataOperands))
            .str();
    if (actualDataOperands < stmt.args.size()) {
      message += "; possible extra config argument '";
      message += stmt.args[actualDataOperands];
      message += "'";
    } else {
      message += "; possible missing config argument for logical input #";
      message += std::to_string(stmt.args.size());
    }
    recordStructureFailure(failure, depth, stmtIndex, Twine(message), op);
    if (failure) failure->reason = SkipReason::ConfigurationError;
    return false;
  }

  auto logicalOperandStart =
      getLogicalOperandStart(op, static_cast<unsigned>(stmt.args.size()));
  if (!logicalOperandStart) {
    std::string message =
        (Twine("configured statement '") + formatPatternStmt(stmt) +
         "' declares " + Twine(stmt.args.size()) +
         " data operand(s), but candidate " + opName + " has too few " +
         "physical operand(s)")
            .str();
    debugLog([&](raw_ostream &os) {
      os << "reject " << stmt.opName << ": operand count "
         << op->getNumOperands() << " is too small for " << stmt.args.size()
         << " configured args";
    });
    recordStructureFailure(failure, depth, stmtIndex, Twine(message), op);
    if (failure) failure->reason = SkipReason::ConfigurationError;
    return false;
  }

  auto it = state.stmtOps.find(stmtIndex);
  if (it != state.stmtOps.end() && it->second != op) {
    debugLog([&](raw_ostream &os) {
      os << "reject " << stmt.opName << ": stmt already bound to a different op";
    });
    recordStructureFailure(failure, depth, stmtIndex,
                           "statement is already bound to a different op");
    return false;
  }
  state.stmtOps[stmtIndex] = op;
  state.matchedSet.insert(op);
  if (!llvm::is_contained(state.matchedOrder, op)) {
    state.matchedOrder.push_back(op);
  }

  for (unsigned i = 0; i < stmt.aliases.size(); ++i) {
    StringRef alias = stmt.aliases[i];
    if (alias == "_") continue;
    Value result = op->getResult(i);
    auto aliasIt = state.aliasValues.find(alias.str());
    if (aliasIt != state.aliasValues.end() && aliasIt->second != result) {
      debugLog([&](raw_ostream &os) {
        os << "reject " << stmt.opName << ": alias '" << alias
           << "' already bound to a different value";
      });
      recordStructureFailure(
          failure, depth, stmtIndex,
          Twine("result alias '") + alias +
              "' is already bound to a different SSA value");
      return false;
    }
    state.aliasValues[alias.str()] = result;
  }

  unsigned operandIndex = *logicalOperandStart;
  unsigned configuredArgIndex = 0;
  for (const std::string &arg : stmt.args) {
    bool matched = false;
    while (operandIndex < op->getNumOperands()) {
      Value operand = op->getOperand(operandIndex);
      if (isImplicitOperand(operand)) {
        debugLog([&](raw_ostream &os) {
          os << "skip implicit operand #" << operandIndex << " for "
             << stmt.opName << ": ";
          operand.print(os);
        });
        ++operandIndex;
        continue;
      }
      if (matchConfiguredOperand(rule, arg, operand, state, failure, stmtIndex,
                                 depth + 1)) {
        matched = true;
        ++operandIndex;
        break;
      }
      debugLog([&](raw_ostream &os) {
        os << "reject " << stmt.opName << ": configured arg '" << arg
           << "' did not match operand #" << operandIndex << " ";
        operand.print(os);
        os << " from ";
        if (Operation *defOp = operand.getDefiningOp()) {
          os << simpleOpName(defOp);
        } else {
          os << "block argument";
        }
      });
      recordStructureFailure(
          failure, depth, stmtIndex,
          Twine("operand #") + Twine(operandIndex) + " for argument '" +
              arg + "' does not match; got " + describeOperand(operand),
          operand.getDefiningOp() ? operand.getDefiningOp() : op);
      return false;
    }
    if (!matched) {
      debugLog([&](raw_ostream &os) {
        os << "reject " << stmt.opName << ": configured arg '" << arg
           << "' was not found";
      });
      if (isVariadicConfigOp(opName) &&
          configuredArgIndex >= actualDataOperands) {
        std::string message =
            (Twine("configuration statement '") + formatPatternStmt(stmt) +
             "' expects " + Twine(stmt.args.size()) +
             " logical input(s), but candidate IR op " + opName + " has " +
             Twine(actualDataOperands) +
             " non-implicit input(s); no operand remains for configured "
             "argument '" + arg + "'")
                .str();
        recordStructureFailure(failure, depth, stmtIndex, Twine(message), op);
      } else {
        recordStructureFailure(
            failure, depth, stmtIndex,
            Twine("configured argument '") + arg +
                "' did not match any remaining non-implicit input of IR op " +
                opName,
            op);
      }
      return false;
    }
    ++configuredArgIndex;
  }

  while (operandIndex < op->getNumOperands()) {
    if (!isImplicitOperand(op->getOperand(operandIndex))) {
      debugLog([&](raw_ostream &os) {
        os << "reject " << stmt.opName << ": trailing operand #"
           << operandIndex << " is not implicit: ";
        op->getOperand(operandIndex).print(os);
      });
      recordStructureFailure(
          failure, depth, stmtIndex,
          Twine("unexpected trailing operand #") + Twine(operandIndex) +
              "; got " + describeOperand(op->getOperand(operandIndex)));
      return false;
    }
    debugLog([&](raw_ostream &os) {
      os << "skip trailing implicit operand #" << operandIndex << " for "
         << stmt.opName << ": ";
      op->getOperand(operandIndex).print(os);
    });
    ++operandIndex;
  }

  debugLog([&](raw_ostream &os) {
    os << "matched stmt #" << stmtIndex << " " << stmt.opName;
  });
  return true;
}

// Validate an already-bound alias. If an unbound alias is defined by an internal
// statement, continue matching through its defining op; otherwise treat it as
// an external boundary input.
static bool matchAlias(const PatternRule &rule, unsigned stmtIndex,
                       StringRef alias, Value value, MatchState &state,
                       StructureMatchFailure *failure, unsigned depth) {
  if (alias == "_") return true;
  auto aliasIt = state.aliasValues.find(alias.str());
  if (aliasIt != state.aliasValues.end()) {
    bool same = aliasIt->second == value;
    if (!same) {
      debugLog([&](raw_ostream &os) {
        os << "alias '" << alias << "' mismatch";
      });
      recordStructureFailure(
          failure, depth, stmtIndex,
          Twine("alias '") + alias +
              "' is bound to a different SSA value than the operand");
    }
    return same;
  }
  if (rule.aliasToStmt.find(alias.str()) == rule.aliasToStmt.end()) {
    debugLog([&](raw_ostream &os) {
      os << "bind external '" << alias << "' to ";
      value.print(os);
    });
    return bindExternalValue(state, alias, value);
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    debugLog([&](raw_ostream &os) {
      os << "alias '" << alias << "' expects stmt #" << stmtIndex
         << " but value has no defining op";
    });
    recordStructureFailure(
        failure, depth, stmtIndex,
        Twine("alias '") + alias + "' expects op " +
            rule.stmts[stmtIndex].opName + ", but got a block argument");
    return false;
  }
  return matchStmt(rule, stmtIndex, defOp, state, failure, depth);
}

// Match the complete pattern subgraph backwards from declared/inferred outputs.
// This avoids enumerating all ops in the forward direction and reduces search.
static bool matchOutputAliases(
    const PatternRule &rule,
    const llvm::StringMap<SmallVector<Operation *, 4>> &opsByName,
    ArrayRef<std::string> outputs, unsigned outputIndex, MatchState state,
    MatchState *outState, PatternReport *report, int64_t warnLimit,
    StructureMatchFailure *failure = nullptr) {
  if (outputIndex >= outputs.size()) {
    if (outState) *outState = std::move(state);
    return true;
  }
  std::string alias = outputs[outputIndex];
  auto stmtIt = rule.aliasToStmt.find(alias);
  if (stmtIt == rule.aliasToStmt.end()) {
    debugLog([&](raw_ostream &os) {
      os << "output alias '" << alias << "' is not defined by graph";
    });
    return false;
  }
  if (state.aliasValues.find(alias) != state.aliasValues.end()) {
    return matchOutputAliases(rule, opsByName, outputs, outputIndex + 1,
                              std::move(state), outState, report, warnLimit,
                              failure);
  }

  const PatternStmt &stmt = rule.stmts[stmtIt->second];
  auto candidatesIt = opsByName.find(stmt.opName);
  if (candidatesIt == opsByName.end()) {
    debugLog([&](raw_ostream &os) {
      os << "no candidates for output alias '" << alias << "' op "
         << stmt.opName;
    });
    recordStructureFailure(
        failure, outputIndex, stmtIt->second,
        Twine("no candidate op named ") + stmt.opName +
            " exists for output alias '" + alias + "'");
    return false;
  }

  for (Operation *op : candidatesIt->second) {
    bool isRootAnchor = outputIndex == 0;
    if (isRootAnchor && report) report->recordAnchor(op);
    if (op->getNumResults() != stmt.aliases.size()) {
      debugLog([&](raw_ostream &os) {
        os << "skip op for output alias '" << alias << "': result count "
           << op->getNumResults() << " != alias count "
           << stmt.aliases.size();
      });
      if (isRootAnchor && report) {
        report->recordSkip(
            SkipReason::StructureMismatch, op,
            (Twine("expected ") + Twine(stmt.aliases.size()) +
             " result(s), got " + Twine(op->getNumResults()))
                .str(),
            0, formatPatternStmt(stmt));
      }
      continue;
    }
    debugLog([&](raw_ostream &os) {
      os << "try output alias '" << alias << "' with candidate ";
      op->print(os, OpPrintingFlags().skipRegions());
    });
    MatchState trial = state;
    StructureMatchFailure candidateFailure;
    StructureMatchFailure *activeFailure =
        isRootAnchor ? &candidateFailure : failure;
    if (matchStmt(rule, stmtIt->second, op, trial, activeFailure,
                  outputIndex) &&
        matchOutputAliases(rule, opsByName, outputs, outputIndex + 1,
                           std::move(trial), outState, report, warnLimit,
                           activeFailure)) {
      debugLog([&](raw_ostream &os) {
        os << "accepted output alias '" << alias << "'";
      });
      return true;
    }
    if (isRootAnchor && report) {
      std::string message = candidateFailure.valid
                                ? candidateFailure.message
                                : "candidate does not match the configured "
                                  "operator graph";
      std::string statement;
      if (candidateFailure.valid &&
          candidateFailure.stmtIndex < rule.stmts.size()) {
        statement = formatPatternStmt(rule.stmts[candidateFailure.stmtIndex]);
      }
      bool failedBelowRoot =
          candidateFailure.valid && candidateFailure.depth > outputIndex;
      Operation *failureOp = candidateFailure.operation
                                 ? candidateFailure.operation
                                 : op;
      report->recordSkip(candidateFailure.reason, failureOp, message,
                         failedBelowRoot ? warnLimit : 0, statement);
    }
    if (isRootAnchor && failure && candidateFailure.valid &&
        (!failure->valid || candidateFailure.depth > failure->depth)) {
      *failure = candidateFailure;
      if (!failure->operation) failure->operation = op;
    }
    debugLog([&](raw_ostream &os) {
      os << "rejected output alias '" << alias << "' candidate";
    });
  }
  debugLog([&](raw_ostream &os) {
    os << "no candidate matched output alias '" << alias << "'";
  });
  return false;
}

// Evaluate lightweight where expressions. It only uses completed alias/external
// bindings and never triggers new IR searches; the language stays small and
// diagnosable by design.
static bool evaluateWhereExpr(const std::string &expr, const PatternRule &rule,
                              const MatchState &state) {
  std::string text = trim(expr);
  if (text.empty()) return true;
  StringRef textRef(text);

  auto lookupValue = [&](StringRef name) -> Value {
    auto aliasIt = state.aliasValues.find(name.str());
    if (aliasIt != state.aliasValues.end()) return aliasIt->second;
    auto extIt = state.externalValues.find(name.str());
    if (extIt != state.externalValues.end()) return extIt->second;
    return Value();
  };

  if (textRef.starts_with("exists(") && textRef.ends_with(")")) {
    std::string inner = text.substr(strlen("exists("), text.size() - strlen("exists(") - 1);
    size_t dot = inner.find('.');
    if (dot == std::string::npos) return false;
    Value value = lookupValue(trim(inner.substr(0, dot)));
    if (!value) return false;
    Operation *op = value.getDefiningOp();
    if (!op) return false;
    return op->hasAttr(trim(inner.substr(dot + 1)));
  }
  if (textRef.starts_with("not_exists(") && textRef.ends_with(")")) {
    std::string inner = text.substr(strlen("not_exists("), text.size() - strlen("not_exists(") - 1);
    size_t dot = inner.find('.');
    if (dot == std::string::npos) return false;
    Value value = lookupValue(trim(inner.substr(0, dot)));
    if (!value) return true;
    Operation *op = value.getDefiningOp();
    if (!op) return true;
    return !op->hasAttr(trim(inner.substr(dot + 1)));
  }
  if (textRef.starts_with("rank(")) {
    size_t close = text.find(')');
    size_t eq = text.find("==", close == std::string::npos ? 0 : close);
    if (close == std::string::npos || eq == std::string::npos) return false;
    std::string name = trim(text.substr(strlen("rank("), close - strlen("rank(")));
    int64_t expected = 0;
    if (!parseInteger(StringRef(text).drop_front(eq + 2), &expected)) return false;
    Value value = lookupValue(name);
    if (!value) return false;
    return getRank(value.getType()) == expected;
  }
  if (textRef.starts_with("one_of(") && textRef.ends_with(")")) {
    std::string inner =
        text.substr(strlen("one_of("), text.size() - strlen("one_of(") - 1);
    std::vector<std::string> parts = splitTopLevel(inner, ',');
    if (parts.size() < 2) return false;
    std::string lhs = trim(parts.front());
    size_t dot = lhs.find('.');
    if (dot == std::string::npos) return false;
    Value value = lookupValue(trim(lhs.substr(0, dot)));
    if (!value) return false;
    Operation *op = value.getDefiningOp();
    if (!op) return false;
    std::string actual = readAttrAsString(op->getAttr(trim(lhs.substr(dot + 1))));

    std::vector<std::string> candidates;
    if (parts.size() == 2 && !parts[1].empty() && parts[1].front() == '[') {
      std::string listText = trim(parts[1]);
      size_t close = listText.rfind(']');
      if (close == std::string::npos) return false;
      for (const std::string &part :
           splitTopLevel(StringRef(listText).slice(1, close), ',')) {
        std::string parsed = parseQuotedString(part);
        candidates.push_back(parsed.empty() ? trim(part) : parsed);
      }
    } else {
      for (size_t i = 1; i < parts.size(); ++i) {
        std::string parsed = parseQuotedString(parts[i]);
        candidates.push_back(parsed.empty() ? trim(parts[i]) : parsed);
      }
    }
    return llvm::is_contained(candidates, actual);
  }

  size_t eq = text.find("==");
  if (eq != std::string::npos) {
    std::string lhs = trim(text.substr(0, eq));
    std::string rhs = trim(text.substr(eq + 2));
    size_t dot = lhs.find('.');
    if (dot == std::string::npos) return false;
    Value value = lookupValue(trim(lhs.substr(0, dot)));
    if (!value) return false;
    Operation *op = value.getDefiningOp();
    if (!op) return false;
    std::string attrName = trim(lhs.substr(dot + 1));
    mlir::Attribute attr = op->getAttr(attrName);
    std::string expected = parseQuotedString(rhs);
    if (expected.empty() && rhs.find('"') == std::string::npos) {
      expected = trim(rhs);
    }
    return readAttrAsString(attr) == expected;
  }
  return false;
}

static bool checkWhereClauses(const PatternRule &rule, const MatchState &state,
                              std::string *failedExpression) {
  for (const std::string &expr : rule.where) {
    if (!evaluateWhereExpr(expr, rule, state)) {
      if (failedExpression) *failedExpression = expr;
      debugLog([&](raw_ostream &os) {
        os << "where clause failed: " << expr;
      });
      return false;
    }
  }
  return true;
}

// Merge static attrs and dynamic captures into pattern_attrs for downstream
// GraphDef rewriting and runtime pattern-specific behavior.
bool collectPatternAttrs(const PatternRule &rule, const MatchState &state,
                         std::map<std::string, std::string> *attrs) {
  *attrs = rule.attrs;
  for (const CaptureSpec &capture : rule.captures) {
    auto aliasIt = state.aliasValues.find(capture.alias);
    if (aliasIt == state.aliasValues.end()) return false;
    Operation *op = aliasIt->second.getDefiningOp();
    if (!op) return false;
    attrs->insert_or_assign(capture.key,
                            readAttrAsString(op->getAttr(capture.attr)));
  }
  return true;
}

// If outputs are omitted, use all non-"_" results of the final graph statement.
std::vector<std::string> inferOutputs(const PatternRule &rule) {
  if (!rule.outputs.empty()) return std::vector<std::string>(rule.outputs.begin(),
                                                             rule.outputs.end());
  if (rule.stmts.empty()) return {};
  const PatternStmt &stmt = rule.stmts.back();
  std::vector<std::string> outputs;
  for (const std::string &alias : stmt.aliases) {
    if (alias != "_") outputs.push_back(alias);
  }
  return outputs;
}

// If inputs are omitted, use the first-seen external-input order from matching.
std::vector<std::string> inferInputs(const PatternRule &rule,
                                     const MatchState &state) {
  if (!rule.inputs.empty()) {
    return std::vector<std::string>(rule.inputs.begin(), rule.inputs.end());
  }
  return std::vector<std::string>(state.externalOrder.begin(),
                                  state.externalOrder.end());
}

// Resolve configured input/output aliases to real SSA values. Failure indicates
// an inconsistency between the pattern declaration and matched subgraph.
static std::vector<Value> buildInputValues(const std::vector<std::string> &inputs,
                                           const MatchState &state) {
  std::vector<Value> values;
  values.reserve(inputs.size());
  for (const std::string &name : inputs) {
    auto it = state.externalValues.find(name);
    if (it == state.externalValues.end()) return {};
    values.push_back(it->second);
  }
  return values;
}

static std::vector<Value> buildOutputValues(const std::vector<std::string> &outputs,
                                            const MatchState &state) {
  std::vector<Value> values;
  values.reserve(outputs.size());
  for (const std::string &name : outputs) {
    auto it = state.aliasValues.find(name);
    if (it == state.aliasValues.end()) return {};
    values.push_back(it->second);
  }
  return values;
}

// Prevent internal results omitted from outputs from escaping the subgraph;
// erasing the original ops would otherwise leave invalid or incorrect uses.
bool hasUnlistedEscapingOutput(const PatternRule &rule,
                               const std::vector<std::string> &outputs,
                               const MatchState &state) {
  SmallPtrSet<Operation *, 32> matchedOps(state.matchedSet.begin(),
                                          state.matchedSet.end());
  std::set<std::string> outputSet(outputs.begin(), outputs.end());
  for (const auto &entry : state.aliasValues) {
    if (rule.aliasToStmt.find(entry.first) == rule.aliasToStmt.end()) continue;
    if (outputSet.count(entry.first) != 0) continue;
    for (Operation *user : entry.second.getUsers()) {
      if (matchedOps.contains(user)) continue;
      if (isa<BufferOp>(user) || isa<ConstantOp>(user)) continue;
      if (isa<func::ReturnOp>(user) || isa<CastOp>(user)) {
        debugLog([&](raw_ostream &os) {
          os << "ignore implicit escape for alias '" << entry.first
             << "' via user ";
          user->print(os, OpPrintingFlags().skipRegions());
        });
        continue;
      }
      debugLog([&](raw_ostream &os) {
        os << "unlisted alias '" << entry.first << "' escapes through user ";
        user->print(os, OpPrintingFlags().skipRegions());
      });
      return true;
    }
  }
  return false;
}

static void appendUniqueValue(std::vector<Value> &values, Value value) {
  if (!value || llvm::is_contained(values, value)) return;
  values.push_back(value);
}

// The outlined kernel parameters combine declared inputs with all real boundary
// operands, including implementation-level output buffers and block arguments.
static std::vector<Value> collectBoundaryValues(
    ArrayRef<Operation *> orderedOps, const MatchState &state,
    ArrayRef<Value> userInputValues) {
  std::vector<Value> values;
  for (Value value : userInputValues) appendUniqueValue(values, value);

  for (Operation *op : orderedOps) {
    for (Value operand : op->getOperands()) {
      if (isImplicitOperand(operand)) continue;
      Operation *def = operand.getDefiningOp();
      if (!def || state.matchedSet.count(def) == 0) {
        appendUniqueValue(values, operand);
      }
    }
  }
  return values;
}

// Replace only uses outside the matched subgraph; internal uses disappear with
// the erased operations.
static void replaceExternalUses(Value oldValue, Value newValue,
                                const MatchState &state) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : oldValue.getUses()) {
    if (state.matchedSet.count(use.getOwner()) == 0) uses.push_back(&use);
  }
  for (OpOperand *use : uses) use->set(newValue);
}

static bool allResultsUnused(Operation *op) {
  for (Value result : op->getResults()) {
    if (!result.use_empty()) return false;
  }
  return true;
}

// Outline the matched subgraph into a private kernel func and insert a call in
// main to replace the original outputs. Clone the body with IRMapping and copy
// metadata to the generated function.
func::FuncOp materializePattern(
    ModuleOp module, PatternRule rule, const MatchState &state,
    const std::vector<std::string> &inputs,
    const std::vector<std::string> &outputs,
    const std::map<std::string, std::string> &patternAttrs) {
  std::vector<Value> inputValues = buildInputValues(inputs, state);
  std::vector<Value> outputValues = buildOutputValues(outputs, state);
  if (inputValues.size() != inputs.size() || outputValues.size() != outputs.size()) {
    debugLog([&](raw_ostream &os) {
      os << "materialization binding mismatch for rule '" << rule.name << "'";
      os << ", inputs=" << inputs.size() << " resolved=" << inputValues.size();
      os << ", outputs=" << outputs.size() << " resolved=" << outputValues.size();
    });
    return nullptr;
  }

  std::vector<Operation *> orderedOps;
  if (state.matchedOrder.empty()) return nullptr;
  Block *parentBlock = state.matchedOrder.front()->getBlock();
  for (Operation *op : state.matchedOrder) {
    if (op->getBlock() != parentBlock) return nullptr;
  }
  for (Operation &op : *parentBlock) {
    if (state.matchedSet.count(&op) != 0) orderedOps.push_back(&op);
  }
  if (orderedOps.empty()) return nullptr;

  std::vector<Value> boundaryValues =
      collectBoundaryValues(orderedOps, state, inputValues);
  debugLog([&](raw_ostream &os) {
    os << "materialize rule '" << rule.name << "'";
    os << ", boundary values=" << boundaryValues.size();
    os << ", input values=" << inputValues.size();
    os << ", output values=" << outputValues.size();
  });

  std::string hash = stableHash(rule.pattern, rule.kernel, inputs, outputs,
                                orderedOps);
  std::string kernelName = uniquifySymbolName(module, sanitizeName(rule.kernel) +
                                                           "_" + hash);

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());

  SmallVector<mlir::Type> inputTypes;
  if (rule.abi == "annc_execution_v2") {
    inputTypes.push_back(LLVM::LLVMPointerType::get(module.getContext()));
  }
  for (Value value : boundaryValues) inputTypes.push_back(value.getType());
  SmallVector<mlir::Type> resultTypes;
  for (Value value : outputValues) resultTypes.push_back(value.getType());

  auto funcType = builder.getFunctionType(inputTypes, resultTypes);
  auto func = builder.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("fusion.pattern",
                builder.getStringAttr(rule.pattern.empty() ? rule.name
                                                           : rule.pattern));
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(builder.getContext()));
  func->setAttr("annc.kernel", UnitAttr::get(builder.getContext()));

  MatchState clonedState = state;
  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  IRMapping mapper;
  unsigned argOffset = rule.abi == "annc_execution_v2" ? 1 : 0;
  for (auto [boundaryValue, arg] : llvm::zip(
           boundaryValues, entry->getArguments().drop_front(argOffset))) {
    mapper.map(boundaryValue, arg);
  }

  for (Operation *op : orderedOps) {
    // Implicit operands such as constants/buffers are not DSL inputs, but cloned
    // ops still need their definitions; populate the mapper before cloning.
    for (Value operand : op->getOperands()) {
      if (!isImplicitOperand(operand) || mapper.contains(operand)) continue;
      Operation *def = operand.getDefiningOp();
      if (!def) continue;
      Operation *clonedImplicit = builder.clone(*def, mapper);
      for (auto [orig, cloneResult] :
           llvm::zip(def->getResults(), clonedImplicit->getResults())) {
        mapper.map(orig, cloneResult);
      }
    }
    Operation *cloned = builder.clone(*op, mapper);
    for (auto [orig, cloneResult] : llvm::zip(op->getResults(), cloned->getResults())) {
      for (auto it = clonedState.aliasValues.begin(); it != clonedState.aliasValues.end(); ++it) {
        if (it->second == orig) it->second = cloneResult;
      }
      mapper.map(orig, cloneResult);
    }
  }

  SmallVector<Value> returns;
  for (const std::string &name : outputs) {
    auto it = clonedState.aliasValues.find(name);
    if (it == clonedState.aliasValues.end()) return nullptr;
    returns.push_back(it->second);
  }
  builder.setInsertionPointToEnd(entry);
  builder.create<func::ReturnOp>(func.getLoc(), returns);

  annc::fusion::FusionInfo info;
  info.name = rule.abi == "annc_execution_v2"
                  ? (rule.pattern.empty() ? rule.name : rule.pattern)
                  : kernelName;
  info.pattern = rule.pattern.empty() ? rule.name : rule.pattern;
  info.kernelName = kernelName;
  info.abi = rule.abi;
  for (const std::string &name : inputs) {
    auto it = state.externalValues.find(name);
    if (it == state.externalValues.end()) return nullptr;
    Value value = it->second;
    annc::fusion::FusionArg arg;
    arg.role = inferInputRole(value);
    arg.tfName = getValueName(value);
    arg.shape = getShape(value.getType());
    arg.rank = getRank(value.getType());
    arg.dtype = getTensorDType(value.getType());
    info.args.push_back(std::move(arg));
  }
  for (const std::string &name : outputs) {
    auto it = state.aliasValues.find(name);
    if (it == state.aliasValues.end()) return nullptr;
    Value value = it->second;
    annc::fusion::FusionArg out;
    out.role = "output";
    out.tfName = getFusionOutputName(value);
    out.shape = getShape(value.getType());
    out.rank = getRank(value.getType());
    out.dtype = getTensorDType(value.getType());
    info.outputs.push_back(std::move(out));
  }
  info.patternAttrs = patternAttrs;

  // Metadata is the sole contract used by fusion-metadata/converter/runtime to
  // identify the kernel, so write it completely before replacing the IR.
  func->setAttr("fusion.metadata", buildFusionMetadata(module.getContext(),
                                                        info, patternAttrs));

  // Execution-v2 fused functions are consumed by the existing FastCodegen
  // custom-pattern path.  Keep the original main graph untouched; prune-func
  // will retain this private fusion function and FastCodegen will rewrite its
  // body to the appropriate CustomizeOp.  Generic outlines keep the legacy
  // main-level call replacement behavior.
  if (rule.abi == "mlir_ciface") {
    builder.setInsertionPoint(orderedOps.front());
    auto call = builder.create<func::CallOp>(
        orderedOps.front()->getLoc(), kernelName, TypeRange(resultTypes),
        ValueRange(boundaryValues));
    for (auto [oldValue, newValue] :
         llvm::zip(outputValues, call.getResults())) {
      replaceExternalUses(oldValue, newValue, state);
    }

    for (Operation *op : llvm::reverse(orderedOps)) {
      if (allResultsUnused(op)) {
        op->erase();
      }
    }
  } else {
    // The execution-v2 path intentionally leaves the main graph in place for
    // the later FastCodegen pass.  Mark the captured region so the config
    // rewrite loop and optional builtin fusion do not consume it again.
    for (Operation *op : orderedOps) {
      op->setAttr("annc.fusion_materialized",
                  UnitAttr::get(module.getContext()));
    }
  }

  return func;
}

// Try matching and materializing one rule once. On success, the caller rescans
// IR because operation and use relationships in main have changed.
bool tryMatchRule(ModuleOp module, func::FuncOp mainFunc,
                  const PatternRule &rule, PatternReport *report,
                  int64_t warnLimit) {
  debugLog([&](raw_ostream &os) {
    os << "try rule '" << rule.name << "' with " << rule.stmts.size()
       << " graph stmts";
  });
  Block &block = mainFunc.front();
  llvm::StringMap<SmallVector<Operation *, 4>> opsByName;
  for (Operation &op : block.getOperations()) {
    if (isa<func::ReturnOp>(op)) continue;
    if (op.hasAttr("annc.fusion_materialized")) continue;
    opsByName[simpleOpName(&op)].push_back(&op);
  }

  std::vector<std::string> outputs = inferOutputs(rule);
  if (outputs.empty()) {
    debugLog([&](raw_ostream &os) {
      os << "rule '" << rule.name << "' has no outputs";
    });
    return false;
  }
  debugLog([&](raw_ostream &os) {
    os << "rule '" << rule.name << "' inferred outputs:";
    for (const std::string &output : outputs) os << " " << output;
  });

  for (unsigned stmtIndex = 0; stmtIndex < rule.stmts.size(); ++stmtIndex) {
    const PatternStmt &stmt = rule.stmts[stmtIndex];
    auto maxInputs = getMaxConfiguredInputs(stmt.opName);
    if (!maxInputs || stmt.args.size() <= *maxInputs) continue;
    std::string message =
        (Twine("configuration statement '") + formatPatternStmt(stmt) +
         "' declares " + Twine(stmt.args.size()) +
         " logical input(s), but op " + stmt.opName + " defines at most " +
         Twine(*maxInputs) + "; possible extra config argument '")
            .str();
    message += stmt.args[*maxInputs];
    message += "'";
    if (report) {
      report->recordSkip(SkipReason::ConfigurationError, nullptr, message,
                         warnLimit, formatPatternStmt(stmt));
    }
    debugLog([&](raw_ostream &os) { os << message; });
    return false;
  }

  if (auto disconnected = findDisconnectedGraphStmt(rule, outputs)) {
    std::string message =
        (Twine("configured alias '") + disconnected->alias +
         "' is disconnected from the outputs and is not consumed by any "
         "graph statement")
            .str();
    if (disconnected->suggestedConsumer) {
      message += "; possible missing argument in '";
      message += formatPatternStmt(
          rule.stmts[*disconnected->suggestedConsumer]);
      message += "'";
    }
    debugLog([&](raw_ostream &os) { os << message; });

    if (report) {
      report->recordSkip(
          SkipReason::ConfigurationError, nullptr, message, warnLimit,
          formatPatternStmt(rule.stmts[disconnected->stmtIndex]));
    }
    return false;
  }

  if (report) {
    auto outputStmt = rule.aliasToStmt.find(outputs.front());
    if (outputStmt != rule.aliasToStmt.end()) {
      auto candidates = opsByName.find(rule.stmts[outputStmt->second].opName);
      if (candidates != opsByName.end()) {
        for (Operation *candidate : candidates->second) {
          report->recordAnchor(candidate);
        }
      }
    }
  }

  MatchState state;
  MatchState matchedState;
  StructureMatchFailure matchingFailure;
  if (!matchOutputAliases(rule, opsByName, outputs, 0, state, &matchedState,
                          nullptr, 0, &matchingFailure)) {
    if (report && matchingFailure.valid &&
        matchingFailure.stmtIndex < rule.stmts.size()) {
      const PatternStmt &stmt = rule.stmts[matchingFailure.stmtIndex];
      std::string message =
          (Twine("configured statement '") + formatPatternStmt(stmt) +
           "' failed: " + matchingFailure.message)
              .str();
      report->recordSkip(matchingFailure.reason, matchingFailure.operation,
                         message, warnLimit, formatPatternStmt(stmt));
    }
    debugLog([&](raw_ostream &os) {
      os << "rule '" << rule.name << "' failed during output alias matching";
    });
    return false;
  }
  state = std::move(matchedState);
  Operation *anchor = nullptr;
  auto anchorIt = state.aliasValues.find(outputs.front());
  if (anchorIt != state.aliasValues.end()) {
    anchor = anchorIt->second.getDefiningOp();
  }
  debugLog([&](raw_ostream &os) {
    os << "rule '" << rule.name << "' matched aliases:";
    for (const auto &entry : state.aliasValues) {
      os << " " << entry.first << "=";
      entry.second.print(os);
    }
    os << "; external values:";
    for (const auto &entry : state.externalValues) {
      os << " " << entry.first << "=";
      entry.second.print(os);
    }
  });
  if (state.stmtOps.size() != rule.stmts.size()) {
    debugLog([&](raw_ostream &os) {
      os << "rule '" << rule.name << "' matched " << state.stmtOps.size()
         << " stmts, expected " << rule.stmts.size();
    });
    if (report) {
      report->recordSkip(SkipReason::StructureMismatch, anchor,
                         "not all configured graph statements were matched",
                         warnLimit);
    }
    return false;
  }

  std::string failedExpression;
  if (!checkWhereClauses(rule, state, &failedExpression)) {
    debugLog([&](raw_ostream &os) {
      os << "rule '" << rule.name << "' failed where clauses";
    });
    if (report) {
      report->recordSkip(
          SkipReason::ConstraintMismatch, anchor,
          (Twine("constraint is not satisfied: ") + failedExpression).str(),
          warnLimit);
    }
    return false;
  }

  std::map<std::string, std::string> patternAttrs;
  if (!collectPatternAttrs(rule, state, &patternAttrs)) {
    debugLog([&](raw_ostream &os) {
      os << "rule '" << rule.name << "' failed attr capture";
    });
    if (report) {
      report->recordSkip(SkipReason::CaptureFailure, anchor,
                         "configured attributes could not be captured",
                         warnLimit);
    }
    return false;
  }

  std::vector<std::string> inputs = inferInputs(rule, state);
  debugLog([&](raw_ostream &os) {
    os << "rule '" << rule.name << "' inferred inputs:";
    for (const std::string &input : inputs) os << " " << input;
  });
  if (!rule.inputs.empty()) {
    std::unordered_set<std::string> expected(rule.inputs.begin(), rule.inputs.end());
    for (const std::string &name : inputs) expected.erase(name);
    if (!expected.empty()) {
      debugLog([&](raw_ostream &os) {
        os << "rule '" << rule.name << "' missing configured inputs:";
        for (const std::string &name : expected) os << " " << name;
      });
      if (report) {
        report->recordSkip(SkipReason::InputMismatch, anchor,
                           "one or more configured inputs are not bound",
                           warnLimit);
      }
      return false;
    }
  }

  if (hasUnlistedEscapingOutput(rule, outputs, state)) {
    debugLog([&](raw_ostream &os) {
      os << "rule '" << rule.name << "' has unlisted escaping output";
    });
    if (report) {
      report->recordSkip(
          SkipReason::BoundaryEscape, anchor,
          "an internal result escapes the fusion boundary but is not listed "
          "as an output",
          warnLimit);
    }
    return false;
  }
  auto func = materializePattern(module, rule, state, inputs, outputs,
                                 patternAttrs);
  if (!func) {
    debugLog([&](raw_ostream &os) {
      os << "rule '" << rule.name << "' failed materialization";
    });
    if (report) {
      report->recordSkip(SkipReason::MaterializationFailure, anchor,
                         "matched graph could not be outlined", warnLimit);
    }
    return false;
  }
  if (report) {
    ++report->matches;
    report->outlined.push_back(func.getName().str());
    for (const std::string &input : inputs) {
      auto valueIt = state.externalValues.find(input);
      if (valueIt != state.externalValues.end()) {
        report->inferredRoles[input] = inferInputRole(valueIt->second);
      }
    }
  }
  debugLog([&](raw_ostream &os) {
    os << "rule '" << rule.name << "' matched and materialized";
  });
  return func != nullptr;
}


}  // namespace atir::config_fusion
