#include "Conversion/StablehloToAtir/V2OpMatcher.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallString.h"

#include <string>

using namespace mlir;

namespace annc {
namespace {

// Parses the ODS verifier arity diagnostics; a miss only disables the
// sentinel repair (locked by a component test).
static bool parseOperandArity(const std::string& message, int& expected,
                              int& found) {
  const llvm::StringRef kNeed = "expected ";
  size_t pos = message.find(kNeed);
  if (pos == std::string::npos) return false;
  llvm::StringRef rest =
      llvm::StringRef(message).drop_front(pos + kNeed.size());
  size_t digits = 0;
  while (digits < rest.size() && llvm::isDigit(rest[digits])) ++digits;
  if (!llvm::to_integer(rest.take_front(digits), expected)) return false;
  rest = rest.drop_front(digits);
  if (!rest.consume_front(" operands, but found ")) return false;
  digits = 0;
  while (digits < rest.size() && llvm::isDigit(rest[digits])) ++digits;
  return llvm::to_integer(rest.take_front(digits), found);
}

// Fill only the inferable num_segments; filling others (TopK's k) changes semantics.
static bool allowsSentinelFill(RegisteredOperationName target) {
  llvm::StringRef name = target.getIdentifier();
  return name == "atir.sparse_segment_mean_v2" ||
         name == "atir.sparse_segment_sum_v2";
}

// Sentinel element type: the last operand must be a ranked integer tensor.
static Type sentinelElemType(Value operand) {
  auto shaped = dyn_cast<ShapedType>(operand.getType());
  if (!shaped || !shaped.hasRank()) return {};
  return dyn_cast<IntegerType>(shaped.getElementType());
}

std::string camelToSnake(llvm::StringRef name) {
  std::string out;
  for (char c : name) {
    if (llvm::isUpper(c) && !out.empty()) out.push_back('_');
    out.push_back(llvm::toLower(c));
  }
  return out;
}

std::string snakeToCamel(llvm::StringRef name) {
  std::string out;
  bool upper = false;
  for (char c : name) {
    if (c == '_') {
      upper = true;
      continue;
    }
    out.push_back(upper ? llvm::toUpper(c) : c);
    upper = false;
  }
  return out;
}

}  // namespace

std::optional<RegisteredOperationName> resolveV2Target(
    MLIRContext* ctx, llvm::StringRef sourceOpType,
    llvm::ArrayRef<std::pair<llvm::StringRef, llvm::StringRef>> opAliases) {
  // Pure table lookup; absent entries yield nullopt (fallback carrier).
  for (const auto& [from, to] : opAliases) {
    if (from == sourceOpType)
      return RegisteredOperationName::lookup("atir." + to.str(), ctx);
  }
  return std::nullopt;
}

LogicalResult emitV2Op(
    RewriterBase& rewriter, Location loc,
    RegisteredOperationName target, ValueRange operands,
    ArrayRef<Type> resultTypes, DictionaryAttr source,
    llvm::ArrayRef<V2AttrAlias> attrAliases,
    llvm::ArrayRef<NamedAttribute> derived,
    llvm::SmallVectorImpl<Value>& results) {
  MLIRContext* ctx = rewriter.getContext();

  SmallVector<NamedAttribute> declared;
  llvm::DenseSet<llvm::StringRef> consumed;
  llvm::DenseSet<llvm::StringRef> emitted;
  for (StringAttr name : target.getAttributeNames()) {
    llvm::SmallVector<llvm::StringRef, 4> candidates;
    for (const V2AttrAlias& alias : attrAliases)
      if (alias.declared == name.getValue())
        candidates.append(alias.sources.begin(), alias.sources.end());
    candidates.push_back(name.getValue());
    std::string snake = camelToSnake(name.getValue());
    std::string camel = snakeToCamel(name.getValue());
    if (snake != name.getValue()) candidates.push_back(snake);
    if (camel != name.getValue()) candidates.push_back(camel);

    for (llvm::StringRef candidate : candidates) {
      std::optional<NamedAttribute> found = source.getNamed(candidate);
      if (!found) continue;
      Attribute value = found->getValue();
      // Unify integers on i64; BoolAttr must keep its type (ODS setters re-cast).
      if (auto intValue = dyn_cast<IntegerAttr>(value)) {
        if (!isa<BoolAttr>(value))
          value = IntegerAttr::get(IntegerType::get(ctx, 64),
                                   intValue.getInt());
      }
      declared.emplace_back(name, value);
      emitted.insert(name.getValue());
      consumed.insert(found->getName().getValue());
      break;
    }
  }

  // 2. Contract-recovered attributes lose to assembled declared ones.
  for (NamedAttribute attr : derived)
    if (emitted.insert(attr.getName().getValue()).second)
      declared.push_back(attr);

  // 3. Preservation: unmatched source attributes ride along (residue_reason drops).
  SmallVector<NamedAttribute> preserved;
  for (NamedAttribute attr : source) {
    llvm::StringRef key = attr.getName().getValue();
    if (consumed.contains(key)) continue;
    if (emitted.contains(key)) continue;
    if (key == "residue_reason") continue;
    preserved.push_back(attr);
  }

  OperationState state(loc, target);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttributes(declared);
  state.addAttributes(preserved);
  Operation* created = rewriter.create(state);

  std::string verifyError;
  auto runVerify = [&](Operation* op) {
    verifyError.clear();
    ScopedDiagnosticHandler handler(
        op->getContext(), [&](Diagnostic& diag) {
          if (diag.getSeverity() == DiagnosticSeverity::Error)
            verifyError += diag.str();
          return ::mlir::success();  // consume: keep stderr clean
        });
    return mlir::verify(op, /*verifyRecursively=*/false);
  };

  if (failed(runVerify(created))) {
    int expected = 0, found = 0;
    const bool oneShort =
        parseOperandArity(verifyError, expected, found) &&
        expected == static_cast<int>(operands.size()) + 1 &&
        found == static_cast<int>(operands.size()) && !operands.empty() &&
        allowsSentinelFill(target);
    const bool sentinelFillable =
        oneShort && sentinelElemType(operands.back());
    if (sentinelFillable) {
      Value sentinel = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(RankedTensorType::get(
                   {}, sentinelElemType(operands.back()))));
      rewriter.eraseOp(created);
      OperationState retry(loc, target);
      retry.addOperands(operands);
      retry.addOperands(ValueRange{sentinel});
      retry.addTypes(resultTypes);
      retry.addAttributes(declared);
      retry.addAttributes(preserved);
      created = rewriter.create(retry);
    } else if (operands.size() >= 2 && operands.size() % 2 == 0) {
      int32_t half = static_cast<int32_t>(operands.size() / 2);
      created->setAttr("operandSegmentSizes",
                       rewriter.getDenseI32ArrayAttr({half, half}));
    }
    if (failed(runVerify(created))) {
      rewriter.eraseOp(created);
      return failure();
    }
  }

  results.append(created->getResults().begin(), created->getResults().end());
  return success();
}

}  // namespace annc
