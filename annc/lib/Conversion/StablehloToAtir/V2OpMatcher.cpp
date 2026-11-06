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

// Parses the ODS verifier arity message ("expected N operands, but found M").
static bool parseOperandArity(const std::string& message, int& expected) {
  const llvm::StringRef kNeed = "expected ";
  size_t pos = message.find(kNeed);
  if (pos == std::string::npos) return false;
  pos += kNeed.size();
  size_t end = pos;
  while (end < message.size() && llvm::isDigit(message[end])) ++end;
  if (end == pos) return false;
  expected = 0;
  for (size_t i = pos; i < end; ++i)
    expected = expected * 10 + (message[i] - '0');
  return true;
}

// Sentinel element type: the last operand must be a ranked integer tensor.
static Type sentinelElemType(Value operand) {
  auto shaped = dyn_cast<ShapedType>(operand.getType());
  if (!shaped || !shaped.hasRank()) return {};
  return dyn_cast<IntegerType>(shaped.getElementType());
}

// camelCase -> snake_case ("TopKV2" -> "top_k_v2"), mirroring AtirOpsV2.td.
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
  // Pure table lookup; an absent entry yields nullopt (fallback carrier).
  for (const auto& [from, to] : opAliases) {
    if (from == sourceOpType)
      return RegisteredOperationName::lookup("atir." + to.str(), ctx);
  }
  return std::nullopt;
}

LogicalResult emitV2Op(
    ConversionPatternRewriter& rewriter, Location loc,
    RegisteredOperationName target, ValueRange operands,
    ArrayRef<Type> resultTypes, DictionaryAttr source,
    llvm::ArrayRef<V2AttrAlias> attrAliases,
    llvm::ArrayRef<NamedAttribute> derived,
    llvm::SmallVectorImpl<Value>& results) {
  MLIRContext* ctx = rewriter.getContext();

  // 1. Declared attribute assembly (contract levels 1-3).
  SmallVector<NamedAttribute> declared;
  llvm::DenseSet<llvm::StringRef> consumed;
  // Emitted names: duplicate NamedAttribute names would trip DictionaryAttr.
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
      // V2 unifies integer attrs on I64Attr; BoolAttr must keep its type
      // (ODS setters re-cast to BoolAttr and drop mismatches).
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

  // 2. Contract-recovered attributes lose to assembled declared attributes.
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

  // 4. Local verify. On an arity mismatch, fill one missing trailing integer
  // operand with scalar 0 (the atir sentinel for inferred values, e.g.
  // num_segments <= 0); anything else falls back to the carrier.
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
    int expected = 0;
    const bool oneShort =
        parseOperandArity(verifyError, expected) &&
        expected == static_cast<int>(operands.size()) + 1 && !operands.empty();
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
    }
    // Multi-variadic targets would need a segment repair; none exist yet.
    if (failed(runVerify(created))) {
      rewriter.eraseOp(created);
      return failure();
    }
  }

  results.append(created->getResults().begin(), created->getResults().end());
  return success();
}

}  // namespace annc
