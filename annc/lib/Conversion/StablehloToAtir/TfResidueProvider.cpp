// TF frontend adapter for the residue sink (V2 emission): three residue forms
// (F1 unregistered `tf.*`, F2 stablehlo.custom_call, F3 declared
// `call_target_name`) and the TF -> atir.*_v2 op map. Construction is
// convention-driven (V2OpMatcher.h); string elements are `!atir.string`.
#include "Conversion/StablehloToAtir/ResidueProvider.h"

#include "mlir/IR/Verifier.h"

#include <string>
#include <utility>

#include "Conversion/StablehloToAtir/StablehloSink.h"
#include "Conversion/StablehloToAtir/V2OpMatcher.h"
#include "Dialect/Atir/AtirOps.h"
#include "Helper.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "nlohmann/json.hpp"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace annc {
namespace {

enum class ResidueForm {
  TfGenericOp,          // F1: unregistered generic `tf.*` op
  StablehloCustomCall,  // F2: `stablehlo.custom_call` wrapper
  DeclaredOp,           // F3: other unregistered op + call_target_name
};

// Recursive JSON -> Attribute translation; a failed element abandons the container.
Attribute jsonToAttr(MLIRContext* ctx, const nlohmann::json& value) {
  if (value.is_boolean()) return BoolAttr::get(ctx, value.get<bool>());
  if (value.is_number_integer())
    return IntegerAttr::get(IntegerType::get(ctx, 64), value.get<int64_t>());
  if (value.is_number_float())
    return FloatAttr::get(Float64Type::get(ctx), value.get<double>());
  if (value.is_string()) return StringAttr::get(ctx, value.get<std::string>());
  if (value.is_array()) {
    SmallVector<Attribute> elems;
    for (const auto& item : value) {
      Attribute elem = jsonToAttr(ctx, item);
      if (!elem) return Attribute();
      elems.push_back(elem);
    }
    return ArrayAttr::get(ctx, elems);
  }
  if (value.is_object()) {
    SmallVector<NamedAttribute> nested;
    for (auto& [key, val] : value.items()) {
      Attribute attr = jsonToAttr(ctx, val);
      if (!attr) return Attribute();
      nested.push_back(NamedAttribute(StringAttr::get(ctx, key), attr));
    }
    return DictionaryAttr::get(ctx, nested);
  }
  return Attribute();
}

// Decodes an F2 backend_config JSON into typed attributes; absent/invalid
// payloads yield an empty dict, undecodable leaves are skipped.
DictionaryAttr decodeBackendConfig(MLIRContext* ctx, StringAttr json) {
  auto parsed = nlohmann::json::parse(json.str(), /*cb=*/nullptr,
                                      /*allow_exceptions=*/false);
  if (parsed.is_discarded() || !parsed.is_object())
    return DictionaryAttr::get(ctx, {});
  SmallVector<NamedAttribute> attrs;
  for (auto& [key, value] : parsed.items()) {
    if (Attribute attr = jsonToAttr(ctx, value))
      attrs.push_back(NamedAttribute(StringAttr::get(ctx, key), attr));
  }
  return DictionaryAttr::get(ctx, attrs);
}

llvm::StringRef stripTfQualifier(llvm::StringRef name) {
  name.consume_front("tf.");
  return name;
}

llvm::StringRef declaredTfOpName(DictionaryAttr attrs) {
  if (!attrs) return {};
  if (auto call = attrs.getAs<StringAttr>("call_target_name"))
    return call.getValue();
  return {};
}

std::optional<ResidueView> residueViewOf(Operation* op, ResidueForm form) {
  const bool registered = op->getName().isRegistered();
  llvm::StringRef name = op->getName().getStringRef();
  switch (form) {
    case ResidueForm::TfGenericOp: {
      if (registered || !name.starts_with("tf.")) return std::nullopt;
      DictionaryAttr attrs = op->getAttrDictionary();
      llvm::StringRef declared = declaredTfOpName(attrs);
      return ResidueView{
          stripTfQualifier(declared.empty() ? name : declared).str(), attrs};
    }
    case ResidueForm::StablehloCustomCall: {
      auto customCall = dyn_cast<stablehlo::CustomCallOp>(op);
      if (!customCall) return std::nullopt;
      DictionaryAttr semantic = op->getAttrDictionary();
      if (auto config = customCall.getBackendConfig()) {
        if (auto dict = dyn_cast<DictionaryAttr>(*config)) {
          semantic = dict;
        } else if (auto str = dyn_cast<StringAttr>(*config); !str.empty()) {
          semantic = decodeBackendConfig(op->getContext(), str);
        }
      }
      return ResidueView{
          stripTfQualifier(customCall.getCallTargetName()).str(), semantic};
    }
    case ResidueForm::DeclaredOp: {
      if (registered || name.starts_with("tf.")) return std::nullopt;
      if (isa<stablehlo::CustomCallOp>(op)) return std::nullopt;
      DictionaryAttr attrs = op->getAttrDictionary();
      llvm::StringRef declared = declaredTfOpName(attrs);
      if (declared.empty()) return std::nullopt;
      return ResidueView{stripTfQualifier(declared).str(), attrs};
    }
  }
  return std::nullopt;
}

bool isTfResidue(Operation* op) {
  llvm::StringRef name = op->getName().getStringRef();
  if (!op->getName().isRegistered() && name.starts_with("tf.")) return true;
  if (isa<stablehlo::CustomCallOp>(op)) return true;
  if (!op->getName().isRegistered() && !name.starts_with("tf.") &&
      !declaredTfOpName(op->getAttrDictionary()).empty())
    return true;
  return false;
}

// TF-op -> atir-v2 mnemonic map: the only mapping contract; absent ops fall
// back to `atir.unsupported` (unknown_op). Complement-only targets.
constexpr std::pair<llvm::StringRef, llvm::StringRef> kOpMap[] = {
    // View carriers keep the `annc.` prefix (stripTfQualifier strips `tf.` only).
    {"annc.getDim", "get_dim_v2"},
    {"annc.setDim", "set_dim_v2"},
    {"annc.encodeString", "encode_string_v2"},
    {"annc.decodeString", "decode_string_v2"},
    {"ResourceGather", "resource_gather_v2"},
    {"SparseFillEmptyRows", "sparse_fill_empty_rows_v2"},
    {"SparseReshape", "sparse_reshape_v2"},
    {"SparseSegmentMean", "sparse_segment_mean_v2"},
    {"SparseSegmentSum", "sparse_segment_sum_v2"},
    {"SparseTensorDenseMatMul", "sparse_tensor_dense_mat_mul_v2"},
    {"SparseToDense", "sparse_to_dense_v2"},
    {"StaticRegexReplace", "static_regex_replace_v2"},
    {"StringSplit", "string_split_v2"},
    {"StringToHashBucketFast", "string_to_hash_bucket_fast_v2"},
    {"StringToNumber", "string_to_number_v2"},
    {"TopK", "top_k_v2"},
    {"TopKV2", "top_k_v2"},
    {"Variable", "variable_v2"},
    {"VariableV2", "variable_v2"},
};
class TfResidueProvider final : public ResidueProvider {
 public:
  bool isResidue(Operation* op) const override { return isTfResidue(op); }

  bool hasMapping(const ResidueView& view,
                  MLIRContext* ctx) const override {
    if (view.opType == "Const") return true;
    return resolveV2Target(ctx, view.opType, kOpMap).has_value();
  }

  std::optional<ResidueView> match(Operation* op) const override {
    for (ResidueForm form : kForms) {
      if (std::optional<ResidueView> view = residueViewOf(op, form))
        return view;
    }
    return std::nullopt;
  }

  Type stringElementType(Type elementType, MLIRContext* ctx) const override {
    if (tfcompat::isTfStringType(elementType))
      return atir::StringType::get(ctx);
    return {};
  }

  LogicalResult emitMapped(ConversionPatternRewriter& rewriter, Operation* op,
                           const ResidueView& view, ArrayRef<Value> operands,
                           ArrayRef<Type> resultTypes,
                           llvm::SmallVectorImpl<Value>& results) const override {
    if (view.opType == "Const") return failure();
    std::optional<RegisteredOperationName> target =
        resolveV2Target(rewriter.getContext(), view.opType, kOpMap);
    if (!target) return failure();

    SmallVector<NamedAttribute> derived;
    buildDerived(op, view, *target, resultTypes, derived);
    return emitV2Op(rewriter, op->getLoc(), *target, operands, resultTypes,
                    view.sourceAttrs, {}, derived, results);
  }

 private:
  static constexpr ResidueForm kForms[] = {
      ResidueForm::TfGenericOp, ResidueForm::StablehloCustomCall,
      ResidueForm::DeclaredOp};

  static bool declares(RegisteredOperationName target, llvm::StringRef name) {
    return llvm::is_contained(target.getAttributeNames(),
                              StringAttr::get(target.getContext(), name));
  }

  // Contract-recovered attributes: fired only when the target declares the
  // attribute and the residue carries none.
  void buildDerived(Operation* op, const ResidueView& view,
                    RegisteredOperationName target,
                    ArrayRef<Type> resultTypes,
                    llvm::SmallVectorImpl<NamedAttribute>& derived) const {
    MLIRContext* ctx = op->getContext();

    // Symbol ops need a sym_name the source op does not carry; derive one.
    if (declares(target, "sym_name") && !view.sourceAttrs.get("sym_name")) {
      std::string name = getLocName(op);
      if (name.empty()) name = ("v2_" + llvm::Twine(symOrdinal_++)).str();
      derived.emplace_back(StringAttr::get(ctx, "sym_name"),
                           StringAttr::get(ctx, name));
    }
  }

  mutable int symOrdinal_ = 0;
};

}  // namespace

std::unique_ptr<ResidueProvider> createTfResidueProvider() {
  return std::make_unique<TfResidueProvider>();
}

}  // namespace annc
