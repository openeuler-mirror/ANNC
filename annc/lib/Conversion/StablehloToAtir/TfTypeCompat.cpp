#include "Conversion/StablehloToAtir/StablehloSink.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;

namespace annc::tfcompat {
namespace {

struct TfTagTypeStorage : public TypeStorage {
  TfTagTypeStorage(StringRef tag, ArrayRef<Type> params)
      : tag(tag), params(params) {}
  using KeyTy = std::pair<StringRef, ArrayRef<Type>>;
  bool operator==(const KeyTy& key) const {
    return tag == key.first && params == key.second;
  }
  static llvm::hash_code hashKey(const KeyTy& key) {
    return llvm::hash_combine(hash_value(key.first), key.second);
  }
  static TfTagTypeStorage* construct(TypeStorageAllocator& allocator,
                                     KeyTy key) {
    return new (allocator.allocate<TfTagTypeStorage>())
        TfTagTypeStorage(allocator.copyInto(key.first),
                         allocator.copyInto(key.second));
  }
  StringRef tag;
  ArrayRef<Type> params;
};

struct TfTagType : public Type::TypeBase<TfTagType, Type, TfTagTypeStorage> {
  using Base::Base;
  static constexpr llvm::StringRef name = "tf_tag";
  static TfTagType get(MLIRContext* context, StringRef tag,
                       ArrayRef<Type> params = {}) {
    return Base::get(context, tag, params);
  }
  StringRef getTag() const { return getImpl()->tag; }
  ArrayRef<Type> getParams() const { return getImpl()->params; }
};

class TfTypeCompatDialect : public Dialect {
 public:
  static StringRef getDialectNamespace() { return "tf_type"; }

  explicit TfTypeCompatDialect(MLIRContext* context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<TfTypeCompatDialect>()) {
    addTypes<TfTagType>();
  }

  // Any `!tf_type.<tag>` parses (params round-trip); `string` is a bare tag.
  Type parseType(DialectAsmParser& parser) const override {
    std::string tag;
    if (parser.parseKeywordOrString(&tag)) return Type();
    SmallVector<Type> params;
    if (succeeded(parser.parseOptionalLess())) {
      if (parser.parseCommaSeparatedList([&]() -> ParseResult {
            Type param;
            if (parser.parseType(param)) return failure();
            params.push_back(param);
            return success();
          }))
        return Type();
      if (parser.parseGreater()) return Type();
    }
    return TfTagType::get(parser.getContext(), tag, params);
  }

  void printType(Type type, DialectAsmPrinter& printer) const override {
    auto tagType = cast<TfTagType>(type);
    printer.printKeywordOrString(tagType.getTag());
    ArrayRef<Type> params = tagType.getParams();
    if (params.empty()) return;
    printer << "<";
    llvm::interleaveComma(params, printer,
                          [&](Type param) { printer.printType(param); });
    printer << ">";
  }
};
}  // namespace

bool isTfStringType(Type type) {
  auto tag = dyn_cast<TfTagType>(type);
  return tag && tag.getTag() == "string";
}

void registerTfTypeCompatDialect(DialectRegistry& registry) {
  registry.insert<TfTypeCompatDialect>();
}
}  // namespace annc::tfcompat
