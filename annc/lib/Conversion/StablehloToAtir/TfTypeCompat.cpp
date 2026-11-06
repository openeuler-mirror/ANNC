#include "Conversion/StablehloToAtir/StablehloSink.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;

namespace annc::tfcompat {
namespace {

struct TfStringType : public Type::TypeBase<TfStringType, Type, TypeStorage> {
  using Base::Base;
  static constexpr llvm::StringRef name = "string";
};

struct TfTagTypeStorage : public TypeStorage {
  explicit TfTagTypeStorage(StringRef tag) : tag(tag) {}
  using KeyTy = StringRef;
  bool operator==(const KeyTy& key) const { return tag == key; }
  static llvm::hash_code hashKey(const KeyTy& key) { return hash_value(key); }
  static TfTagTypeStorage* construct(TypeStorageAllocator& allocator,
                                     KeyTy key) {
    return new (allocator.allocate<TfTagTypeStorage>())
        TfTagTypeStorage(allocator.copyInto(key));
  }
  StringRef tag;
};

struct TfTagType : public Type::TypeBase<TfTagType, Type, TfTagTypeStorage> {
  using Base::Base;
  // Registration hook only; parsing goes through the dialect-level parseType.
  static constexpr llvm::StringRef name = "tf_tag";
  static TfTagType get(MLIRContext* context, StringRef tag) {
    return Base::get(context, tag);
  }
  StringRef getTag() const { return getImpl()->tag; }
};

class TfTypeCompatDialect : public Dialect {
 public:
  static StringRef getDialectNamespace() { return "tf_type"; }

  explicit TfTypeCompatDialect(MLIRContext* context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<TfTypeCompatDialect>()) {
    addTypes<TfStringType, TfTagType>();
  }

  // Any `!tf_type.<tag>` parses; `string` gets the dedicated carrier type.
  Type parseType(DialectAsmParser& parser) const override {
    std::string tag;
    if (parser.parseKeywordOrString(&tag)) return Type();
    if (tag == "string") return TfStringType::get(parser.getContext());
    return TfTagType::get(parser.getContext(), tag);
  }

  void printType(Type type, DialectAsmPrinter& printer) const override {
    if (auto tagType = dyn_cast<TfTagType>(type)) {
      printer.printKeywordOrString(tagType.getTag());
      return;
    }
    printer << "string";
  }

};

}  // namespace

bool isTfStringType(Type type) { return isa<TfStringType>(type); }

void registerTfTypeCompatDialect(DialectRegistry& registry) {
  registry.insert<TfTypeCompatDialect>();
}
}  // namespace annc::tfcompat
