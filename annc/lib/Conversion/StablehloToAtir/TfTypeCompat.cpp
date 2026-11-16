#include "Conversion/StablehloToAtir/StablehloSink.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace annc::tfcompat {
namespace {

struct TfStringType : public Type::TypeBase<TfStringType, Type, TypeStorage> {
  using Base::Base;
  static constexpr llvm::StringRef name = "string";
};

// Parameterized storage keyed by the source tag (resource/variant/...).
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
  // Required by addTypes registration; never used for parsing — the
  // dialect-level parseType hook handles every tag.
  static constexpr llvm::StringRef name = "tf_tag";
  static TfTagType get(MLIRContext* context, StringRef tag,
                       ArrayRef<Type> params = {}) {
    return Base::get(context, tag, params);
  }
  StringRef getTag() const { return getImpl()->tag; }
  ArrayRef<Type> getParams() const { return getImpl()->params; }
};

// Parameterized attribute storage: dimension list with negatives encoding `?`
// (unknown extent); `unranked` (*) carries an empty dim list.
struct TfShapeAttrStorage : public AttributeStorage {
  using KeyTy = std::pair<ArrayRef<int64_t>, bool>;
  explicit TfShapeAttrStorage(KeyTy key)
      : shape(key.first), unranked(key.second) {}
  bool operator==(const KeyTy& key) const {
    return shape == key.first && unranked == key.second;
  }
  static llvm::hash_code hashKey(const KeyTy& key) {
    return llvm::hash_combine(key.first, key.second);
  }
  static TfShapeAttrStorage* construct(AttributeStorageAllocator& allocator,
                                       KeyTy key) {
    return new (allocator.allocate<TfShapeAttrStorage>())
        TfShapeAttrStorage({allocator.copyInto(key.first), key.second});
  }
  ArrayRef<int64_t> shape;
  bool unranked;
};

struct TfShapeAttr
    : public Attribute::AttrBase<TfShapeAttr, Attribute, TfShapeAttrStorage> {
  using Base::Base;
  static constexpr llvm::StringRef name = "shape";
  static TfShapeAttr get(MLIRContext* context, ArrayRef<int64_t> shape,
                         bool unranked = false) {
    assert((!unranked || shape.empty()) && "unranked shape carries no dims");
    return Base::get(context, TfShapeAttrStorage::KeyTy{shape, unranked});
  }
  ArrayRef<int64_t> getShape() const { return getImpl()->shape; }
  bool isUnranked() const { return getImpl()->unranked; }
};

class TfTypeCompatDialect : public Dialect {
 public:
  static StringRef getDialectNamespace() { return "tf_type"; }

  explicit TfTypeCompatDialect(MLIRContext* context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<TfTypeCompatDialect>()) {
    addTypes<TfStringType, TfTagType>();
    addAttributes<TfShapeAttr>();
  }

  // `string` gets the dedicated carrier type; every other tag round-trips as
  // an opaque type with its parameter list.
  Type parseType(DialectAsmParser& parser) const override {
    std::string tag;
    if (parser.parseKeywordOrString(&tag)) return Type();
    if (tag == "string") return TfStringType::get(parser.getContext());
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
    if (auto tagType = dyn_cast<TfTagType>(type)) {
      printer.printKeywordOrString(tagType.getTag());
      ArrayRef<Type> params = tagType.getParams();
      if (params.empty()) return;
      printer << "<";
      llvm::interleaveComma(params, printer,
                            [&](Type param) { printer.printType(param); });
      printer << ">";
      return;
    }
    printer << "string";
  }

  Attribute parseAttribute(DialectAsmParser& parser, Type) const override {
    llvm::StringRef mnemonic;
    if (parser.parseKeyword(&mnemonic)) return Attribute();
    if (mnemonic != "shape") {
      parser.emitError(parser.getNameLoc())
          << "unsupported 'tf_type' attribute: #" << mnemonic;
      return Attribute();
    }
    if (parser.parseLess()) return Attribute();
    SmallVector<int64_t> dims;
    bool unranked = false;
    if (succeeded(parser.parseOptionalStar())) {
      unranked = true;
    } else if (succeeded(parser.parseOptionalLSquare())) {
      bool closed = succeeded(parser.parseOptionalRSquare());
      while (!closed) {
        int64_t dim = -1;
        if (failed(parser.parseOptionalQuestion())) {
          if (parser.parseInteger(dim) || dim < 0) {
            parser.emitError(parser.getCurrentLocation())
                << "invalid dimension in 'tf_type.shape' attribute";
            return Attribute();
          }
        }
        dims.push_back(dim);
        closed = succeeded(parser.parseOptionalRSquare());
        if (!closed && failed(parser.parseOptionalComma())) {
          parser.emitError(parser.getCurrentLocation())
              << "expected ',' or ']' in 'tf_type.shape' attribute";
          return Attribute();
        }
      }
    }
    if (parser.parseGreater()) return Attribute();
    return TfShapeAttr::get(parser.getContext(), dims, unranked);
  }

  void printAttribute(Attribute attr,
                      DialectAsmPrinter& printer) const override {
    auto shape = cast<TfShapeAttr>(attr);
    if (shape.isUnranked()) {
      printer << "shape<*>";
      return;
    }
    printer << "shape<[";
    llvm::interleave(
        shape.getShape(), printer,
        [&](int64_t dim) {
          if (dim < 0)
            printer << "?";
          else
            printer << dim;
        },
        ", ");
    printer << "]>";
  }
};

}  // namespace

bool isTfStringType(Type type) {
  return isa<TfStringType>(type);
}

void registerTfTypeCompatDialect(DialectRegistry& registry) {
  registry.insert<TfTypeCompatDialect>();
}
}  // namespace annc::tfcompat
