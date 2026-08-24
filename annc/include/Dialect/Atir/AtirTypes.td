//===----------------------------------------------------------------------===//
// Atir Type definitions
//===----------------------------------------------------------------------===//
#ifndef ATIR_TYPES_TD
#define ATIR_TYPES_TD
include "mlir/IR/AttrTypeBase.td"
include "Dialect/Atir/AtirDialect.td"

class Atir_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<Atir_Dialect, name, traits> {
    let mnemonic = _mnemonic;
}

def Atir_Tensor : Atir_TypeDef<"Tensor", "tensor"> {
  let summary = "Tensor descriptor type (`::atir::TensorType`) in Astralax IR type system";

  let description = [{
    Tensor descriptor.
    The shape and elementType descripbe tensor size and precision.
    The stride describes the alignment on each axis, has the same rank with shape.
    The layout is the name used to mark the stride of tensor.
    The memType is an enum attribute used to describe memory type (location).
    The address is an abstract address, probably can be used to locate tensor in memory space.
    The deviceParallel describes tensor tiling according to device number.
    The onchipParallel describes tensor tiling according to onchip thread/block number.
    The cacheData is a dense elements, which is defined for interpretation inside IR.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "::mlir::Type":$elementType,
    OptionalParameter<"::mlir::StringAttr">:$name,
    OptionalParameter<"::mlir::Attribute">:$encoding,
    OptionalParameter<"::mlir::ArrayAttr">:$stride,
    OptionalParameter<"::mlir::StringAttr">:$layout,
    OptionalParameter<"::atir::MemTypeAttr">:$memType,
    OptionalParameter<"::mlir::IntegerAttr">:$address,
    OptionalParameter<"::atir::TilingAttr">:$deviceParallel,
    OptionalParameter<"::atir::TilingAttr">:$onchipParallel,
    OptionalParameter<"::mlir::DenseElementsAttr">:$cacheData,
    OptionalParameter<"::mlir::BoolAttr">:$rankKnown
  );

  let extraClassDeclaration = [{
    TensorType clone();
    TensorType cloneRawType();
    TensorType cloneWithName(::mlir::StringAttr name);
    TensorType cloneWithDeviceTiling(::atir::TilingAttr tiling);
    TensorType cloneWithOnchipTiling(::atir::TilingAttr tiling);
    TensorType cloneWithCacheData(::mlir::DenseElementsAttr cacheData);
    TensorType cloneWithDynamicShape(std::vector<int64_t> dynIdx);
    bool hasKnownRank() const;
    
    void setEncoding(::mlir::Attribute encoding);
    void setName(::mlir::StringAttr name);
    void setStride(::mlir::ArrayAttr stride);
    void setLayout(::mlir::StringAttr layout);
    void setMemType(::atir::MemTypeAttr memType);
    void setAddress(::mlir::IntegerAttr address);
    void setDeviceParallel(::atir::TilingAttr deviceParallel);
    void setOnchipParallel(::atir::TilingAttr onchipParallel);
    void setCacheData(::mlir::DenseElementsAttr cacheData);

    std::vector<int64_t> getValueOfShape();
    std::string getValueOfElementType();
    std::string getValueOfName();
    std::vector<int64_t> getValueOfStride();
    std::string getValueOfLayout();
    atir::MemType getValueOfMemType();
    int64_t getValueOfAddress();
  }];
  let extraClassDefinition = [{
    TensorType TensorType::clone() {
      return TensorType::get(getContext(), getShape(), getElementType(), getName(), getEncoding(), getStride(), getLayout(), getMemType(), getAddress(), getDeviceParallel(), getOnchipParallel(), getCacheData(), getRankKnown());
    }
    TensorType TensorType::cloneRawType() {
      ::mlir::Attribute encoding{};
      ::mlir::StringAttr name{};
      ::mlir::ArrayAttr stride{};
      ::mlir::StringAttr layout{};
      ::atir::MemTypeAttr memType{};
      ::mlir::IntegerAttr address{};
      ::atir::TilingAttr deviceParallel{};
      ::atir::TilingAttr onchipParallel{};
      ::mlir::DenseElementsAttr cacheData{};
      return TensorType::get(getContext(), getShape(), getElementType(), name, encoding, stride, layout, memType, address, deviceParallel, onchipParallel, cacheData, getRankKnown());
    }
    TensorType TensorType::cloneWithName(::mlir::StringAttr name) {
      return TensorType::get(getContext(), getShape(), getElementType(), name, getEncoding(), getStride(), getLayout(), getMemType(), getAddress(), getDeviceParallel(), getOnchipParallel(), getCacheData(), getRankKnown());
    }
    TensorType TensorType::cloneWithDeviceTiling(::atir::TilingAttr tiling) {
      return TensorType::get(getContext(), getShape(), getElementType(), getName(), getEncoding(), getStride(), getLayout(), getMemType(), getAddress(), tiling, getOnchipParallel(), getCacheData(), getRankKnown());
    }
    TensorType TensorType::cloneWithOnchipTiling(::atir::TilingAttr tiling) {
      return TensorType::get(getContext(), getShape(), getElementType(), getName(), getEncoding(), getStride(), getLayout(), getMemType(), getAddress(), getDeviceParallel(), tiling, getCacheData(), getRankKnown());
    }
    TensorType TensorType::cloneWithCacheData(::mlir::DenseElementsAttr cacheData) {
      return TensorType::get(getContext(), getShape(), getElementType(), getName(), getEncoding(), getStride(), getLayout(), getMemType(), getAddress(), getDeviceParallel(), getOnchipParallel(), cacheData, getRankKnown());
    }
    TensorType TensorType::cloneWithDynamicShape(std::vector<int64_t> dynIdx) {
      std::vector<int64_t> shape = getShape();
      for (int64_t idx : dynIdx) {
        shape[idx] =  ShapedType::kDynamic;
      }
      return TensorType::get(getContext(), shape, getElementType(), getName(), getEncoding(), getStride(), getLayout(), getMemType(), getAddress(), getDeviceParallel(), getOnchipParallel(), getCacheData(), getRankKnown());
    }

    bool TensorType::hasKnownRank() const {
      return !getRankKnown() || getRankKnown().getValue();
    }


    void TensorType::setEncoding(::mlir::Attribute encoding) {
      getImpl()->encoding = encoding;
    }
    void TensorType::setName(::mlir::StringAttr name) {
      getImpl()->name = name;
    }
    void TensorType::setStride(::mlir::ArrayAttr stride) {
      getImpl()->stride = stride;
    }
    void TensorType::setLayout(::mlir::StringAttr layout) {
      getImpl()->layout = layout;
    }
    void TensorType::setMemType(::atir::MemTypeAttr memType) {
      getImpl()->memType = memType;
    }
    void TensorType::setAddress(::mlir::IntegerAttr address) {
      getImpl()->address = address;
    }
    void TensorType::setDeviceParallel(::atir::TilingAttr deviceParallel) {
      getImpl()->deviceParallel = deviceParallel;
    }
    void TensorType::setOnchipParallel(::atir::TilingAttr onchipParallel) {
      getImpl()->onchipParallel = onchipParallel;
    }
    void TensorType::setCacheData(::mlir::DenseElementsAttr cacheData) {
      getImpl()->cacheData = cacheData;
    }

    std::vector<int64_t> TensorType::getValueOfShape() {
      auto shape_ = getShape();
      std::vector<int64_t> shape(shape_.begin(), shape_.end());
      return shape;
    }
    std::string TensorType::getValueOfElementType() {
      auto elemType = getElementType();
      if (elemType.isF16()) return "f16";
      if (elemType.isF32()) return "f32";
      if (elemType.isF64()) return "f64";
      if (elemType.isF80()) return "f80";
      if (elemType.isF128()) return "f128";
      if (elemType.isSignlessInteger(1)) return "i1";
      if (elemType.isInteger(1)) return "i1";
      if (elemType.isInteger(4)) return "i4";
      if (elemType.isInteger(5)) return "i5";
      if (elemType.isInteger(6)) return "i6";
      if (elemType.isInteger(7)) return "i7";
      if (elemType.isInteger(8)) return "i8";
      if (elemType.isInteger(9)) return "i9";
      if (elemType.isInteger(10)) return "i10";
      if (elemType.isInteger(11)) return "i11";
      if (elemType.isInteger(12)) return "i12";
      if (elemType.isInteger(13)) return "i13";
      if (elemType.isInteger(14)) return "i14";
      if (elemType.isInteger(15)) return "i15";
      if (elemType.isInteger(16)) return "i16";
      if (elemType.isInteger(32)) return "i32";
      if (elemType.isInteger(64)) return "i64";
      return "";
    }
    std::string TensorType::getValueOfName() {
      if (getName() == nullptr) return "";
      return getName().str();
    }
    std::vector<int64_t> TensorType::getValueOfStride() {
      if (getStride() == nullptr) return {};
      std::vector<int64_t> strides;
      for (auto& i : getStride()) {
        strides.push_back(dyn_cast<mlir::IntegerAttr>(i).getInt());
      }
      return strides;
    }
    std::string TensorType::getValueOfLayout() {
      if (getLayout() == nullptr) return "";
      return getLayout().str();
    }
    atir::MemType TensorType::getValueOfMemType() {
      if (getMemType() == nullptr) return atir::MemType::Undefined;
      return getMemType().getValue();
    }
    int64_t TensorType::getValueOfAddress() {
      if (getAddress() == nullptr) return -1;
      return dyn_cast<mlir::IntegerAttr>(getAddress()).getInt();
    }
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "::llvm::ArrayRef<int64_t>":$shape,
                                         "::mlir::Type":$elementType,
                                         "::mlir::DenseElementsAttr":$cacheData),
    [{
      ::mlir::Attribute encoding = {};
      ::mlir::StringAttr name = {};
      ::mlir::ArrayAttr stride = {};
      ::mlir::StringAttr layout = {};
      ::atir::MemTypeAttr memType = {};
      ::mlir::IntegerAttr address = {};
      ::atir::TilingAttr deviceParallel = {};
      ::atir::TilingAttr onchipParallel = {};
      ::mlir::BoolAttr rankKnown = {};
      return Base::get(elementType.getContext(), shape, elementType, name, encoding, stride, layout, memType, address, deviceParallel, onchipParallel, cacheData, rankKnown);
    }]>,
    TypeBuilderWithInferredContext<(ins "::llvm::ArrayRef<int64_t>":$shape,
                                         "::mlir::Type":$elementType,
                                         "::mlir::StringAttr":$name,
                                         "::mlir::DenseElementsAttr":$cacheData),
    [{
      ::mlir::Attribute encoding = {};
      ::mlir::ArrayAttr stride = {};
      ::mlir::StringAttr layout = {};
      ::atir::MemTypeAttr memType = {};
      ::mlir::IntegerAttr address = {};
      ::atir::TilingAttr deviceParallel = {};
      ::atir::TilingAttr onchipParallel = {};
      ::mlir::BoolAttr rankKnown = {};
      return Base::get(elementType.getContext(), shape, elementType, name, encoding, stride, layout, memType, address, deviceParallel, onchipParallel, cacheData, rankKnown);
    }]>,
    TypeBuilderWithInferredContext<(ins "::llvm::ArrayRef<int64_t>":$shape,
                                         "::mlir::Type":$elementType,
                                         CArg<"::mlir::StringAttr", "{}">:$name,
                                         CArg<"::mlir::Attribute", "{}">:$encoding,
                                         CArg<"::mlir::ArrayAttr", "{}">:$stride,
                                         CArg<"::mlir::StringAttr", "{}">:$layout,
                                         CArg<"::atir::MemTypeAttr", "{}">:$memType,
                                         CArg<"::mlir::IntegerAttr", "{}">:$address,
                                         CArg<"::atir::TilingAttr", "{}">:$deviceParallel,
                                         CArg<"::atir::TilingAttr", "{}">:$onchipParallel,
                                         CArg<"::mlir::DenseElementsAttr", "{}">:$cacheData,
                                         CArg<"::mlir::BoolAttr", "{}">:$rankKnown),
    [{
      return Base::get(elementType.getContext(), shape, elementType, name, encoding, stride, layout, memType, address, deviceParallel, onchipParallel, cacheData, rankKnown);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}

def Atir_Unknown : Atir_TypeDef<"Unknown", "unknown"> {
  let summary = "unknown element type preserved by opaque operations";
}

def Atir_Resource : Atir_TypeDef<"Resource", "resource"> {
  let summary = "opaque handle to a runtime resource";
}

def Atir_TensorOrNone : AnyTypeOf<[Atir_Tensor, AnyStaticShapeMemRef, NoneType]>;

#endif // ATIR_TYPES_TD
