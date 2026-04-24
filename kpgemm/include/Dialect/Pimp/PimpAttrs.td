//===----------------------------------------------------------------------===//
// Pimp Attribute definitions
//===----------------------------------------------------------------------===//
#ifndef PIMP_ATTRS_TD
#define PIMP_ATTRS_TD
include "Dialect/Pimp/PimpDialect.td"

def Pimp_MemTypeAttr : EnumAttr<Pimp_Dialect, Pimp_MemType, "MemTypeAttr"> {
  let cppNamespace = "::pimp";
}

def Pimp_TilingAttr : AttrDef<Pimp_Dialect, "Tiling", []> {
  let summary = "Tiling infomation for tensor";
  let mnemonic = "Tiling";
  let cppNamespace = "::pimp";
  let parameters = (ins
    OptionalParameter<"::mlir::ArrayAttr">:$axes,  // e.g. = [0, 1] mesh tensor on N & H dimension
    OptionalParameter<"::mlir::ArrayAttr">:$start, // e.g. = [[0, 10], [0, 100]] start position on each dimension
    OptionalParameter<"::mlir::ArrayAttr">:$size   // e.g. = [[10, 10], [100, 100]] line size on each dimension
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::MLIRContext *":$context), [{
      return $_get(context);
    }]>,
    AttrBuilderWithInferredContext<(ins "::mlir::MLIRContext *":$context,
                                        "::llvm::SmallVector<int64_t>":$axes,
                                        "::llvm::SmallVector<::llvm::SmallVector<int64_t>>":$start,
                                        "::llvm::SmallVector<::llvm::SmallVector<int64_t>>":$size), 
    [{
      ::mlir::OpBuilder builder(context);
      auto axes_ = builder.getI64ArrayAttr(axes);
      ::llvm::SmallVector<::mlir::Attribute> startArr = {};
      for (auto& arr : start) {
        startArr.push_back(builder.getI64ArrayAttr(arr));
      }
      auto start_ = ::mlir::ArrayAttr::get(context, startArr);
      ::llvm::SmallVector<::mlir::Attribute> sizeArr = {};
      for (auto& arr : size) {
        sizeArr.push_back(builder.getI64ArrayAttr(arr));
      }
      auto size_ = ::mlir::ArrayAttr::get(context, sizeArr);
      return $_get(context, axes_, start_, size_);
    }]>,
    AttrBuilderWithInferredContext<(ins "const TilingAttr &":$tilingAttr), [{
      return $_get(tilingAttr.getContext(), tilingAttr.getAxes(), tilingAttr.getStart(), tilingAttr.getSize());
    }]>
  ];
  let extraClassDeclaration = [{
    std::vector<int64_t> getTilingAxes() const;
    std::vector<std::vector<int64_t>> getTilingStart() const;
    std::vector<std::vector<int64_t>> getTilingSize() const;
  }];

  let extraClassDefinition = [{
    std::vector<int64_t> TilingAttr::getTilingAxes() const {
      std::vector<int64_t> axes;
      for (auto& attr : getAxes()) {
        axes.push_back(dyn_cast<::mlir::IntegerAttr>(attr).getInt());
      }
      return axes;
    }
    std::vector<std::vector<int64_t>> TilingAttr::getTilingStart() const {
      std::vector<std::vector<int64_t>> starts;
      for (auto& arr : getStart()) {
        std::vector<int64_t> start;
        for (auto& attr : dyn_cast<::mlir::ArrayAttr>(arr)) {
          start.push_back(dyn_cast<::mlir::IntegerAttr>(attr).getInt());
        }
        starts.push_back(start);
      }
      return starts;
    }
    std::vector<std::vector<int64_t>> TilingAttr::getTilingSize() const {
      std::vector<std::vector<int64_t>> sizes;
      for (auto& arr : getSize()) {
        std::vector<int64_t> size;
        for (auto& attr : dyn_cast<::mlir::ArrayAttr>(arr)) {
          size.push_back(dyn_cast<::mlir::IntegerAttr>(attr).getInt());
        }
        sizes.push_back(size);
      }
      return sizes;
    }
  }];
}

#endif // PIMP_ATTRS_TD