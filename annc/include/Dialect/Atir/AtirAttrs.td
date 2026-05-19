//===----------------------------------------------------------------------===//
// Atir Attribute definitions
//===----------------------------------------------------------------------===//
#ifndef ATIR_ATTRS_TD
#define ATIR_ATTRS_TD
include "Dialect/Atir/AtirDialect.td"

def Atir_MemTypeAttr : EnumAttr<Atir_Dialect, Atir_MemType, "MemTypeAttr"> {
  let cppNamespace = "::atir";
}

def Atir_TilingAttr : AttrDef<Atir_Dialect, "Tiling", []> {
  let summary = "Tiling infomation for tensor";
  let mnemonic = "Tiling";
  let cppNamespace = "::atir";
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

def Atir_LoweringConfigAttr : AttrDef<Atir_Dialect, "LoweringConfig", []> {
  let summary = "Drive lowering of an operation";
  let mnemonic = "lowering_config";
  let cppNamespace = "::atir";
  let parameters = (ins
    OptionalParameter<"::mlir::ArrayAttr">:$distribution,  // e.g. = [128, 128, 0] 128 * 128 on M & N per thread
    OptionalParameter<"::mlir::ArrayAttr">:$cache_parallel, // e.g. = [64, 64, 0] block on M & N
    OptionalParameter<"::mlir::ArrayAttr">:$cache_reduction,   // e.g. = [0, 0, 16] block on reduce dim eg K
    OptionalParameter<"::mlir::ArrayAttr">:$vector_common_parallel,   // e.g. = [4, 4, 0] micro kernel on M & N
    OptionalParameter<"::mlir::ArrayAttr">:$vector_reduction,   // e.g. = [0, 0, 4] micro kernel on reduce dim ex K
    OptionalParameter<"::mlir::ArrayAttr">:$vector_inner_parallel   // e.g. = [0, 0, 0] inner parallel for instruction pipeline
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::MLIRContext *":$context), [{
      return $_get(context);
    }]>,
    AttrBuilderWithInferredContext<(ins "::mlir::MLIRContext *":$context,
                                        "::llvm::SmallVector<int64_t>":$distribution,
                                        "::llvm::SmallVector<int64_t>":$cache_parallel,
                                        "::llvm::SmallVector<int64_t>":$cache_reduction,
                                        "::llvm::SmallVector<int64_t>":$vector_common_parallel,
                                        "::llvm::SmallVector<int64_t>":$vector_reduction,
                                        "::llvm::SmallVector<int64_t>":$vector_inner_parallel),
    [{
      ::mlir::OpBuilder builder(context);
      auto distribution_ = builder.getI64ArrayAttr(distribution);
      auto cache_parallel_ = builder.getI64ArrayAttr(cache_parallel);
      auto cache_reduction_ = builder.getI64ArrayAttr(cache_reduction);
      auto vector_common_parallel_ = builder.getI64ArrayAttr(vector_common_parallel);
      auto vector_reduction_ = builder.getI64ArrayAttr(vector_reduction);
      auto vector_inner_parallel_ = builder.getI64ArrayAttr(vector_inner_parallel);

      return $_get(context, distribution_, cache_parallel_, cache_reduction_, vector_common_parallel_, vector_reduction_, vector_inner_parallel_);
    }]>,
    AttrBuilderWithInferredContext<(ins "const LoweringConfigAttr &":$loweringConfigAttr), [{
      return $_get(loweringConfigAttr.getContext(), loweringConfigAttr.getDistribution(), loweringConfigAttr.getCacheParallel(),
            loweringConfigAttr.getCacheReduction(), loweringConfigAttr.getVectorCommonParallel(), loweringConfigAttr.getVectorReduction(), loweringConfigAttr.getVectorInnerParallel());
    }]>
  ];
  let extraClassDeclaration = [{
    std::vector<int64_t> getLoweringConfigDistribution() const;
    std::vector<int64_t> getLoweringConfigCacheParallel() const;
    std::vector<int64_t> getLoweringConfigCacheReduction() const;
    std::vector<int64_t> getLoweringConfigVectorCommonParallel() const;
    std::vector<int64_t> getLoweringConfigVectorReduction() const;
    std::vector<int64_t> getLoweringConfigVectorInnerParallel() const;
  }];

  let extraClassDefinition = [{
    std::vector<int64_t> LoweringConfigAttr::getLoweringConfigDistribution() const {
      std::vector<int64_t> distributions;
      for (auto& attr : getDistribution()) {
        distributions.push_back(dyn_cast<::mlir::IntegerAttr>(attr).getInt());
      }
      return distributions;
    }
    std::vector<int64_t> LoweringConfigAttr::getLoweringConfigCacheParallel() const {
      std::vector<int64_t> cacheParallels;
      for (auto& attr : getCacheParallel()) {
        cacheParallels.push_back(dyn_cast<::mlir::IntegerAttr>(attr).getInt());
      }
      return cacheParallels;
    }
    std::vector<int64_t> LoweringConfigAttr::getLoweringConfigCacheReduction() const {
      std::vector<int64_t> cacheReductions;
      for (auto& attr : getCacheReduction()) {
        cacheReductions.push_back(dyn_cast<::mlir::IntegerAttr>(attr).getInt());
      }
      return cacheReductions;
    }
    std::vector<int64_t> LoweringConfigAttr::getLoweringConfigVectorCommonParallel() const {
      std::vector<int64_t> vectorCommonParallels;
      for (auto& attr : getVectorCommonParallel()) {
        vectorCommonParallels.push_back(dyn_cast<::mlir::IntegerAttr>(attr).getInt());
      }
      return vectorCommonParallels;
    }
    std::vector<int64_t> LoweringConfigAttr::getLoweringConfigVectorReduction() const {
      std::vector<int64_t> vectorReductions;
      for (auto& attr : getVectorReduction()) {
        vectorReductions.push_back(dyn_cast<::mlir::IntegerAttr>(attr).getInt());
      }
      return vectorReductions;
    }
    std::vector<int64_t> LoweringConfigAttr::getLoweringConfigVectorInnerParallel() const {
      std::vector<int64_t> vectorInnerParallels;
      for (auto& attr : getVectorInnerParallel()) {
        vectorInnerParallels.push_back(dyn_cast<::mlir::IntegerAttr>(attr).getInt());
      }
      return vectorInnerParallels;
    }
  }];
}

#endif // ATIR_ATTRS_TD