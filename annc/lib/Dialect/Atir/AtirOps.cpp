#include "Dialect/Atir/AtirOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace atir;

#include "Dialect/Atir/AtirDialect.cpp.inc"
#include "Dialect/Atir/AtirEnums.cpp.inc"
void AtirDialect::initialize() {
  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "Dialect/Atir/AtirAttr.cpp.inc"
  >();
  addOperations<
    #define GET_OP_LIST
    #include "Dialect/Atir/AtirOps.cpp.inc"
  >();
  addTypes<
    #define GET_TYPEDEF_LIST
    #include "Dialect/Atir/AtirTypes.cpp.inc"
  >();
}

#define GET_ATTRDEF_CLASSES
#include "Dialect/Atir/AtirAttr.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/Atir/AtirOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Atir/AtirTypes.cpp.inc"

namespace atir {
#define PARSE_TENSOR_ERROR(err) {                              \
  odsParser.emitError(odsParser.getCurrentLocation(),          \
                   std::string(errPrintMsg)+std::string(err)); \
  return {};}

constexpr const char errPrintMsg[] = "Failed to parse: ";
constexpr const char kTensorEncoding[] = "encoding";
constexpr const char kTensorName[] = "name";
constexpr const char kTensorStride[] = "stride";
constexpr const char kTensorLayout[] = "layout";
constexpr const char kTensorMemType[] = "memory";
constexpr const char kTensorAddress[] = "address";
constexpr const char kTensorDeviceParl[] = "device";
constexpr const char kTensorOnchipParl[] = "onchip";
constexpr const char kTensorData[] = "data";

Type TensorType::parse(AsmParser &odsParser) {
  if (odsParser.parseLess())
    PARSE_TENSOR_ERROR("Less")
  SmallVector<int64_t> shape;
  if (odsParser.parseDimensionList(shape, true))
    PARSE_TENSOR_ERROR("shape")
  Type elementType;
  if (odsParser.parseType(elementType))
    PARSE_TENSOR_ERROR("element-type")

  FailureOr<DenseElementsAttr> cacheData;
  Attribute encoding;
  StringAttr name;
  ArrayAttr stride;
  StringAttr layout;
  MemTypeAttr memType;
  IntegerAttr address;
  TilingAttr device;
  TilingAttr onchip;
  while (!odsParser.parseOptionalComma()) {
    if (!odsParser.parseOptionalKeyword(kTensorEncoding)) {
      if (odsParser.parseEqual())
        PARSE_TENSOR_ERROR(kTensorEncoding)
      if (odsParser.parseLess())
        PARSE_TENSOR_ERROR("Less")
      if (odsParser.parseAttribute(encoding))
        PARSE_TENSOR_ERROR(kTensorEncoding)
      if (odsParser.parseGreater())
        PARSE_TENSOR_ERROR("Greater")
    }
    if (!odsParser.parseOptionalKeyword(kTensorName)) {
      if (odsParser.parseEqual())
        PARSE_TENSOR_ERROR(kTensorName)
      if (odsParser.parseAttribute(name))
        PARSE_TENSOR_ERROR(kTensorName)
    }
    if (!odsParser.parseOptionalKeyword(kTensorStride)) {
      if (odsParser.parseEqual())
        PARSE_TENSOR_ERROR(kTensorStride)
      if (odsParser.parseAttribute(stride))
        PARSE_TENSOR_ERROR(kTensorStride)
    }
    if (!odsParser.parseOptionalKeyword(kTensorLayout)) {
      if (odsParser.parseEqual())
        PARSE_TENSOR_ERROR(kTensorLayout)
      if (odsParser.parseAttribute(layout))
        PARSE_TENSOR_ERROR(kTensorLayout)
    }
    if (!odsParser.parseOptionalKeyword(kTensorMemType)) {
      if (odsParser.parseEqual())
        PARSE_TENSOR_ERROR(kTensorMemType)
      if (odsParser.parseAttribute(memType))
        PARSE_TENSOR_ERROR(kTensorMemType)
    }
    if (!odsParser.parseOptionalKeyword(kTensorAddress)) {
      if (odsParser.parseEqual())
        PARSE_TENSOR_ERROR(kTensorAddress)
      if (odsParser.parseAttribute(address))
        PARSE_TENSOR_ERROR(kTensorAddress)
    }
    if (!odsParser.parseOptionalKeyword(kTensorDeviceParl)) {
      if (odsParser.parseEqual())
        PARSE_TENSOR_ERROR(kTensorDeviceParl)
      if (odsParser.parseAttribute(device))
        PARSE_TENSOR_ERROR(kTensorDeviceParl)
    }
    if (!odsParser.parseOptionalKeyword(kTensorOnchipParl)) {
      if (odsParser.parseEqual())
        PARSE_TENSOR_ERROR(kTensorOnchipParl)
      if (odsParser.parseAttribute(onchip))
        PARSE_TENSOR_ERROR(kTensorOnchipParl)
    }
    if (!odsParser.parseOptionalKeyword(kTensorData)) {
      if (odsParser.parseEqual())
        PARSE_TENSOR_ERROR(kTensorData)
      cacheData = FieldParser<DenseElementsAttr>::parse(odsParser);
      if (failed(cacheData))
        PARSE_TENSOR_ERROR(kTensorData)
    }
  }
  
  if (odsParser.parseGreater())
    PARSE_TENSOR_ERROR("Greater")
  auto tensor = TensorType::get(odsParser.getContext(), 
    shape, elementType, name, encoding, stride, layout, memType, 
    address, device, onchip, cacheData.value_or(DenseElementsAttr()));
  return tensor;
}

void TensorType::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  for (auto dim : getShape()) {
    if (dim == mlir::ShapedType::kDynamic)
      odsPrinter << "?x";
    else
      odsPrinter << dim << "x";
  }
  odsPrinter << getElementType();

  if (getEncoding()) {
    odsPrinter << ", " << kTensorEncoding << " = <" << getEncoding() << ">";
  }
  if (getName()) {
    odsPrinter << ", " << kTensorName << " = " << getName();
  }
  if (getStride()) {
    odsPrinter << ", " << kTensorStride << " = " << getStride();
  }
  if (getLayout()) {
    odsPrinter << ", " << kTensorLayout << " = " << getLayout();
  }
  if (getMemType()) {
    odsPrinter << ", " << kTensorMemType << " = " << getMemType();
  }
  if (getAddress()) {
    odsPrinter << ", " << kTensorAddress << " = " << getAddress();
  }
  if (getDeviceParallel()) {
    odsPrinter << ", " << kTensorDeviceParl << " = " << getDeviceParallel();
  }
  if (getOnchipParallel()) {
    odsPrinter << ", " << kTensorOnchipParl << " = " << getOnchipParallel();
  }
  if (getCacheData()) {
    odsPrinter << ", " << kTensorData << " = ";
    odsPrinter.printStrippedAttrOrType(getCacheData());
  }
  odsPrinter << ">";
}

TensorType getAtirTensorType(MLIRContext* ctx) {
  llvm::SmallVector<int64_t> shape;
  Type type = Float32Type::get(ctx);
  return TensorType::get(shape, type);
}

void ForOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState, int64_t idxCnt,
                  ::mlir::ValueRange iterArgs, ::mlir::IntegerAttr axis,
                  ::mlir::StringAttr postOp, ::mlir::BoolAttr parallel) {
  odsState.addAttribute(odsBuilder.getStringAttr("idxCnt"),
                        odsBuilder.getI64IntegerAttr(idxCnt));
  odsState.addOperands(iterArgs);
  for (size_t i = idxCnt; i < iterArgs.size(); i++)
    odsState.addTypes(iterArgs[i].getType());
  if (axis) {
    odsState.addAttribute(odsBuilder.getStringAttr("axis"), axis);
  }
  if (postOp) {
    odsState.addAttribute(odsBuilder.getStringAttr("postOp"), postOp);
  }
  if (parallel) {
    odsState.addAttribute(odsBuilder.getStringAttr("parallel"), parallel);
  }

  Region *bodyRegion = odsState.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  for (auto i = 0; i < idxCnt; i++) {
    bodyBlock.addArgument(odsBuilder.getIndexType(), odsState.location);
  }
  // for (Value v : iterArgs)
  //   bodyBlock.addArgument(v.getType(), v.getLoc());
}
void ForOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState, int64_t start, int64_t end,
                  int64_t step, ::mlir::ValueRange iterArgs,
                  ::mlir::IntegerAttr axis, ::mlir::StringAttr postOp,
                  ::mlir::BoolAttr parallel) {
  odsState.addAttributes({
      NamedAttribute(odsBuilder.getStringAttr("start"),
                     odsBuilder.getI64IntegerAttr(start)),
      NamedAttribute(odsBuilder.getStringAttr("end"),
                     odsBuilder.getI64IntegerAttr(end)),
      NamedAttribute(odsBuilder.getStringAttr("step"),
                     odsBuilder.getI64IntegerAttr(step)),
  });
  build(odsBuilder, odsState, 0, iterArgs, axis, postOp, parallel);
}

void printInitializationList(OpAsmPrinter &p,
                                    Block::BlockArgListType blocksArgs,
                                    ValueRange initializers,
                                    StringRef prefix = "") {
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");
  if (initializers.empty())
    return;

  p << prefix << '(';
  llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
    p << std::get<0>(it) << " = " << std::get<1>(it);
  });
  p << ")";
}

void ForOp::print(OpAsmPrinter &p) {
  if (getParallel().value_or(false)) {
    p << " (parallel)";
  }
  if (getStart().has_value()) {
    p << " " << getInductionVars()[0] << " = " << *getStart() << " to "
      << *getEnd() << " step " << *getStep();
    if (!getOperands().empty()) {
      printInitializationList(p, getInductionVars().drop_front(), getOperands(), " iter_args");
    }
  } else {
    if (!getOperands().empty()) {
      printInitializationList(p, getInductionVars(), getOperands(), " iter_args");
    }
  }

  if (!getOperands().empty())
    p << " -> (" << getOperands().getTypes() << ')';
  p << ' ';
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  BoolAttr parallel;
  if (succeeded(parser.parseOptionalLParen())) {
    if (succeeded(parser.parseOptionalKeyword("parallel"))) {
      parallel = builder.getBoolAttr(true);
      if (parser.parseRParen())
        return failure();
    }
  }
  OpAsmParser::Argument inductionVariable;
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  inductionVariable.type = builder.getIndexType();
  IntegerAttr start, end, step;

  // Parse the induction variable followed by '='.
  if (parser.parseOptionalArgument(inductionVariable).has_value()) {
    if (parser.parseEqual() || parser.parseAttribute(start) ||
        parser.parseKeyword("to") || parser.parseAttribute(end) ||
        parser.parseKeyword("step") || parser.parseAttribute(step))
      return failure();
    regionArgs.push_back(inductionVariable);
    printf("TODO: need check parse for op with start end step\n");
  }

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
    if (regionArgs.size() != result.types.size())
      return parser.emitError(
          parser.getNameLoc(),
          "mismatch in number of loop-carried values and defined values");

    // Resolve input operands.
    for (auto argOperandType : llvm::zip(regionArgs, operands, result.types)) {
      Type type = std::get<2>(argOperandType);
      std::get<0>(argOperandType).type = builder.getIndexType();
      if (parser.resolveOperand(std::get<1>(argOperandType), type,
                                result.operands))
        return failure();
    }
  }

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  // TMP
  result.types.clear();
  return success();
}

/// Parse the case regions and values.
ParseResult parseSwitchCases(OpAsmParser &p, DenseI64ArrayAttr &cases,
                 SmallVectorImpl<std::unique_ptr<Region>> &caseRegions) {
  SmallVector<int64_t> caseValues;
  while (succeeded(p.parseOptionalKeyword("case"))) {
    int64_t value;
    Region &region = *caseRegions.emplace_back(std::make_unique<Region>());
    if (p.parseInteger(value) || p.parseRegion(region, /*arguments=*/{}))
      return failure();
    caseValues.push_back(value);
  }
  cases = p.getBuilder().getDenseI64ArrayAttr(caseValues);
  return success();
}

/// Print the case regions and values.
void printSwitchCases(OpAsmPrinter &p, Operation *op,
                             DenseI64ArrayAttr cases, RegionRange caseRegions) {
  for (auto [value, region] : llvm::zip(cases.asArrayRef(), caseRegions)) {
    p.printNewline();
    p << "case " << value << ' ';
    p.printRegion(*region, /*printEntryBlockArgs=*/false);
  }
}

unsigned SwitchCaseOp::getNumCases() { return getCases().size(); }

Block &SwitchCaseOp::getDefaultBlock() {
  return getDefaultRegion().front();
}

Block &SwitchCaseOp::getCaseBlock(unsigned idx) {
  assert(idx < getNumCases() && "case index out-of-bounds");
  return getCaseRegions()[idx].front();
}

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

OpFoldResult IdentityOp::fold(FoldAdaptor adaptor) {
  return getInput();
}

//===----------------------------------------------------------------------===//
// ShapeOp
//===----------------------------------------------------------------------===//

LogicalResult ShapeOp::verify() {
  auto inputType = llvm::dyn_cast<TensorType>(getInput().getType());
  auto outputType = llvm::dyn_cast<TensorType>(getOutput().getType());
  // 1D
  if (outputType.getShape().size() != 1) {
    return emitOpError() << "output must be 1D tensor, got rank "
                         << outputType.getShape().size();
  }
  // 
  Type elementType = outputType.getElementType();
  if (!elementType.isInteger(32) && !elementType.isInteger(64)) {
    return emitOpError() << "output element type must be i32 or i64, got "
                         << elementType;
  }
  // 
  int64_t inputRank = inputType.getShape().size();
  int64_t outputDim = outputType.getShape()[0];
  if (inputRank >= 0 && outputDim >= 0 && outputDim != inputRank) {
    return emitOpError() << "output dimension (" << outputDim
                         << ") must match input rank (" << inputRank << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SizeOp
//===----------------------------------------------------------------------===//

LogicalResult SizeOp::verify() {
  // 
  auto outputType = llvm::dyn_cast<TensorType>(getOutput().getType());
  auto outputShape = outputType.getShape();
  if (outputShape.size() != 0) {
    return emitOpError("output must be a scalar tensor");
  }
  auto elementType = outputType.getElementType();
  if (!elementType.isIntOrIndex()) {
    return emitOpError("output element type must be integer or index");
  }
  return success();
}

OpFoldResult SizeOp::fold(FoldAdaptor adaptor) {
  auto inputType = llvm::dyn_cast<TensorType>(getInput().getType());
  auto shape = inputType.getShape();
  int64_t numElements = 1;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      return {};
    }
    numElements *= dim;
  }
  auto outputElementType = llvm::dyn_cast<TensorType>(getOutput().getType()).getElementType();
  if (auto intType = llvm::dyn_cast<IntegerType>(outputElementType)) {
    return IntegerAttr::get(intType, numElements);
  } else if (outputElementType.isIndex()) {
    return IntegerAttr::get(IndexType::get(getContext()), numElements);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

LogicalResult RangeOp::verify(){
  auto input1Type = llvm::dyn_cast<TensorType>(getStart().getType());
  auto input1Shape = input1Type.getShape();
  auto input2Type = llvm::dyn_cast<TensorType>(getLimit().getType());
  auto input2Shape = input2Type.getShape();
  auto input3Type = llvm::dyn_cast<TensorType>(getDelta().getType());
  auto input3Shape = input3Type.getShape();
  auto outputType = llvm::dyn_cast<TensorType>(getResult().getType());
  auto outputShape = outputType.getShape();
  if (input1Shape.size() != 0 || input2Shape.size() != 0 || input3Shape.size() != 0) {
    return emitOpError("input parameter must be a scalar");
  }
  if (outputShape.size() != 1) {
    return emitOpError("output must be a one-dimensional tensor");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ParallelDynamicStitchOp
//===----------------------------------------------------------------------===//

LogicalResult ParallelDynamicStitchOp::verify() {
  if (getIndices().size() != getData().size()) {
    return emitOpError("number of index tensors must equal number of data tensors");
  }
  return success();
} 

//===----------------------------------------------------------------------===//
// FillOp
//===----------------------------------------------------------------------===//

LogicalResult FillOp::verify() {
  auto input1Type = llvm::dyn_cast<TensorType>(getShapeInput().getType());
  auto input1Shape = input1Type.getShape();
  auto input2Type = llvm::dyn_cast<TensorType>(getValueInput().getType());
  auto input2Shape = input2Type.getShape();
  if (input1Shape.size() != 1) {
    return emitOpError("shapeInput must be a one-dimensional tensor");
  }
  if (input2Shape.size() != 0) {
    return emitOpError("valueInput must be a scalar");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BatchMatMulOp
//===----------------------------------------------------------------------===//

LogicalResult BatchMatMulOp::verify() {
  auto AType = llvm::dyn_cast<TensorType>(getA().getType());
  auto BType = llvm::dyn_cast<TensorType>(getB().getType());
  auto outputType = llvm::dyn_cast<TensorType>(getOutput().getType());
  
  auto aElemType = AType.getElementType();
  auto bElemType = BType.getElementType();
  auto outElemType = outputType.getElementType();
  
  if (aElemType != bElemType || aElemType != outElemType) {
    return emitOpError("all tensors must have the same element type");
  }
  if (!aElemType.isIntOrFloat()) {
    return emitOpError("only integer and floating point types are supported");
  }
  auto aShape = AType.getShape();
  auto bShape = BType.getShape();
  auto outShape = outputType.getShape();
  
  if (aShape.size() < 2 || bShape.size() < 2) {
    return emitOpError("inputs must have at least 2 dimensions");
  }
  
  int64_t aRows = aShape[aShape.size() - 2];
  int64_t aCols = aShape[aShape.size() - 1];
  int64_t bRows = bShape[bShape.size() - 2];
  int64_t bCols = bShape[bShape.size() - 1];
  
  if (getTransposeA()) std::swap(aRows, aCols);
  if (getTransposeB()) std::swap(bRows, bCols);
  
  if (aCols != bRows) {
    return emitOpError("incompatible inner dimensions for matrix multiplication");
  }
  
  if (outShape.size() != std::max(aShape.size(), bShape.size())) {
    return emitOpError("output rank mismatch");
  }
  
  if (outShape[outShape.size() - 2] != aRows || 
      outShape[outShape.size() - 1] != bCols) {
    return emitOpError("output matrix dimensions incorrect");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

LogicalResult TopKOp::verify() {
    auto inputType = llvm::dyn_cast<TensorType>(getInput().getType());
    auto kType = llvm::dyn_cast<TensorType>(getK().getType());
    auto valuesType = llvm::dyn_cast<TensorType>(getValues().getType());
    auto indicesType = llvm::dyn_cast<TensorType>(getIndices().getType());
    auto kElemType = kType.getElementType();
    if (!kElemType.isInteger()) {
      return emitOpError("k must have integer type, got ")
             << kElemType;
    }
    
    auto kShape = kType.getShape();
    if (!kShape.empty() && kShape.size() != 0) {
      return emitOpError("k must be a scalar");
    }
  
    auto inputShape = inputType.getShape();
    if (inputShape.empty()) {
      return emitOpError("input must have at least 1 dimension");
    }
  
    auto valuesShape = valuesType.getShape();
    auto indicesShape = indicesType.getShape();
    
    if (valuesShape != indicesShape) {
      return emitOpError("values and indices must have the same shape");
    }

    return success();
  }

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

LogicalResult PadOp::verify() {
  auto paddingsType = llvm::dyn_cast<TensorType>(getPaddings().getType());
  auto paddingsShape = paddingsType.getShape();
  if (paddingsShape.size() != 2) {
    return emitOpError("paddings must be a 2D tensor, got shape: ")
           << paddingsShape;
  }
  
  auto inputType = llvm::dyn_cast<TensorType>(getInput().getType());
  unsigned inputRank = inputType.getShape().size();
  if (paddingsShape[0] != inputRank) {
    return emitOpError("paddings first dimension must equal input rank (")
           << inputRank << "), got " << paddingsShape[0];
  }
  if (paddingsShape[1] != 2) {
    return emitOpError("paddings second dimension must be 2, got ")
           << paddingsShape[1];
  }

  auto valueType = llvm::dyn_cast<TensorType>(getValue().getType());
  if (valueType.getShape().size() != 0) {
    return emitOpError("value must be a scalar tensor");
  }
  if (valueType.getElementType() != inputType.getElementType()) {
    return emitOpError("value element type must match input element type");
  }
  
  auto outputType = llvm::dyn_cast<TensorType>(getOutput().getType());
  auto outputShape = outputType.getShape();
  if (outputShape.size() != inputRank) {
    return emitOpError("output rank must equal input rank, got ")
            << outputShape.size() << " vs " << inputRank;
  }
  return success();
}

} // namespace atir