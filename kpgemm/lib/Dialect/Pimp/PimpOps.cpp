#include "Dialect/Pimp/PimpOps.h"

using namespace mlir;
using namespace pimp;

#include "Dialect/Pimp/PimpDialect.cpp.inc"
#include "Dialect/Pimp/PimpEnums.cpp.inc"
void PimpDialect::initialize() {
  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "Dialect/Pimp/PimpAttr.cpp.inc"
  >();
  addOperations<
    #define GET_OP_LIST
    #include "Dialect/Pimp/PimpOps.cpp.inc"
  >();
  addTypes<
    #define GET_TYPEDEF_LIST
    #include "Dialect/Pimp/PimpTypes.cpp.inc"
  >();
}

#define GET_ATTRDEF_CLASSES
#include "Dialect/Pimp/PimpAttr.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/Pimp/PimpOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Pimp/PimpTypes.cpp.inc"

namespace pimp {
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

TensorType getPimpTensorType(MLIRContext* ctx) {
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
} // namespace pimp