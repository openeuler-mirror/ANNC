#include "Dialect/Atir/Passes/GemmEpilogueCandidate.h"

#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace atir {
namespace {
constexpr llvm::StringLiteral kOpcodeNames[] = {"add", "mul", "relu",
                                                "sigmoid"};
constexpr llvm::StringLiteral kBroadcastNames[] = {"scalar", "m", "n",
                                                   "matrix"};
constexpr llvm::StringLiteral kViewNames[] = {"identity",
                                              "insert_unit_dimension"};
template <typename Enum, size_t Size>
llvm::StringRef enumName(Enum value, const llvm::StringLiteral (&names)[Size]) {
  size_t index = static_cast<size_t>(value);
  return index < Size ? llvm::StringRef(names[index])
                      : llvm::StringRef("unknown");
}

template <typename Enum, size_t Size>
std::optional<Enum> parseEnum(llvm::StringRef value,
                              const llvm::StringLiteral (&names)[Size]) {
  for (size_t index = 0; index < Size; ++index)
    if (value == names[index]) return static_cast<Enum>(index);
  return std::nullopt;
}

bool compatibleTensorTypes(Type lhs, Type rhs) {
  auto lhsType = llvm::dyn_cast<TensorType>(lhs);
  auto rhsType = llvm::dyn_cast<TensorType>(rhs);
  return lhsType && rhsType &&
         lhsType.getElementType() == rhsType.getElementType() &&
         lhsType.getShape() == rhsType.getShape();
}

std::optional<BroadcastKind> classifyEpilogueBroadcast(MatMulOp matmul,
                                                       Value value) {
  TensorType output = llvm::dyn_cast<TensorType>(matmul.getResult().getType());
  TensorType input = llvm::dyn_cast<TensorType>(value.getType());
  if (!output || !input || output.getElementType() != input.getElementType())
    return std::nullopt;
  return classifyEpilogueBroadcastShape(output.getShape(), input.getShape());
}

mlir::FailureOr<EpilogueStep> makeStep(EpilogueOpcode opcode,
                                       int64_t inputIndex,
                                       BroadcastKind broadcast,
                                       std::optional<float> reluLimit) {
  const bool unary =
      opcode == EpilogueOpcode::kRelu || opcode == EpilogueOpcode::kSigmoid;
  if (inputIndex < 0 || (unary && inputIndex != 0) ||
      (!unary && inputIndex > 1) ||
      (unary && broadcast != BroadcastKind::kScalar) ||
      (opcode == EpilogueOpcode::kRelu) != reluLimit.has_value())
    return mlir::failure();
  return EpilogueStep{opcode, static_cast<unsigned>(inputIndex), broadcast,
                      reluLimit};
}

bool setBinaryStep(MatMulOp matmul, Value current, Operation *operation,
                   EpilogueOpcode opcode, EpilogueStep &step) {
  Value lhs = operation->getOperand(1);
  Value rhs = operation->getOperand(2);
  Value external;
  if (lhs == current)
    external = rhs;
  else if (rhs == current)
    external = lhs;
  else
    return false;
  auto broadcast = classifyEpilogueBroadcast(matmul, external);
  if (!broadcast) return false;
  const unsigned externalInput = lhs == current ? 1 : 0;
  step = {opcode,       externalInput, *broadcast,
          std::nullopt, operation,     operation->getResult(0)};
  return true;
}

bool isSupportedTerminalEpilogueView(Value input, Value output) {
  TensorType inputType = llvm::dyn_cast<TensorType>(input.getType());
  TensorType outputType = llvm::dyn_cast<TensorType>(output.getType());
  return inputType && outputType && inputType.getShape().size() == 2 &&
         outputType.getShape().size() == 3 &&
         !ShapedType::isDynamicShape(inputType.getShape()) &&
         !ShapedType::isDynamicShape(outputType.getShape()) &&
         outputType.getShape()[0] == inputType.getShape()[0] &&
         outputType.getShape()[1] == 1 &&
         outputType.getShape()[2] == inputType.getShape()[1];
}

llvm::StringRef stringify(EpilogueOpcode opcode) {
  return enumName(opcode, kOpcodeNames);
}

llvm::StringRef stringify(BroadcastKind broadcast) {
  return enumName(broadcast, kBroadcastNames);
}

llvm::StringRef stringify(OutputView view) {
  return enumName(view, kViewNames);
}

}  // namespace

std::optional<BroadcastKind> classifyEpilogueBroadcastShape(
    llvm::ArrayRef<int64_t> outputShape, llvm::ArrayRef<int64_t> inputShape) {
  if (inputShape.empty() || (inputShape.size() == 1 && inputShape[0] == 1))
    return BroadcastKind::kScalar;
  if (outputShape.size() != 2 || ShapedType::isDynamicShape(inputShape))
    return std::nullopt;
  const int64_t m = outputShape[0], n = outputShape[1];
  if (inputShape.size() == 1)
    return !ShapedType::isDynamic(n) && inputShape[0] == n
               ? std::optional(BroadcastKind::kN)
               : std::nullopt;
  if (inputShape.size() != 2) return std::nullopt;
  const int64_t rows = inputShape[0], columns = inputShape[1];
  if (rows == 1 && columns == 1) return BroadcastKind::kScalar;
  if (!ShapedType::isDynamic(m) && !ShapedType::isDynamic(n) && rows == m &&
      columns == n)
    return BroadcastKind::kMatrix;
  if (!ShapedType::isDynamic(m) && rows == m && columns == 1)
    return BroadcastKind::kM;
  if (!ShapedType::isDynamic(n) && rows == 1 && columns == n)
    return BroadcastKind::kN;
  return std::nullopt;
}

mlir::FailureOr<EpilogueProgram> parseEpilogueCandidatePlan(
    mlir::DictionaryAttr candidate) {
  if (!candidate) return mlir::failure();
  auto version = candidate.getAs<mlir::IntegerAttr>("version");
  auto role = candidate.getAs<mlir::StringAttr>("output_role");
  auto view = candidate.getAs<mlir::StringAttr>("terminal_view");
  auto steps = candidate.getAs<mlir::ArrayAttr>("steps");
  if (!version || version.getInt() != kGemmEpilogueCandidateVersion || !role ||
      role.getValue() != "main_d" || !view || !steps || steps.empty())
    return mlir::failure();
  auto parsedView = parseEnum<OutputView>(view.getValue(), kViewNames);
  if (!parsedView) return mlir::failure();
  EpilogueProgram program;
  program.terminalView = *parsedView;
  for (mlir::Attribute attr : steps) {
    auto encoded = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
    if (!encoded) return mlir::failure();
    auto opcodeAttr = encoded.getAs<mlir::StringAttr>("opcode");
    auto input = encoded.getAs<mlir::IntegerAttr>("input");
    auto broadcastAttr = encoded.getAs<mlir::StringAttr>("broadcast");
    if (!opcodeAttr || !input || !broadcastAttr || encoded.get("input2"))
      return mlir::failure();
    auto opcode =
        parseEnum<EpilogueOpcode>(opcodeAttr.getValue(), kOpcodeNames);
    auto broadcast =
        parseEnum<BroadcastKind>(broadcastAttr.getValue(), kBroadcastNames);
    if (!opcode || !broadcast) return mlir::failure();
    std::optional<float> limit;
    if (auto value = encoded.getAs<mlir::FloatAttr>("limit"))
      limit = static_cast<float>(value.getValueAsDouble());
    auto step = makeStep(*opcode, input.getInt(), *broadcast, limit);
    if (mlir::failed(step)) return mlir::failure();
    program.steps.push_back(*step);
  }
  return program;
}

mlir::FailureOr<EpilogueProgram> discoverOrderedEpilogue(MatMulOp matmul) {
  if (!matmul.getEpilogueInputs().empty() || matmul.getDoRelu() ||
      matmul.getWithBias() || matmul.getLeftTranspose() ||
      matmul.getRightTranspose() || matmul.getOutputTranspose() ||
      !llvm::isa<TensorType>(matmul.getResult().getType()) ||
      !compatibleTensorTypes(matmul.getC().getType(),
                             matmul.getResult().getType()))
    return failure();

  EpilogueProgram program;
  Value current = matmul.getResult();
  Operation *operation =
      current.hasOneUse() ? *current.getUsers().begin() : nullptr;
  while (operation) {
    EpilogueStep step;
    bool supported = true;
    if (auto add = llvm::dyn_cast<AddOp>(operation)) {
      if (add.getInputs().size() != 2 || add.getScalar() || add.getDoRelu() ||
          !setBinaryStep(matmul, current, operation, EpilogueOpcode::kAdd,
                         step))
        supported = false;
    } else if (llvm::isa<MulOp>(operation)) {
      if (!setBinaryStep(matmul, current, operation, EpilogueOpcode::kMul,
                         step))
        supported = false;
    } else if (auto relu = llvm::dyn_cast<ReluOp>(operation)) {
      if (relu.getInput() != current) {
        supported = false;
      } else {
        step = {EpilogueOpcode::kRelu,
                0,
                BroadcastKind::kScalar,
                static_cast<float>(relu.getReluLimitAttr().getValueAsDouble()),
                operation,
                relu.getResult()};
      }
    } else if (auto sigmoid = llvm::dyn_cast<LogisticOp>(operation)) {
      if (sigmoid.getInput() != current) {
        supported = false;
      } else {
        step = {EpilogueOpcode::kSigmoid,
                0,
                BroadcastKind::kScalar,
                std::nullopt,
                operation,
                sigmoid.getResult()};
      }
    } else if (auto reshape = llvm::dyn_cast<ReshapeOp>(operation)) {
      if (reshape.getInput() != current ||
          !reshape.getTargetShape().getDefiningOp<ConstantOp>() ||
          !isSupportedTerminalEpilogueView(current, reshape.getResult()))
        break;
      program.terminalView = OutputView::kTerminalInsertUnitDimension;
      program.terminalViewSource = operation;
      current = reshape.getResult();
      break;
    } else if (llvm::isa<func::ReturnOp>(operation)) {
      break;
    } else {
      break;
    }
    if (!supported) break;
    program.steps.push_back(step);
    current = operation->getResult(0);
    operation = current.hasOneUse() ? *current.getUsers().begin() : nullptr;
  }

  auto func = matmul->getParentOfType<func::FuncOp>();
  if (program.steps.empty()) return failure();
  if (!current.use_empty() && !current.hasOneUse())
    return failure();
  if (program.terminalView != OutputView::kIdentity &&
      !current.use_empty() &&
      !llvm::isa<func::ReturnOp>(*current.getUsers().begin()))
    return failure();
  Value finalDestination = program.steps.back().source->getOperand(0);
  if (auto reshape = current.getDefiningOp<ReshapeOp>()) {
    finalDestination = reshape.getOutput();
    if (!compatibleTensorTypes(reshape.getInput().getType(),
                               matmul.getC().getType()))
      return failure();
  }
  auto blockArg = llvm::dyn_cast<BlockArgument>(finalDestination);
  auto buffer = finalDestination.getDefiningOp<BufferOp>();
  const bool validDestination =
      func && ((blockArg && blockArg.getOwner() == &func.getBody().front()) ||
               (buffer && buffer->getParentOfType<func::FuncOp>() == func));
  if (!validDestination)
    return failure();
  program.result = current;
  return program;
}

mlir::LogicalResult bindEpilogueProgramToSource(MatMulOp matmul,
                                                EpilogueProgram &program) {
  auto discovered = discoverOrderedEpilogue(matmul);
  if (failed(discovered) || discovered->terminalView != program.terminalView ||
      discovered->steps.size() != program.steps.size())
    return failure();
  for (auto [encoded, source] : llvm::zip(program.steps, discovered->steps))
    if (encoded.opcode != source.opcode ||
        encoded.inputIndex != source.inputIndex ||
        encoded.broadcast != source.broadcast ||
        encoded.reluLimit != source.reluLimit)
      return failure();
  program = std::move(*discovered);
  return success();
}

mlir::DictionaryAttr buildEpilogueCandidateAttr(
    mlir::Builder &builder, const EpilogueProgram &program) {
  llvm::SmallVector<mlir::Attribute> encodedSteps;
  for (const EpilogueStep &step : program.steps) {
    mlir::NamedAttrList encoded;
    encoded.append("opcode", builder.getStringAttr(stringify(step.opcode)));
    encoded.append("input", builder.getI64IntegerAttr(step.inputIndex));
    encoded.append("broadcast",
                   builder.getStringAttr(stringify(step.broadcast)));
    if (step.reluLimit)
      encoded.append("limit", builder.getF32FloatAttr(*step.reluLimit));
    encodedSteps.push_back(encoded.getDictionary(builder.getContext()));
  }
  mlir::NamedAttrList candidate;
  candidate.append("version",
                   builder.getI64IntegerAttr(kGemmEpilogueCandidateVersion));
  candidate.append("output_role", builder.getStringAttr("main_d"));
  candidate.append("terminal_view",
                   builder.getStringAttr(stringify(program.terminalView)));
  candidate.append("steps", builder.getArrayAttr(encodedSteps));
  return candidate.getDictionary(builder.getContext());
}

std::string staticSymbolSuffix(const EpilogueProgram &program,
                               bool accumulate) {
  std::string suffix = accumulate ? "_acc" : "";
  for (const EpilogueStep &step : program.steps) {
    suffix += "_" + stringify(step.opcode).str();
    if (step.opcode == EpilogueOpcode::kAdd ||
        step.opcode == EpilogueOpcode::kMul) {
      suffix += "_" + stringify(step.broadcast).str() + "_i" +
                llvm::utostr(step.inputIndex);
    } else if (step.reluLimit) {
      llvm::APFloat value(*step.reluLimit);
      suffix += "_l" + llvm::utohexstr(value.bitcastToAPInt().getZExtValue());
    }
  }
  suffix += "_f32";
  return suffix;
}

}  // namespace atir
