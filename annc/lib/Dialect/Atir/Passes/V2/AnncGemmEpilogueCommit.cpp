#include <optional>
#include <utility>

#include "Dialect/Atir/Passes/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace atir {
namespace {

enum class EpilogueOpcode { Add, Mul, Relu };
enum class BroadcastKind { Scalar, N, Matrix };

struct Step {
  EpilogueOpcode opcode;
  unsigned inputIndex;
  BroadcastKind broadcast;
  std::optional<float> limit;
  Operation *source;
  Value result;
};

struct BroadcastValue {
  Value source;
  BroadcastKind kind;
  SmallVector<Operation *> wrappers;
};

llvm::StringRef opcodeName(EpilogueOpcode opcode) {
  switch (opcode) {
    case EpilogueOpcode::Add:
      return "add";
    case EpilogueOpcode::Mul:
      return "mul";
    case EpilogueOpcode::Relu:
      return "relu";
  }
  llvm_unreachable("unknown epilogue opcode");
}

llvm::StringRef broadcastName(BroadcastKind kind) {
  switch (kind) {
    case BroadcastKind::Scalar:
      return "scalar";
    case BroadcastKind::N:
      return "n";
    case BroadcastKind::Matrix:
      return "matrix";
  }
  llvm_unreachable("unknown broadcast kind");
}

std::optional<BroadcastValue> unwrapBroadcast(Value value,
                                              RankedTensorType outputType) {
  SmallVector<Operation *> wrappers;
  auto outputShape = outputType.getShape();
  while (auto broadcast = value.getDefiningOp<stablehlo::BroadcastInDimOp>()) {
    auto inputType =
        dyn_cast<RankedTensorType>(broadcast.getOperand().getType());
    if (!inputType || broadcast.getType().getShape() != outputShape)
      return std::nullopt;
    SmallVector<int64_t> dims(broadcast.getBroadcastDimensions());
    if (inputType.getRank() == outputType.getRank()) {
      if (dims.size() != static_cast<size_t>(inputType.getRank()))
        return std::nullopt;
      bool identity = true;
      for (int64_t i = 0; i < inputType.getRank(); ++i)
        identity &= dims[i] == i;
      if (!identity) return std::nullopt;
      wrappers.push_back(broadcast);
      value = broadcast.getOperand();
      continue;
    }
    BroadcastKind kind;
    if (inputType.getRank() == 0) {
      kind = BroadcastKind::Scalar;
    } else if (inputType.getRank() == 1 && dims.size() == 1 && dims[0] == 1) {
      kind = BroadcastKind::N;
    } else if (inputType.getShape() == outputShape &&
               inputType.getRank() == 2 && dims == SmallVector<int64_t>{0, 1}) {
      kind = BroadcastKind::Matrix;
    } else {
      return std::nullopt;
    }
    wrappers.push_back(broadcast);
    return BroadcastValue{broadcast.getOperand(), kind, std::move(wrappers)};
  }

  auto inputType = dyn_cast<RankedTensorType>(value.getType());
  if (!inputType || inputType.getShape() != outputShape) return std::nullopt;
  return BroadcastValue{value, BroadcastKind::Matrix, std::move(wrappers)};
}

bool isZeroScalar(Value value, SmallVectorImpl<Operation *> &wrappers) {
  SmallVector<Operation *> candidates;
  while (auto broadcast = value.getDefiningOp<stablehlo::BroadcastInDimOp>()) {
    candidates.push_back(broadcast);
    value = broadcast.getOperand();
  }
  auto constant = value.getDefiningOp<stablehlo::ConstantOp>();
  if (!constant) return false;
  auto elements = dyn_cast<DenseFPElementsAttr>(constant.getValue());
  if (!elements || !elements.isSplat()) return false;
  if (!elements.getSplatValue<APFloat>().isZero()) return false;
  candidates.push_back(constant);
  wrappers.append(candidates.begin(), candidates.end());
  return true;
}

DictionaryAttr buildEpilogueAttr(Builder &builder, ArrayRef<Step> steps) {
  SmallVector<Attribute> encodedSteps;
  for (const Step &step : steps) {
    NamedAttrList encoded;
    encoded.append("opcode", builder.getStringAttr(opcodeName(step.opcode)));
    encoded.append("input", builder.getI64IntegerAttr(step.inputIndex));
    encoded.append("broadcast",
                   builder.getStringAttr(broadcastName(step.broadcast)));
    if (step.limit)
      encoded.append("limit", builder.getF32FloatAttr(*step.limit));
    encodedSteps.push_back(encoded.getDictionary(builder.getContext()));
  }
  NamedAttrList plan;
  plan.append("version", builder.getI64IntegerAttr(1));
  plan.append("output_role", builder.getStringAttr("main_d"));
  plan.append("terminal_view", builder.getStringAttr("identity"));
  plan.append("steps", builder.getArrayAttr(encodedSteps));
  return plan.getDictionary(builder.getContext());
}

std::optional<Value> unwrapResultBroadcasts(
    Value value, RankedTensorType outputType,
    SmallVectorImpl<Operation *> &wrappers) {
  while (value.hasOneUse()) {
    Operation *user = *value.getUsers().begin();
    auto broadcast = dyn_cast<stablehlo::BroadcastInDimOp>(user);
    if (!broadcast) break;
    auto inputType =
        dyn_cast<RankedTensorType>(broadcast.getOperand().getType());
    if (!inputType || inputType.getShape() != outputType.getShape() ||
        broadcast.getType().getShape() != outputType.getShape())
      break;
    SmallVector<int64_t> dims(broadcast.getBroadcastDimensions());
    if (dims.size() != static_cast<size_t>(inputType.getRank())) break;
    bool identity = true;
    for (int64_t i = 0; i < inputType.getRank(); ++i) identity &= dims[i] == i;
    if (!identity) break;
    wrappers.push_back(broadcast);
    value = broadcast.getResult();
  }
  return value;
}

LogicalResult commit(stablehlo::DotOp dot) {
  auto outputType = dyn_cast<RankedTensorType>(dot.getType());
  if (!outputType || outputType.getRank() != 2 ||
      !outputType.getElementType().isF32())
    return success();

  SmallVector<Operation *> chain;
  SmallVector<Operation *> wrappers;
  SmallVector<Step> steps;
  Value current = dot.getResult();
  while (true) {
    auto carriedValue = unwrapResultBroadcasts(current, outputType, wrappers);
    if (!carriedValue) break;
    Value carried = *carriedValue;
    Operation *user =
        carried.hasOneUse() ? *carried.getUsers().begin() : nullptr;
    if (!user) break;

    if (auto add = dyn_cast<stablehlo::AddOp>(user)) {
      Value lhs = add.getLhs(), rhs = add.getRhs();
      bool currentIsLhs = lhs == carried;
      bool currentIsRhs = rhs == carried;
      if (currentIsLhs == currentIsRhs) break;
      Value external = currentIsLhs ? rhs : lhs;
      auto externalValue = unwrapBroadcast(external, outputType);
      if (!externalValue) break;
      wrappers.append(externalValue->wrappers.begin(),
                      externalValue->wrappers.end());
      steps.push_back({EpilogueOpcode::Add, currentIsLhs ? 1u : 0u,
                       externalValue->kind, std::nullopt, user,
                       add.getResult()});
      chain.push_back(user);
      current = add.getResult();
    } else if (auto mul = dyn_cast<stablehlo::MulOp>(user)) {
      Value lhs = mul.getLhs(), rhs = mul.getRhs();
      bool currentIsLhs = lhs == carried;
      bool currentIsRhs = rhs == carried;
      if (currentIsLhs == currentIsRhs) break;
      Value external = currentIsLhs ? rhs : lhs;
      auto externalValue = unwrapBroadcast(external, outputType);
      if (!externalValue) break;
      wrappers.append(externalValue->wrappers.begin(),
                      externalValue->wrappers.end());
      steps.push_back({EpilogueOpcode::Mul, currentIsLhs ? 1u : 0u,
                       externalValue->kind, std::nullopt, user,
                       mul.getResult()});
      chain.push_back(user);
      current = mul.getResult();
    } else if (auto maximum = dyn_cast<stablehlo::MaxOp>(user)) {
      Value lhs = maximum.getLhs(), rhs = maximum.getRhs();
      Value operand;
      if (lhs == carried && isZeroScalar(rhs, wrappers))
        operand = lhs;
      else if (rhs == carried && isZeroScalar(lhs, wrappers))
        operand = rhs;
      else
        break;
      (void)operand;
      steps.push_back({EpilogueOpcode::Relu, 0, BroadcastKind::Scalar,
                       std::nullopt, user, maximum.getResult()});
      chain.push_back(user);
      current = maximum.getResult();
    } else {
      break;
    }
  }

  // A bare dot (no epilogue chain) commits too: stablehlo.dot has no V2
  // lowering, so leaving it in place would starve GEMM extraction downstream.
  Value output = steps.empty() ? dot.getResult() : steps.back().result;
  if (!outputType || !llvm::isa<RankedTensorType>(output.getType()))
    return success();

  SmallVector<Value> inputs;
  for (const Step &step : steps) {
    if (step.opcode == EpilogueOpcode::Relu) continue;
    Value external = step.source->getOperand(step.inputIndex);
    auto unwrapped = unwrapBroadcast(external, outputType);
    if (!unwrapped) return failure();
    inputs.push_back(unwrapped->source);
  }

  OpBuilder builder(dot->getContext());
  if (steps.empty())
    builder.setInsertionPoint(dot);
  else
    builder.setInsertionPointAfter(steps.back().source);
  OperationState state(dot.getLoc(), "atir.fused_matmul");
  state.addOperands({dot.getLhs(), dot.getRhs()});
  state.addOperands(inputs);
  state.addTypes(output.getType());
  for (NamedAttribute attr : dot->getAttrs())
    if (attr.getName() != "annc.source_node" &&
        attr.getName() != "annc.source_op" &&
        attr.getName() != "annc.source_ports")
      state.addAttribute(attr.getName(), attr.getValue());
  if (!steps.empty())
    state.addAttribute("annc.gemm.epilogue", buildEpilogueAttr(builder, steps));
  for (StringRef name :
       {"annc.source_node", "annc.source_op", "annc.source_ports"})
    if (Attribute attr = dot->getAttr(name)) state.addAttribute(name, attr);
  Operation *fused = builder.create(state);
  output.replaceAllUsesWith(fused->getResult(0));

  llvm::SmallPtrSet<Operation *, 16> liveChain(chain.begin(), chain.end());
  llvm::SmallPtrSet<Operation *, 16> liveWrappers;
  for (Operation *wrapper : wrappers) liveWrappers.insert(wrapper);
  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> pending(liveChain.begin(), liveChain.end());
    for (Operation *op : pending) {
      if (op->use_empty()) {
        liveChain.erase(op);
        op->erase();
        changed = true;
      }
    }
    pending.assign(liveWrappers.begin(), liveWrappers.end());
    for (Operation *op : pending) {
      if (op->use_empty()) {
        liveWrappers.erase(op);
        op->erase();
        changed = true;
      }
    }
    if (dot && dot->use_empty()) {
      dot.erase();
      dot = {};
      changed = true;
    }
  } while (changed);
  return success();
}

class AnncGemmEpilogueCommitPass
    : public AnncGemmEpilogueCommitBase<AnncGemmEpilogueCommitPass> {
 public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);
    registry.insert<stablehlo::StablehloDialect>();
  }

  void runOnOperation() override {
    SmallVector<stablehlo::DotOp> dots;
    getOperation().walk([&](stablehlo::DotOp dot) { dots.push_back(dot); });
    for (stablehlo::DotOp dot : dots)
      if (failed(commit(dot))) {
        signalPassFailure();
        return;
      }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAnncGemmEpilogueCommitPass() {
  return std::make_unique<AnncGemmEpilogueCommitPass>();
}

}  // namespace atir
