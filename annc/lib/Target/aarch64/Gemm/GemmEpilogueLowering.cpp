#include <algorithm>

#include "Dialect/Atir/Passes/GemmEpilogueCandidate.h"
#include "GemmMicrokernelSelection.h"
#include "GemmPlan.h"
#include "GemmTilingUtils.h"
#include "Target/aarch64/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace annc {
namespace {

namespace codegen = annc::aarch64::gemm::codegen;
std::vector<codegen::Step> convertEpilogueSteps(
    const atir::EpilogueProgram &program) {
  std::vector<codegen::Step> steps;
  steps.reserve(program.steps.size());
  for (const auto &step : program.steps) {
    codegen::Step converted;
    switch (step.opcode) {
      case atir::EpilogueOpcode::kAdd:
        converted.opcode = codegen::Opcode::Add;
        break;
      case atir::EpilogueOpcode::kMul:
        converted.opcode = codegen::Opcode::Mul;
        break;
      case atir::EpilogueOpcode::kRelu:
        converted.opcode = codegen::Opcode::Relu;
        break;
      case atir::EpilogueOpcode::kSigmoid:
        converted.opcode = codegen::Opcode::Sigmoid;
        break;
    }
    if (atir::isBinaryEpilogueOpcode(step.opcode)) {
      switch (step.broadcast) {
        case atir::BroadcastKind::kScalar:
          converted.broadcast = codegen::Broadcast::Scalar;
          break;
        case atir::BroadcastKind::kN:
          converted.broadcast = codegen::Broadcast::N;
          break;
        case atir::BroadcastKind::kM:
          converted.broadcast = codegen::Broadcast::M;
          break;
        case atir::BroadcastKind::kMatrix:
          converted.broadcast = codegen::Broadcast::Matrix;
          break;
      }
    }
    converted.reluLimit = step.reluLimit;
    steps.push_back(converted);
  }
  return steps;
}

bool hasLegacyMatmulEpilogueAttrs(Operation *op) {
  if (auto attr = op->getAttrOfType<BoolAttr>("withBias"); attr && attr.getValue())
    return true;
  if (auto attr = op->getAttrOfType<BoolAttr>("do_relu"); attr && attr.getValue())
    return true;
  return false;
}

Value f32Constant(OpBuilder &builder, Location loc, float value) {
  return builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(value));
}

Value buildSigmoid(OpBuilder &builder, Location loc, Value input) {
  auto constant = [&](float value) { return f32Constant(builder, loc, value); };
  Value zero = constant(0.0f), one = constant(1.0f);
  Value isNan = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNO,
                                              input, input);
  Value x = builder.create<arith::MaxNumFOp>(loc, input, constant(-88.0f));
  x = builder.create<arith::MinNumFOp>(loc, x, constant(88.0f));
  Value y = builder.create<arith::MulFOp>(
      loc, builder.create<arith::NegFOp>(loc, x), constant(1.4426950216f));
  Value positive =
      builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, y, zero);
  Value bias = builder.create<arith::SelectOp>(loc, positive, constant(0.5f),
                                               constant(-0.5f));
  Value exponent = builder.create<arith::FPToSIOp>(
      loc, builder.getI32Type(), builder.create<arith::AddFOp>(loc, y, bias));
  Value fraction = builder.create<arith::SubFOp>(
      loc, y,
      builder.create<arith::SIToFPOp>(loc, builder.getF32Type(), exponent));
  Value polynomial = f32Constant(builder, loc, 0.0550683066f);
  for (float coefficient : {0.0555041087f, 0.2402265072f, 0.6931471825f})
    polynomial = builder.create<arith::AddFOp>(
        loc, constant(coefficient),
        builder.create<arith::MulFOp>(loc, fraction, polynomial));
  polynomial = builder.create<arith::AddFOp>(
      loc, one, builder.create<arith::MulFOp>(loc, fraction, polynomial));
  Value bits = builder.create<arith::ShLIOp>(
      loc,
      builder.create<arith::AddIOp>(
          loc, exponent, builder.create<arith::ConstantIntOp>(loc, 127, 32)),
      builder.create<arith::ConstantIntOp>(loc, 23, 32));
  Value exp = builder.create<arith::MulFOp>(
      loc, polynomial,
      builder.create<arith::BitcastOp>(loc, builder.getF32Type(), bits));
  Value result = builder.create<arith::DivFOp>(
      loc, one, builder.create<arith::AddFOp>(loc, one, exp));
  Value nan = builder.create<arith::DivFOp>(loc, zero, zero);
  return builder.create<arith::SelectOp>(loc, isNan, nan, result);
}

LogicalResult materializeSuffix(linalg::GenericOp generic,
                                const atir::EpilogueProgram &program,
                                unsigned prefixSize) {
  auto outputType =
      llvm::dyn_cast<MemRefType>(generic.getOutputs()[0].getType());
  if (!outputType || !outputType.getElementType().isF32())
    return generic.emitOpError("requires an f32 epilogue output");
  OpBuilder builder(generic);
  builder.setInsertionPointAfter(generic);
  SmallVector<Value> inputs{generic.getInputs().front()};
  ArrayRef<AffineMap> carrierMaps = generic.getIndexingMapsArray();
  if (carrierMaps.size() != generic.getInputs().size() + 1)
    return generic.emitOpError("has incomplete epilogue indexing maps");
  SmallVector<AffineMap> maps{carrierMaps.front()};
  unsigned binaryIndex = 0;
  for (unsigned index = 0; index < program.steps.size(); ++index) {
    const atir::EpilogueStep &step = program.steps[index];
    if (!atir::isBinaryEpilogueOpcode(step.opcode)) continue;
    if (index >= prefixSize) {
      if (1 + binaryIndex >= generic.getInputs().size() ||
          1 + binaryIndex >= carrierMaps.size() - 1)
        return generic.emitOpError("has incomplete epilogue inputs");
      Value input = generic.getInputs()[1 + binaryIndex];
      inputs.push_back(input);
      maps.push_back(carrierMaps[1 + binaryIndex]);
    }
    ++binaryIndex;
  }
  maps.push_back(carrierMaps.back());
  SmallVector<atir::EpilogueStep> suffix(program.steps.begin() + prefixSize,
                                         program.steps.end());
  const unsigned inputCount = inputs.size();
  builder.create<linalg::GenericOp>(
      generic.getLoc(), TypeRange{}, inputs, generic.getOutputs(), maps,
      SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                       utils::IteratorType::parallel},
      [suffix, inputCount](OpBuilder &nested, Location loc, ValueRange args) {
        Value value = args[inputCount];
        unsigned input = 1;
        for (const atir::EpilogueStep &step : suffix) {
          if (step.opcode == atir::EpilogueOpcode::kAdd)
            value = nested.create<arith::AddFOp>(loc, value, args[input++]);
          else if (step.opcode == atir::EpilogueOpcode::kMul)
            value = nested.create<arith::MulFOp>(loc, value, args[input++]);
          else if (step.opcode == atir::EpilogueOpcode::kRelu) {
            value = nested.create<arith::MaxNumFOp>(
                loc, value, f32Constant(nested, loc, 0.0f));
            if (step.reluLimit && *step.reluLimit >= 0.0f)
              value = nested.create<arith::MinNumFOp>(
                  loc, value, f32Constant(nested, loc, *step.reluLimit));
          } else {
            value = buildSigmoid(nested, loc, value);
          }
        }
        nested.create<linalg::YieldOp>(loc, value);
      });
  return success();
}

func::FuncOp getOrCreateLeafDeclaration(ModuleOp module, StringRef name,
                                        FunctionType type) {
  if (auto declaration = module.lookupSymbol<func::FuncOp>(name))
    return declaration;
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto declaration = builder.create<func::FuncOp>(module.getLoc(), name, type);
  declaration.setPrivate();
  return declaration;
}

FailureOr<Value> materializeCarrierEpilogueArgs(func::CallOp call,
                                                linalg::GenericOp carrier,
                                                ArrayRef<atir::EpilogueStep> prefix,
                                                OpBuilder &builder) {
  unsigned binaryInputs = 0;
  for (const atir::EpilogueStep &step : prefix)
    binaryInputs += atir::isBinaryEpilogueOpcode(step.opcode);
  if (binaryInputs > aarch64::gemm::kEpilogueArgsPointerSlots ||
      carrier.getInputs().size() < binaryInputs + 1)
    return carrier.emitOpError("exceeds the EpilogueArgs external pointer capacity");

  auto function = call->getParentOfType<func::FuncOp>();
  OpBuilder entryBuilder(function.getContext());
  entryBuilder.setInsertionPointToStart(&function.front());
  auto args = entryBuilder.create<memref::AllocaOp>(
      call.getLoc(),
      MemRefType::get({aarch64::gemm::kEpilogueArgsPointerSlots},
                      entryBuilder.getIndexType()));
  unsigned slot = 0;
  for (unsigned index = 0; index < prefix.size(); ++index) {
    if (!atir::isBinaryEpilogueOpcode(prefix[index].opcode)) continue;
    FailureOr<aarch64::gemm::MemRefBaseAndOffset> base =
        aarch64::gemm::getMemRefBaseAndOffset(
            builder, call.getLoc(), carrier.getInputs()[slot + 1]);
    if (failed(base)) return failure();
    Value pointer = builder.create<memref::ExtractAlignedPointerAsIndexOp>(
        call.getLoc(), base->base);
    Value bytes = builder.create<arith::ConstantIndexOp>(call.getLoc(),
                                                         sizeof(float));
    pointer = builder.create<arith::AddIOp>(
        call.getLoc(), pointer,
        builder.create<arith::MulIOp>(call.getLoc(), base->offset, bytes));
    builder.create<memref::StoreOp>(
        call.getLoc(), pointer, args,
        ValueRange{builder.create<arith::ConstantIndexOp>(call.getLoc(),
                                                          slot)});
    ++slot;
  }
  return args.getResult();
}

LogicalResult makeFusedCall(func::CallOp call, linalg::GenericOp carrier,
                            Value epilogueArgs) {
  StringRef fusedName;
  if (call.getCallee() == aarch64::gemm::kMicrokernelLeafName)
    fusedName = aarch64::gemm::kFusedMicrokernelLeafName;
  else if (call.getCallee() == aarch64::gemm::kMicrokernelRmLeafName)
    fusedName = aarch64::gemm::kFusedRowMajorMicrokernelLeafName;
  else
    return call.emitOpError("cannot fuse epilogue into this GEMM leaf");

  ModuleOp module = call->getParentOfType<ModuleOp>();
  SmallVector<Value> operands(call.getOperands().begin(), call.getOperands().end());
  operands.push_back(epilogueArgs);
  SmallVector<Type> types;
  for (Value operand : operands) types.push_back(operand.getType());
  getOrCreateLeafDeclaration(module, fusedName,
                             FunctionType::get(call.getContext(), types, {}));
  OpBuilder builder(call);
  auto replacement = builder.create<func::CallOp>(call.getLoc(), fusedName,
                                                  TypeRange{}, operands);
  replacement->setAttrs(call->getAttrs());
  replacement.setCallee(fusedName);
  call.erase();
  return success();
}

class AArch64GemmEpilogueLowering
    : public AArch64GemmEpilogueLoweringBase<AArch64GemmEpilogueLowering> {
 public:
  using Base::Base;

  void runOnOperation() override {
    SmallVector<linalg::GenericOp> carriers;
    getOperation().walk([&](linalg::GenericOp carrier) {
      if (carrier->hasAttr(aarch64::gemm::kEpilogueCarrierAttrName))
        carriers.push_back(carrier);
    });
    for (linalg::GenericOp carrier : carriers)
      if (failed(lowerEpilogue(carrier))) {
        signalPassFailure();
        return;
      }
  }

 private:
  LogicalResult lowerEpilogue(linalg::GenericOp carrier) {
    auto call = llvm::dyn_cast_or_null<func::CallOp>(carrier->getPrevNode());
    if (!call || !aarch64::gemm::isMicrokernelLeaf(call.getCallee()) ||
        !call->hasAttr(aarch64::gemm::kEpilogueInsertionPointAttrName))
      return carrier.emitOpError("has no preceding selected GEMM leaf");
    if (hasLegacyMatmulEpilogueAttrs(carrier))
      return carrier.emitOpError("legacy MatMul Bias/Relu attributes conflict with annc.gemm.epilogue");
    auto program = atir::parseEpilogueCandidatePlan(
        carrier->getAttrOfType<mlir::DictionaryAttr>(atir::kGemmEpilogueAttr));
    if (mlir::failed(program))
      return carrier.emitOpError("has an unsupported GEMM epilogue attr");
    FailureOr<aarch64::gemm::FinalMicrokernelSelection> selection =
        aarch64::gemm::readFinalMicrokernelSelection(call);
    if (failed(selection) || !selection->supportsEpilogueCodegen)
      return call.emitOpError(
          "selected microkernel does not support epilogue codegen");
    std::vector<codegen::Step> steps = convertEpilogueSteps(*program);
    std::string error;
    codegen::PreparedKernel generated;
    if (!codegen::prepare(selection->codegenKernel, steps, generated, error))
      return carrier.emitOpError() << "cannot prepare an epilogue prefix: "
                                   << error;
    const unsigned prefixSize = generated.supportedPrefixLength;
    if (prefixSize > program->steps.size())
      return carrier.emitOpError("codegen returned an invalid prefix length");

    atir::EpilogueProgram prefix = *program;
    prefix.steps.assign(program->steps.begin(),
                        program->steps.begin() + prefixSize);
    if (prefixSize != 0) {
      OpBuilder argsBuilder(call);
      FailureOr<Value> args = materializeCarrierEpilogueArgs(
          call, carrier, prefix.steps, argsBuilder);
      if (failed(args)) return failure();
      if (failed(makeFusedCall(call, carrier, *args))) return failure();
      call = llvm::cast<func::CallOp>(carrier->getPrevNode());
      Builder attrBuilder(&getContext());
      call->setAttr(atir::kGemmEpilogueAttr,
                    atir::buildEpilogueCandidateAttr(attrBuilder, prefix));
    }
    if (prefixSize != 0) {
      mlir::Builder builder(&getContext());
      mlir::NamedAttrList metadata;
      metadata.append("version", builder.getI32IntegerAttr(1));
      metadata.append("symbol", builder.getStringAttr(generated.entrySymbol));
      metadata.append("normalized_request",
                      builder.getStringAttr(generated.normalizedRequest));
      metadata.append("artifact_key", builder.getStringAttr(generated.artifactKey));
      metadata.append("build_id", builder.getStringAttr(generated.buildId));
      call->setDiscardableAttr(aarch64::gemm::kGeneratedMicrokernelAttrName,
                               metadata.getDictionary(&getContext()));
    }
    if (prefixSize == program->steps.size()) {
      carrier.erase();
      return success();
    }
    if (failed(materializeSuffix(carrier, *program, prefixSize)))
      return failure();
    carrier.erase();
    return success();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64GemmEpilogueLowering() {
  return std::make_unique<AArch64GemmEpilogueLowering>();
}

}  // namespace annc
