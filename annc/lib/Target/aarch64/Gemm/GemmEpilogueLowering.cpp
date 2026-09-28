#include <algorithm>

#include "Dialect/Atir/Passes/GemmEpilogueCandidate.h"
#include "GemmPlan.h"
#include "Target/aarch64/Gemm/Epilogue/EpilogueEmitter.h"
#include "Target/aarch64/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace annc {
namespace {

namespace epilogue = annc::aarch64::gemm::epilogue;
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
  auto m = builder.getAffineDimExpr(0);
  auto n = builder.getAffineDimExpr(1);
  SmallVector<Value> inputs;
  SmallVector<AffineMap> maps;
  auto broadcastMap = [&](MemRefType type) -> FailureOr<AffineMap> {
    auto map = [&](ArrayRef<AffineExpr> results) {
      return AffineMap::get(2, 0, results, builder.getContext());
    };
    auto zero = builder.getAffineConstantExpr(0);
    if (type.getRank() == 0) return map({});
    if (type.getRank() == 1)
      return type.getDimSize(0) == 1 ? map({zero}) : map({n});
    if (type.getRank() != 2) return failure();
    if (type.getShape() == outputType.getShape()) return map({m, n});
    if (type.getDimSize(0) == 1 && type.getDimSize(1) == 1)
      return map({zero, zero});
    if (type.getDimSize(1) == 1) return map({m, zero});
    if (type.getDimSize(0) == 1) return map({zero, n});
    return failure();
  };
  unsigned binaryIndex = 0;
  for (unsigned index = 0; index < program.steps.size(); ++index) {
    const atir::EpilogueStep &step = program.steps[index];
    if (!atir::isBinaryEpilogueOpcode(step.opcode)) continue;
    if (index >= prefixSize) {
      if (2 + binaryIndex >= generic.getInputs().size())
        return generic.emitOpError("has incomplete epilogue inputs");
      Value input = generic.getInputs()[2 + binaryIndex];
      inputs.push_back(input);
      auto inputType = llvm::dyn_cast<MemRefType>(input.getType());
      if (!inputType)
        return generic.emitOpError(
            "epilogue suffix inputs must be ranked memrefs");
      auto map = broadcastMap(inputType);
      if (failed(map))
        return generic.emitOpError("has an invalid epilogue suffix shape");
      maps.push_back(*map);
    }
    ++binaryIndex;
  }
  maps.push_back(AffineMap::get(2, 0, {m, n}, builder.getContext()));
  SmallVector<atir::EpilogueStep> suffix(program.steps.begin() + prefixSize,
                                         program.steps.end());
  builder.create<linalg::GenericOp>(
      generic.getLoc(), TypeRange{}, inputs, generic.getOutputs(), maps,
      SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                       utils::IteratorType::parallel},
      [suffix](OpBuilder &nested, Location loc, ValueRange args) {
        Value value = args.back();
        unsigned input = 0;
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

epilogue::FragmentRegisterBinding fixedRegisterLayout(
    aarch64::gemm::GemmIsa isa, int64_t mr, int64_t ng) {
  epilogue::FragmentRegisterBinding layout;
  if (mr <= 0 || ng <= 0) return layout;
  const int64_t accStride =
      isa == aarch64::gemm::GemmIsa::kSve ? 4 : ng;
  for (int64_t row = 0; row < mr; ++row)
    for (int64_t group = 0; group < ng; ++group)
      layout.accumulatorRegisters.push_back(
          static_cast<unsigned>(8 + row * accStride + group));
  layout.scratchRegisters = {0, 1, 2, 3, 4, 5, 6, 7};
  if (isa == aarch64::gemm::GemmIsa::kSve) layout.predicateRegister = 0;
  layout.nGroups = static_cast<unsigned>(ng);
  return layout;
}

bool emitExternalLoad(aarch64::gemm::GemmIsa isa, atir::BroadcastKind broadcast,
                      int64_t ng, int64_t microN, unsigned inputSlot,
                      unsigned scratchOffset,
                      const epilogue::FragmentRegisterBinding &layout,
                      llvm::SmallVectorImpl<std::string> &instructions) {
  if (broadcast != atir::BroadcastKind::kScalar &&
      broadcast != atir::BroadcastKind::kN)
    return false;
  instructions.push_back("ldr x9, [x7, #" + llvm::utostr(inputSlot * 8) + "]");
  for (int64_t group = 0; group < ng; ++group) {
    std::string reg = llvm::utostr(
        layout.scratchRegisters[scratchOffset + static_cast<unsigned>(group)]);
    if (isa == aarch64::gemm::GemmIsa::kSve) {
      std::string predicate = "p" + llvm::utostr(group) + "/z, [x9";
      instructions.push_back(broadcast == atir::BroadcastKind::kScalar
                                 ? "ld1rw z" + reg + ".s, " + predicate + "]"
                                 : "ld1w z" + reg + ".s, " + predicate + ", #" +
                                       llvm::utostr(group) + ", MUL VL]");
    } else {
      if (broadcast == atir::BroadcastKind::kScalar) {
        instructions.push_back("ld1r {v" + reg + ".4s}, [x9]");
      } else {
        int64_t lanes = std::min<int64_t>(4, microN - group * 4);
        if (lanes == 4) {
          instructions.push_back("ldr q" + reg + ", [x9], #16");
        } else {
          instructions.push_back("movi v" + reg + ".4s, #0");
          for (int64_t lane = 0; lane < lanes; ++lane)
            instructions.push_back("ld1 {v" + reg + ".s}[" +
                                   llvm::utostr(lane) + "], [x9], #4");
        }
      }
    }
  }
  return true;
}

mlir::FailureOr<unsigned> selectPrefix(const atir::EpilogueProgram &program,
                                       const aarch64::gemm::GemmPlan &plan) {
  const auto &registry = epilogue::getEpilogueEmitterRegistry(plan.isa);
  return epilogue::selectFusiblePrefix(program, plan, registry);
}

class AArch64GemmEpilogueLowering
    : public AArch64GemmEpilogueLoweringBase<AArch64GemmEpilogueLowering> {
 public:
  using Base::Base;

  void runOnOperation() override {
    SmallVector<linalg::GenericOp> anchors;
    getOperation().walk([&](linalg::GenericOp generic) {
      if (generic->hasAttr(atir::kGemmEpilogueAttr) &&
          generic->hasAttr(aarch64::gemm::kPlanAttrName))
        anchors.push_back(generic);
    });
    for (linalg::GenericOp generic : anchors)
      if (failed(selectAnchorPrefix(generic))) {
        signalPassFailure();
        return;
      }
    getOperation().walk([&](func::CallOp call) {
      if (!aarch64::gemm::isMicrokernelLeaf(call.getCallee())) return;
      if (!call->hasAttr(atir::kGemmEpilogueAttr)) return;
      if (failed(lowerEpilogue(call))) {
        signalPassFailure();
        return;
      }
    });
  }

 private:
  LogicalResult selectAnchorPrefix(linalg::GenericOp generic) {
    if (hasLegacyMatmulEpilogueAttrs(generic))
      return generic.emitOpError("legacy MatMul Bias/Relu attributes conflict with annc.gemm.epilogue");
    if (auto role = generic->getAttrOfType<StringAttr>(
            aarch64::gemm::kKcRoleAttrName)) {
      if (role.getValue() == "first" || role.getValue() == "middle") {
        generic->removeAttr(atir::kGemmEpilogueAttr);
        return success();
      }
    }
    auto program =
        atir::parseEpilogueCandidatePlan(generic->getAttrOfType<DictionaryAttr>(
            atir::kGemmEpilogueAttr));
    FailureOr<aarch64::gemm::GemmPlan> plan = aarch64::gemm::readPlan(generic);
    if (failed(program) || failed(plan))
      return generic.emitOpError("has an invalid fixed epilogue plan");
    auto selected = selectPrefix(*program, *plan);
    if (failed(selected))
      return generic.emitOpError("cannot select an epilogue prefix");
    unsigned prefixSize = *selected;
    if (prefixSize == program->steps.size()) return success();
    if (failed(materializeSuffix(generic, *program, prefixSize)))
      return failure();
    if (prefixSize == 0) {
      generic->removeAttr(atir::kGemmEpilogueAttr);
      return success();
    }
    atir::EpilogueProgram prefix;
    prefix.steps.assign(program->steps.begin(),
                        program->steps.begin() + prefixSize);
    Builder builder(&getContext());
    generic->setAttr(atir::kGemmEpilogueAttr,
                     atir::buildEpilogueCandidateAttr(builder, prefix));
    return success();
  }

  mlir::LogicalResult lowerEpilogue(func::CallOp call) {
    if (hasLegacyMatmulEpilogueAttrs(call))
      return call.emitOpError("legacy MatMul Bias/Relu attributes conflict with annc.gemm.epilogue");
    auto program = atir::parseEpilogueCandidatePlan(
        call->getAttrOfType<mlir::DictionaryAttr>(
            atir::kGemmEpilogueAttr));
    if (mlir::failed(program))
      return call.emitOpError("has an unsupported GEMM epilogue attr");

    FailureOr<aarch64::gemm::GemmPlan> plan = aarch64::gemm::readPlan(call);
    if (mlir::failed(plan))
      return call.emitOpError("cannot read the GEMM plan for epilogue fusion");
    auto selected = selectPrefix(*program, *plan);
    if (mlir::failed(selected))
      return call.emitOpError("cannot select an epilogue prefix");
    if (!*selected) {
      call->removeAttr(atir::kGemmEpilogueAttr);
      return success();
    }
    if (*selected < program->steps.size()) program->steps.resize(*selected);
    auto mAttr = call->getAttrOfType<mlir::IntegerAttr>(
        aarch64::gemm::kMicrokernelMAttrName);
    auto nAttr = call->getAttrOfType<mlir::IntegerAttr>(
        aarch64::gemm::kMicrokernelNAttrName);
    auto kAttr = call->getAttrOfType<mlir::IntegerAttr>(
        aarch64::gemm::kMicrokernelKAttrName);
    auto kcModeAttr =
        call->getAttrOfType<mlir::StringAttr>(aarch64::gemm::kKcModeAttrName);
    if (!mAttr || !nAttr || !kcModeAttr ||
        (kcModeAttr.getValue() != "overwrite" &&
         kcModeAttr.getValue() != "accumulate"))
      return call.emitOpError(
          "requires microkernel M/N and a valid KC mode for epilogue fusion");
    auto kcRoleAttr =
        call->getAttrOfType<mlir::StringAttr>(aarch64::gemm::kKcRoleAttrName);
    if (!kcRoleAttr)
      return call.emitOpError("requires an explicit KC role for fusion");
    if (kcRoleAttr.getValue() == "first" || kcRoleAttr.getValue() == "middle") {
      call->removeAttr(atir::kGemmEpilogueAttr);
      return success();
    }
    if (kcRoleAttr.getValue() != "single" && kcRoleAttr.getValue() != "final")
      return call.emitOpError("has an invalid KC role for epilogue fusion");

    const int64_t mr = mAttr.getInt();
    const int64_t microN = nAttr.getInt();
    const int64_t lanes = plan->vectorLengthBytes / sizeof(float);
    if (lanes <= 0)
      return call.emitOpError(
          "has an invalid vector length for epilogue fusion");
    const int64_t ng = (microN + lanes - 1) / lanes;
    const int64_t maxMr =
        plan->executionKind ==
                aarch64::gemm::GemmExecutionKind::kMatrixVector
            ? 4
            : 6;
    if (mr < 1 || mr > maxMr || ng < 1 || ng > 4)
      return call.emitOpError("fused epilogue generation requires NG in 1..4");

    const auto &registry = epilogue::getEpilogueEmitterRegistry(plan->isa);

    const auto layout = fixedRegisterLayout(plan->isa, mr, ng);
    llvm::SmallVector<std::string> instructions;
    unsigned inputSlot = 0;
    for (const atir::EpilogueStep &step : program->steps) {
      if (atir::isBinaryEpilogueOpcode(step.opcode)) {
        if (!emitExternalLoad(plan->isa, step.broadcast, ng, microN, inputSlot,
                              0, layout, instructions)) {
          return call.emitOpError(
                         "epilogue step broadcast is not supported by the "
                         "microkernel lowering yet")
                     .attachNote()
                 << "opcode=" << static_cast<unsigned>(step.opcode)
                 << " broadcast=" << static_cast<unsigned>(step.broadcast);
        }
        ++inputSlot;
      }
      epilogue::EmitRequest request{step.opcode, layout, step.reluLimit};
      auto emitted = registry.emit(request);
      if (mlir::failed(emitted))
        return call.emitOpError("epilogue emitter failed");
      for (std::string &instruction : *emitted)
        instructions.push_back(std::move(instruction));
    }

    epilogue::MicrokernelRequest kernelRequest;
    kernelRequest.isa = plan->isa;
    kernelRequest.mr = mr;
    kernelRequest.ng = ng;
    kernelRequest.rowMajor =
        plan->rhsPacking == aarch64::gemm::RhsPacking::kDirect;
    kernelRequest.microN = microN;
    kernelRequest.kcMode = kcModeAttr.getValue() == "accumulate"
                               ? aarch64::gemm::KcMode::kAccumulate
                               : aarch64::gemm::KcMode::kOverwrite;
    kernelRequest.matrixVector =
        plan->executionKind ==
        aarch64::gemm::GemmExecutionKind::kMatrixVector;
    std::string suffix = atir::staticSymbolSuffix(*program, false);
    llvm::StringRef epiloguePart(suffix);
    if (!epiloguePart.starts_with("_") || !epiloguePart.ends_with("_f32"))
      return call.emitOpError("cannot encode the epilogue suffix");
    epiloguePart = epiloguePart.drop_front().drop_back(4);
    kernelRequest.epilogueSuffix = epiloguePart.str();
    kernelRequest.epilogueInstructions = std::move(instructions);
    if (plan->isa == aarch64::gemm::GemmIsa::kNeon) {
      if (!kAttr)
        return call.emitOpError("requires microkernel K for NEON fusion");
      const auto &abi = aarch64::gemm::getGemmKernelABI(
          plan->target, plan->isa, plan->dataType, plan->executionKind);
      auto unroll = aarch64::gemm::getGemmKScalarUnroll(abi, plan->dataType);
      if (mlir::failed(unroll))
        return call.emitOpError("cannot determine the NEON K unroll");
      kernelRequest.kResidue = kAttr.getInt() % *unroll;
      kernelRequest.kHasGroups = kAttr.getInt() >= *unroll;
    }
    auto generated = epilogue::generateMicrokernel(kernelRequest);
    if (mlir::failed(generated))
      return call.emitOpError("microkernel generation failed");

    mlir::Builder builder(&getContext());
    mlir::NamedAttrList metadata;
    metadata.append("symbol", builder.getStringAttr(generated->symbol));
    metadata.append("asm", builder.getStringAttr(generated->assembly));
    call->setDiscardableAttr(aarch64::gemm::kGeneratedMicrokernelAttrName,
                             metadata.getDictionary(&getContext()));
    return mlir::success();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64GemmEpilogueLowering() {
  return std::make_unique<AArch64GemmEpilogueLowering>();
}

}  // namespace annc
