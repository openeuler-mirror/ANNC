#include <functional>
#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/FusionV2/Naming.h"
#include "Dialect/Atir/Passes/FusionV2/Patterns.h"
#include "Dialect/Atir/Passes/FusionV2/Provenance.h"
#include "Dialect/Atir/Passes/GemmEpilogueCandidate.h"
#include "Dialect/Atir/TemplateFingerprint.h"
#include "Support/GemmConfig.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace atir::fusionv2;

namespace atir {
namespace {

constexpr unsigned kMaxEpilogueSteps = 8;

// TF node name for the legalized dot: legalization drops metadata, so the
// name is recovered from the location (wd_dcn/presort: NameLoc or fused
// [...]; the fused case scans parts in reverse so the innermost name wins).
static std::string anchorName(Operation *op) {
  Location loc = op->getLoc();
  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (Location part : llvm::reverse(fused.getLocations())) {
      if (auto name = dyn_cast<NameLoc>(part)) {
        if (!name.getName().empty()) return name.getName().str();
      }
    }
  }
  if (auto name = dyn_cast<NameLoc>(loc)) return name.getName().str();
  return "";
}

// i64 array attribute helper (metadata fields).
static ArrayAttr makeI64Array(MLIRContext *ctx, ArrayRef<int64_t> values) {
  SmallVector<Attribute> attrs;
  Builder builder(ctx);
  for (int64_t value : values)
    attrs.push_back(builder.getI64IntegerAttr(value));
  return ArrayAttr::get(ctx, attrs);
}

// Boundary record {role, tf_name, shape, rank, dtype} on builtin
// RankedTensorType (V2 carrier; the V1 helpers only accept !atir.tensor).
// dtype is "f32": the first batch admits f32 sites only (shape guard).
static DictionaryAttr makeArgRecord(MLIRContext *ctx, StringRef role,
                                    StringRef tfName, RankedTensorType type) {
  Builder builder(ctx);
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("role", builder.getStringAttr(role)));
  attrs.push_back(
      builder.getNamedAttr("tf_name", builder.getStringAttr(tfName)));
  attrs.push_back(builder.getNamedAttr(
      "shape", makeI64Array(ctx, type.getShape())));
  attrs.push_back(builder.getNamedAttr(
      "rank", builder.getI64IntegerAttr(
                  static_cast<int64_t>(type.getShape().size()))));
  attrs.push_back(builder.getNamedAttr(
      "dtype", builder.getStringAttr("f32")));
  return DictionaryAttr::get(ctx, attrs);
}

// TF name of a boundary value: tf.arg_names by argument position for block
// args, the loc anchor otherwise (a name absent from the pb drops a site).
// TF name of a boundary value; an empty result means the name is not a pb
// node and the caller must drop the site (converter-side fail-closed).
static std::string operandName(Value value) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto fn = dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
      if (auto names = fn->getAttrOfType<ArrayAttr>("tf.arg_names");
          names && blockArg.getArgNumber() < names.size()) {
        if (auto name = dyn_cast<StringAttr>(names[blockArg.getArgNumber()]);
            name && !name.getValue().empty())
          return name.getValue().str();
      }
    }
    if (auto loc = dyn_cast<NameLoc>(blockArg.getLoc()))
      return loc.getName().str();
    return "";
  }
  if (Operation *op = value.getDefiningOp()) return anchorName(op);
  return "";
}

// V1 hasMatchingBatchDim on builtin types: an input whose leading dim
// equals the output leading dim carries the batch axis (dynamic).
static bool hasMatchingLeadingDim(RankedTensorType inputType,
                                  RankedTensorType outputType) {
  return !inputType.getShape().empty() &&
         inputType.getShape().front() == outputType.getShape().front();
}

//===----------------------------------------------------------------------===//
// Epilogue chain matching (single-use tail off the dot, wd-sample shapes)
//===----------------------------------------------------------------------===//

struct EpilogueStep {
  Operation *member;   // stablehlo add/multiply/maximum, cloned into kernel
  StringRef opcode;    // "add" / "mul" / "relu" (V1 epilogue naming)
  Value external;      // epilogue input value (kernel param); relu: null
  Operation *identityBcast = nullptr;  // chain-side self-shape broadcast
  unsigned inputIndex = 0;  // external operand slot in the binary op (0/1)
};

struct EpilogueMatch {
  SmallVector<EpilogueStep> steps;
  Value tail;          // terminal value (returned; may have many users)
  bool empty() const { return steps.empty(); }
  std::string patternName() const {
    std::string name = "matmul";
    for (const EpilogueStep &step : steps)
      name += (Twine("_") + step.opcode).str();
    return name;
  }
};

// Recognizes the relu lower bound: a splatted-zero f32 constant, possibly
// through a scalar dynamic_broadcast_in_dim (wd-sample legalization shape).
static bool isZeroReluBound(Value v) {
  if (auto bcast =
          v.getDefiningOp<stablehlo::DynamicBroadcastInDimOp>()) {
    if (!bcast.getBroadcastDimensions().empty()) return false;
    v = bcast.getOperand();
  }
  auto cst = v.getDefiningOp<stablehlo::ConstantOp>();
  if (!cst) return false;
  auto type = dyn_cast<RankedTensorType>(cst.getType());
  if (!type || !type.getElementType().isF32()) return false;
  auto dense = cst.getValue();
  return dense && dense.isSplat() &&
         dense.getSplatValue<float>() == 0.0f;
}

// Recognizes the weight broadcast of the wd sample: rank<=1 f32 input
// broadcast along the output axis [1] (bias/scale/shift alignment).
static stablehlo::DynamicBroadcastInDimOp weightBroadcast(Value v) {
  auto bcast = v.getDefiningOp<stablehlo::DynamicBroadcastInDimOp>();
  if (!bcast) return nullptr;
  auto dims = bcast.getBroadcastDimensions();
  if (dims.size() != 1 || dims.front() != 1) return nullptr;
  auto type = dyn_cast<RankedTensorType>(bcast.getOperand().getType());
  if (!type || type.getRank() > 1 || !type.getElementType().isF32())
    return nullptr;
  return bcast;
}

// Identity broadcast: full-rank dimension mapping with a shape-compatible
// output — the legalization wrapper TF AddV2/Mul puts around tensor values
// before the elementwise epilogue op (dims = [0, 1, ...]).
static bool isIdentityBroadcast(stablehlo::DynamicBroadcastInDimOp bcast) {
  auto inType = dyn_cast<RankedTensorType>(bcast.getOperand().getType());
  auto outType = dyn_cast<RankedTensorType>(bcast.getType());
  if (!inType || !outType || inType.getRank() != outType.getRank())
    return false;
  auto dims = bcast.getBroadcastDimensions();
  if (static_cast<int64_t>(dims.size()) != inType.getRank()) return false;
  for (int64_t axis = 0; axis < inType.getRank(); ++axis) {
    if (dims[axis] != axis) return false;
    if (inType.getShape()[axis] != outType.getShape()[axis]) return false;
  }
  return true;
}

// Shape-only scaffolding ops (legalization side chains); a chain value
// may be co-referenced by these without breaking the single-value contract
// of the site.
static bool isShapeScaffold(Operation *op) {
  return isa<shape::ShapeOfOp, shape::BroadcastOp, shape::ConstShapeOp,
             shape::ConcatOp, shape::SplitAtOp, shape::ToExtentTensorOp,
             arith::IndexCastOp, tensor::FromElementsOp, tensor::DimOp>(op);
}

// Walks the single-use epilogue chain off the dot result; multi-user
// intermediates truncate the chain there (deliberate V2 departure from the
// V1 all-or-nothing rule). Identity broadcasts are absorbed as members.
static EpilogueMatch matchEpilogueChain(stablehlo::DotOp dot) {
  EpilogueMatch match;
  Value cur = dot.getResult();
  for (unsigned i = 0; i < kMaxEpilogueSteps; ++i) {
    Operation *user = nullptr;
    Operation *identity = nullptr;
    auto coRefsScaffoldOnly = [&](Operation *except) {
      for (Operation *other : cur.getUsers())
        if (other != except && !isShapeScaffold(other)) return false;
      return true;
    };
    for (Operation *u : cur.getUsers()) {
      auto b = dyn_cast<stablehlo::DynamicBroadcastInDimOp>(u);
      if (!b || !isIdentityBroadcast(b)) continue;
      if (!u->getResult(0).hasOneUse()) continue;
      if (!coRefsScaffoldOnly(u)) continue;
      identity = u;
      user = *u->getResult(0).getUsers().begin();
      break;
    }
    if (!user) {
      for (Operation *u : cur.getUsers()) {
        if (!isa<stablehlo::MaxOp>(u)) continue;
        if (!coRefsScaffoldOnly(u)) continue;
        user = u;
        break;
      }
    }
    if (!user) {
      if (!cur.hasOneUse()) break;
      user = *cur.getUsers().begin();
    }

    Value chainSide = identity ? identity->getResult(0) : cur;
    Operation *member = nullptr;
    StringRef opcode;
    Value external;
    unsigned inputIndex = 0;
    if (auto add = dyn_cast<stablehlo::AddOp>(user)) {
      bool extIsRhs = add.getLhs() == chainSide;
      auto bcast = weightBroadcast(extIsRhs ? add.getRhs() : add.getLhs());
      if (!bcast) break;
      member = user;
      opcode = "add";
      external = bcast.getOperand();
      inputIndex = extIsRhs ? 1u : 0u;
    } else if (auto mul = dyn_cast<stablehlo::MulOp>(user)) {
      bool extIsRhs = mul.getLhs() == chainSide;
      auto bcast = weightBroadcast(extIsRhs ? mul.getRhs() : mul.getLhs());
      if (!bcast) break;
      member = user;
      opcode = "mul";
      external = bcast.getOperand();
      inputIndex = extIsRhs ? 1u : 0u;
    } else if (isa<stablehlo::MaxOp>(user)) {
      Value other = user->getOperand(0) == chainSide
                        ? user->getOperand(1)
                        : user->getOperand(0);
      if (!isZeroReluBound(other)) break;
      member = user;
      opcode = "relu";
    } else {
      break;
    }
    match.steps.push_back({member, opcode, external, identity, inputIndex});
    cur = user->getResult(0);
  }
  match.tail = cur;
  return match;
}

// V1-compatible epilogue candidate plan; encoding mirrors the V1 parser
// vocabulary (parseEpilogueCandidatePlan / buildEpilogueCandidateAttr) but
// is spelled out locally to avoid a Passes -> Target symbol dependency.
static DictionaryAttr buildV2EpilogueCandidate(
    MLIRContext *ctx, const EpilogueMatch &site,
    PatternRewriter &rewriter) {
  SmallVector<Attribute> steps;
  for (const EpilogueStep &step : site.steps) {
    StringRef broadcast = (step.opcode == "relu") ? "scalar" : "n";
    SmallVector<NamedAttribute> encoded;
    encoded.push_back(rewriter.getNamedAttr(
        "opcode", rewriter.getStringAttr(step.opcode)));
    encoded.push_back(rewriter.getNamedAttr(
        "input", rewriter.getI64IntegerAttr(step.inputIndex)));
    encoded.push_back(
        rewriter.getNamedAttr("broadcast", rewriter.getStringAttr(broadcast)));
    if (step.opcode == "relu")
      encoded.push_back(
          rewriter.getNamedAttr("limit", rewriter.getF32FloatAttr(-1.0f)));
    steps.push_back(DictionaryAttr::get(ctx, encoded));
  }
  SmallVector<NamedAttribute> candidate;
  candidate.push_back(rewriter.getNamedAttr(
      "version", rewriter.getI64IntegerAttr(kGemmEpilogueCandidateVersion)));
  candidate.push_back(
      rewriter.getNamedAttr("output_role", rewriter.getStringAttr("main_d")));
  candidate.push_back(rewriter.getNamedAttr(
      "terminal_view", rewriter.getStringAttr("identity")));
  candidate.push_back(
      rewriter.getNamedAttr("steps", ArrayAttr::get(ctx, steps)));
  return DictionaryAttr::get(ctx, candidate);
}


//===----------------------------------------------------------------------===//
// Pattern
//===----------------------------------------------------------------------===//

struct DotGemmV2Pattern : public OpRewritePattern<stablehlo::DotOp> {
  using OpRewritePattern<stablehlo::DotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DotOp dot,
                                PatternRewriter &rewriter) const override {
    const annc::GemmConfig &gemm = annc::gemmConfig();
    if (gemm.mode == annc::GemmMode::Off) return failure();
    if (dot->hasAttr("annc.fusion_materialized")) return failure();

    auto lhsType = dyn_cast<RankedTensorType>(dot.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(dot.getRhs().getType());
    auto outType = dyn_cast<RankedTensorType>(dot.getType());
    if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2 || outType.getRank() != 2 ||
        !lhsType.getElementType().isF32() ||
        !rhsType.getElementType().isF32()) {
      return failure();
    }

    func::FuncOp parent = dot->getParentOfType<func::FuncOp>();
    ModuleOp module = dot->getParentOfType<ModuleOp>();
    if (!parent || !module) return failure();
    MLIRContext *ctx = rewriter.getContext();

    EpilogueMatch epilogue;
    if (gemm.epilogue) epilogue = matchEpilogueChain(dot);
    if (epilogue.steps.empty()) epilogue.tail = dot.getResult();

    auto tryBuild = [&](EpilogueMatch &site) -> LogicalResult {
      const std::string patternName = site.patternName();

      bool lhsIsDynamic = hasMatchingLeadingDim(lhsType, outType);
      bool rhsIsDynamic = hasMatchingLeadingDim(rhsType, outType);
      if (lhsIsDynamic == rhsIsDynamic) {
        lhsIsDynamic = true;
        rhsIsDynamic = false;
      }
      SmallVector<Value> fixedInputs = {dot.getRhs()};
      for (const EpilogueStep &step : site.steps)
        if (step.external &&
            !llvm::is_contained(fixedInputs, step.external))
          fixedInputs.push_back(step.external);
      Value dynamicInput = dot.getLhs();
      if (!lhsIsDynamic) {
        fixedInputs[0] = dot.getLhs();
        dynamicInput = dot.getRhs();
      }
      // Signature order must match the metadata args order (fixed weights
      // first, dynamic input last): the runtime collects shapes by the
      // ANNCFused input position, so a dynamic-first signature misaligns
      // every shape by one slot (argument rank mismatch on specialize).
      SmallVector<Value> signatureInputs = fixedInputs;
      signatureInputs.push_back(dynamicInput);
      const std::string kernelName = uniquifySymbolName(
          module, site.empty()
                      ? "fused_matmul_" +
                            hash4(anchorName(dot.getOperation()))
                      : "fused_" + patternName + "_" +
                            hash4(anchorName(dot.getOperation()), patternName));

      // All pb names are resolved before the kernel is created: a failure
      // here must leave no half-built site behind (the converter drops
      // nameless metadata entries, so naming is fail-closed).
      SmallVector<std::string> fixedNames;
      for (Value fixed : fixedInputs) {
        fixedNames.push_back(operandName(fixed));
        if (fixedNames.back().empty()) return failure();
      }
      std::string dynamicName = operandName(dynamicInput);
      if (dynamicName.empty()) return failure();
      std::string siteName = site.empty() ? operandName(dot.getResult())
                                          : operandName(site.tail);
      if (siteName.empty()) return failure();

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());
      SmallVector<Type> inputTypes;
      for (Value input : signatureInputs) inputTypes.push_back(input.getType());
      auto tailType = cast<RankedTensorType>(site.tail.getType());
      auto funcType = rewriter.getFunctionType(inputTypes, TypeRange{tailType});
      auto kernel = rewriter.create<func::FuncOp>(
          module.getLoc(), kernelName, funcType);
      kernel.setPrivate();
      kernel->setAttr("llvm.emit_c_interface",
                      UnitAttr::get(rewriter.getContext()));
      kernel->setAttr("fusion.pattern", rewriter.getStringAttr(patternName));
      kernel->setAttr("annc.kernel", rewriter.getUnitAttr());
      kernel->setAttr("annc.execution_mode", rewriter.getStringAttr("jit"));

      Block *entry = kernel.addEntryBlock();
      rewriter.setInsertionPointToEnd(entry);

      IRMapping mapping;
      for (auto [input, arg] :
           llvm::zip(signatureInputs, entry->getArguments()))
        mapping.map(input, arg);

      SmallVector<Operation *, 16> members = {dot.getOperation()};
      for (const EpilogueStep &step : site.steps) {
        if (step.identityBcast) members.push_back(step.identityBcast);
        members.push_back(step.member);
      }

      SmallPtrSet<Operation *, 16> memberSet(members.begin(), members.end());

      SmallVector<Operation *, 16> auxQueue;
      SmallPtrSet<Operation *, 16> auxSeen;
      std::function<LogicalResult(Value)> emitValue =
          [&](Value v) -> LogicalResult {
        if (mapping.contains(v)) return success();
        Operation *def = v.getDefiningOp();
        if (!def) return failure();  // main block arg that is not a param
        if (memberSet.contains(def)) {
          for (Value operand : def->getOperands())
            if (failed(emitValue(operand))) return failure();
          rewriter.clone(*def, mapping);
          return success();
        }
        if (!isa<stablehlo::ConstantOp, stablehlo::DynamicBroadcastInDimOp,
                 stablehlo::GetDimensionSizeOp, stablehlo::ReshapeOp,
                 stablehlo::ConcatenateOp, stablehlo::ConvertOp,
                 shape::ShapeOfOp, shape::BroadcastOp, shape::ConstShapeOp,
                 shape::ConcatOp, shape::SplitAtOp, shape::ToExtentTensorOp,
                 arith::IndexCastOp, tensor::FromElementsOp, tensor::DimOp>(
                def))
          return failure();
        if (auxSeen.insert(def).second) {
          for (Value operand : def->getOperands())
            if (failed(emitValue(operand))) return failure();
          auxQueue.push_back(def);
          rewriter.clone(*def, mapping);
        } else if (!llvm::is_contained(auxQueue, def)) {
          return failure();  // cyclic dependency — bail out
        }
        return success();
      };

      if (failed(emitValue(site.tail))) {
        rewriter.eraseOp(kernel);
        return failure();
      }
      rewriter.create<func::ReturnOp>(kernel.getLoc(),
                                      mapping.lookup(site.tail));

      if (!site.empty()) {
        Operation *dotClone = mapping.lookup(dot.getOperation());
        dotClone->setAttr(kGemmEpilogueCandidateAttr,
                          buildV2EpilogueCandidate(ctx, site, rewriter));
      }

      SmallVector<Attribute> args;
      for (auto [fixed, name] : llvm::zip(fixedInputs, fixedNames))
        args.push_back(makeArgRecord(
            ctx, "fixed", name, cast<RankedTensorType>(fixed.getType())));
      args.push_back(
          makeArgRecord(ctx, "dynamic", dynamicName, lhsType));
      SmallVector<Attribute> outputs = {
          makeArgRecord(ctx, "output", siteName + ":0", tailType)};
      SmallVector<NamedAttribute> metadata;
      metadata.push_back(rewriter.getNamedAttr(
          "fusion.pattern", rewriter.getStringAttr(patternName)));
      metadata.push_back(rewriter.getNamedAttr(
          "execution_mode", rewriter.getStringAttr("jit")));
      metadata.push_back(rewriter.getNamedAttr(
          "kernel_name", rewriter.getStringAttr(kernelName)));
      metadata.push_back(rewriter.getNamedAttr(
          "tf.name",
          rewriter.getStringAttr(uniquifyFusionName(
              module, sanitizeName("annc_fused_" + siteName)))));
      metadata.push_back(
          rewriter.getNamedAttr("args", ArrayAttr::get(ctx, args)));
      metadata.push_back(
          rewriter.getNamedAttr("outputs", ArrayAttr::get(ctx, outputs)));
      // Context v3 contract (outlineContextBuiltin line): outputs are
      // allocated through the context, so kernel_arg_order stays empty and
      // no dynamic-input mapping is attached.
      metadata.push_back(rewriter.getNamedAttr(
          "abi", rewriter.getStringAttr("annc_context_v3")));
      metadata.push_back(
          rewriter.getNamedAttr("kernel_arg_order", makeI64Array(ctx, {})));
      metadata.push_back(rewriter.getNamedAttr(
          "symbolic_signature", rewriter.getStringAttr("")));
      metadata.push_back(rewriter.getNamedAttr(
          "fallback_function", rewriter.getStringAttr("original_subgraph")));
      metadata.push_back(rewriter.getNamedAttr(
          "template_fingerprint",
          rewriter.getStringAttr(
              atir::computeAtirTemplateFingerprint(kernel))));
      kernel->setAttr("fusion.metadata",
                      rewriter.getDictionaryAttr(metadata));

      Operation *lastMember = members.back();
      rewriter.setInsertionPointAfter(lastMember);
      auto call = rewriter.create<func::CallOp>(
          lastMember->getLoc(), kernel, signatureInputs);
      rewriter.replaceAllUsesWith(site.tail, call.getResult(0));
      // Scaffolding may co-reference intermediate chain values (admitted by
      // coRefsScaffoldOnly); epilogue steps are shape-preserving, so their
      // remaining main-side users re-point to the call result before erasing.
      for (Operation *member : members)
        for (Value r : member->getResults())
          if (r != site.tail)
            r.replaceUsesWithIf(call.getResult(0), [&](OpOperand &use) {
              return use.getOwner()->getParentOfType<func::FuncOp>() !=
                     kernel;
            });
      for (Operation *member : llvm::reverse(members))
        rewriter.eraseOp(member);
      for (Operation *def : llvm::reverse(auxQueue))
        if (def->use_empty()) rewriter.eraseOp(def);
      return success();
    };

    if (failed(tryBuild(epilogue))) {
      if (epilogue.empty()) return failure();
      epilogue = EpilogueMatch{};
      epilogue.tail = dot.getResult();
      if (failed(tryBuild(epilogue))) return failure();
    }
    return success();
  }
};

}  // namespace

// Registers the dot-anchored GEMM pattern (gated by ANNC_GEMM at match).
void addDotGemmV2Pattern(RewritePatternSet &patterns) {
  patterns.add<DotGemmV2Pattern>(patterns.getContext());
}

}  // namespace atir
