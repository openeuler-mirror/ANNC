#include "Dialect/Atir/Passes/FusionV2/FusionV2Outline.h"

#include <functional>
#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/FusionV2/CarrierMatchers.h"
#include "Support/Log.h"
#include "Dialect/Atir/Passes/FusionV2/Provenance.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace atir::fusionv2;

namespace atir {
namespace fusionv2 {

// Strips the '@' suffix (functionalize branch copy names) and ':port' segments.
std::string stripAnchorSuffixes(StringRef name) {
  return name.split('@').first.split(':').first.str();
}

void collectLocNames(Location loc, SmallVectorImpl<StringRef> &out) {
  if (auto nl = dyn_cast<NameLoc>(loc)) {
    out.push_back(nl.getName());
    return;
  }
  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (Location member : fused.getLocations()) collectLocNames(member, out);
  }
}

std::string anchorFromLoc(Location loc) {
  SmallVector<StringRef, 4> names;
  collectLocNames(loc, names);
  std::string fallback;
  for (StringRef raw : names) {
    if (raw.empty()) continue;
    if (!raw.contains(':')) return stripAnchorSuffixes(raw);
    if (fallback.empty()) fallback = raw.split(':').first.str();
  }
  return fallback;
}

// Single-name anchor: only a lone NameLoc is trusted; CSE-merged shared
// constants carry a FusedLoc of all use points - an arbitrary use name
// wired back to the pb points at an unrelated node.
std::string singleAnchorName(Location loc) {
  if (auto nl = dyn_cast<NameLoc>(loc))
    return stripAnchorSuffixes(nl.getName());
  return "";
}

// Unwraps a single-value assuming-region yield.
Value peelAssumingYield(Value v) {
  auto assuming = dyn_cast<shape::AssumingOp>(v.getDefiningOp());
  if (!assuming) return v;
  auto yield =
      dyn_cast<shape::AssumingYieldOp>(assuming.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != 1) return Value();
  return yield.getOperand(0);
}

bool isShapeScaffolding(Operation *op) {
  return op && op->getDialect()->getNamespace() == "shape";
}

// Sole data user once shape-scaffolding users are ignored; null if not
// unique.
Operation *singleDataUser(Value value) {
  Operation *data = nullptr;
  for (Operation *user : value.getUsers()) {
    if (isShapeScaffolding(user)) continue;
    if (data) return nullptr;
    data = user;
  }
  return data;
}

// ---- Metadata fields --------------------------------------------------------

// Canonical spelling of FusionArg.dtype; unknown element types yield an
// empty string (the caller gives up the match).
std::string dtypeOf(Type elem) {
  if (isa<Float32Type>(elem)) return "f32";
  if (isa<Float64Type>(elem)) return "f64";
  if (isa<Float16Type>(elem)) return "f16";
  if (isa<BFloat16Type>(elem)) return "bf16";
  if (isa<atir::StringType>(elem)) return "string";
  if (auto intTy = dyn_cast<IntegerType>(elem)) {
    if (intTy.isSignlessInteger(1)) return "bool";
    bool isUnsigned = intTy.isUnsigned();
    switch (intTy.getWidth()) {
      case 8:
        return isUnsigned ? "ui8" : "i8";
      case 16:
        return isUnsigned ? "ui16" : "i16";
      case 32:
        return isUnsigned ? "ui32" : "i32";
      case 64:
        return isUnsigned ? "ui64" : "i64";
      default:
        return "";
    }
  }
  return "";
}

bool isAuxScaffolding(Operation *op) {
  if (!op) return false;
  if (isa<stablehlo::ConstantOp>(op)) return true;
  StringRef ns =
      op->getDialect() ? op->getDialect()->getNamespace() : StringRef();
  return ns == "shape" || ns == "arith" || ns == "tensor";
}

DictionaryAttr probeArgAttr(Builder &builder, const ProbeArg &arg) {
  return DictionaryAttr::get(
      builder.getContext(),
      {builder.getNamedAttr("tf_name", builder.getStringAttr(arg.tfName)),
       builder.getNamedAttr("role", builder.getStringAttr(arg.role)),
       builder.getNamedAttr("shape",
                            builder.getI64ArrayAttr(arg.type.getShape())),
       builder.getNamedAttr("rank",
                            builder.getI64IntegerAttr(arg.type.getRank())),
       builder.getNamedAttr("dtype", builder.getStringAttr(
                                         dtypeOf(arg.type.getElementType())))});
}

SmallVector<std::string> entryInputNames(func::FuncOp main) {
  SmallVector<std::string> entryInputs;
  if (auto argNamesAttr = main->getAttrOfType<ArrayAttr>("tf.arg_names")) {
    for (Attribute name : argNamesAttr)
      if (auto str = dyn_cast<StringAttr>(name))
        entryInputs.push_back(stripAnchorSuffixes(str.getValue()));
  } else if (auto entry =
                 main->getAttrOfType<DictionaryAttr>("tf.entry_function")) {
    if (auto inputs = dyn_cast<StringAttr>(entry.get("inputs"))) {
      SmallVector<StringRef, 8> parts;
      llvm::SplitString(inputs.getValue(), parts, ",");
      for (StringRef part : parts)
        entryInputs.push_back(stripAnchorSuffixes(part));
    }
  }
  return entryInputs;
}

std::string abiNameFor(func::FuncOp main, Value value, StringRef anchorName,
                       StringRef suffix) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner()->getParentOp() == main.getOperation()) {
      SmallVector<std::string> names = entryInputNames(main);
      if (arg.getArgNumber() < names.size()) return names[arg.getArgNumber()];
    }
  } else if (Operation *def = value.getDefiningOp()) {
    std::string name = sourceNameOf(value);
    if (!name.empty()) return name;
    name = anchorFromLoc(def->getLoc());
    if (!name.empty()) return name;
  }
  return (anchorName + suffix).str();
}

LogicalResult collectBoundary(
    func::FuncOp main, ArrayRef<Operation *> members,
    SmallVectorImpl<ProbeArg> &args,
    ArrayRef<std::pair<Value, std::string>> promotedConstants,
    ArrayRef<Value> fixedValues,
    CallPortTable *portTable) {
  DenseSet<Value> produced;
  for (Operation *op : members)
    for (Value r : op->getResults()) produced.insert(r);

  SmallVector<std::string> entryInputs = entryInputNames(main);

  DenseSet<Value> promotedVals;
  for (const auto &[value, tfName] : promotedConstants)
    promotedVals.insert(value);

  DenseSet<Value> seenBoundary;
  for (Operation *op : members) {
    const fusionv2::ReshapeLike reshape = fusionv2::ReshapeLike::match(op);
    for (const auto &it : llvm::enumerate(op->getOperands())) {
      Value operand = it.value();
      if (produced.contains(operand)) continue;
      if (promotedVals.contains(operand)) continue;
      if (reshape && it.index() == 1) {
        Operation *tsDef = operand.getDefiningOp();
        if (tsDef && isa<stablehlo::ConstantOp>(tsDef)) continue;
      }
      Operation *def = operand.getDefiningOp();
      if (def != nullptr && isAuxScaffolding(def))
        continue;
      if (def != nullptr &&
          (isa<stablehlo::ConstantOp>(def) || isa<arith::ConstantOp>(def)))
        continue;

      if (!seenBoundary.insert(operand).second) continue;

      ProbeArg arg;
      auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
      if (!tensorTy) {
        ANNC_LOG_DEBUG("op-fusion-v2")
            << "boundary skip: unranked operand " << it.index() << " of "
            << op->getName().getStringRef().str() << "\n";
        return failure();
      }
      arg.type = tensorTy;
      arg.value = operand;

      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        const size_t index = blockArg.getArgNumber();
        if (blockArg.getOwner()->getParentOp() != main.getOperation())
          continue;
        if (index >= entryInputs.size()) {
          ANNC_LOG_DEBUG("op-fusion-v2")
              << "boundary skip: arg " << index
              << " has no name (entry/arg-names=" << entryInputs.size()
              << ", func=" << main.getName().str() << ")\n";
          return failure();
        }
        arg.tfName = entryInputs[index];
        arg.role =
            llvm::is_contained(fixedValues, operand) ? "fixed" : "dynamic";
      } else {
        Operation *def = operand.getDefiningOp();
        if (auto callOp = dyn_cast<func::CallOp>(def); def && portTable) {
          auto it = portTable->entries.find(callOp.getOperation());
          if (it != portTable->entries.end() &&
              it->second.first == callOp.getCallee().str()) {
            if (auto res = llvm::dyn_cast<OpResult>(operand);
                res && res.getResultNumber() < it->second.second.size()) {
              arg.tfName = it->second.second[res.getResultNumber()];
              arg.type = tensorTy;
              arg.value = operand;
              arg.role = llvm::is_contained(fixedValues, operand) ? "fixed"
                                                                  : "dynamic";
              args.push_back(arg);
              continue;
            }
          }
        }
        arg.tfName = sourceNameOf(operand);
        if (arg.tfName.empty()) {
          arg.tfName = anchorFromLoc(def->getLoc());
          if (arg.tfName.empty()) {
            Operation *named = def;
            for (int hop = 0; hop < 8 && named; ++hop) {
              if (isa<atir::GetDimV2Op, atir::SetDimV2Op>(named))
                named = named->getOperand(0).getDefiningOp();
              else
                break;
            }
            if (named) arg.tfName = anchorFromLoc(named->getLoc());
          }
        }
        arg.role =
            llvm::is_contained(fixedValues, operand)
                ? "fixed"
                : (isa<stablehlo::ConstantOp>(def) ? "constant" : "dynamic");
        if (arg.tfName.empty()) {
          ANNC_LOG_DEBUG("op-fusion-v2")
              << "skip: empty anchor for operand " << it.index() << " of "
              << op->getName().getStringRef().str() << "\n";
          return failure();
        }
      }
      if (dtypeOf(tensorTy.getElementType()).empty()) {
        std::string tyStr;
        llvm::raw_string_ostream(tyStr) << tensorTy.getElementType();
        ANNC_LOG_DEBUG("op-fusion-v2")
            << "boundary skip: unknown dtype " << tyStr << "\n";
        return failure();
      }
      args.push_back(arg);
    }
  }

  for (const auto &[value, tfName] : promotedConstants) {
    auto existed = llvm::find_if(
        args, [&](const ProbeArg &a) { return a.value == value; });
    if (existed != args.end()) {
      existed->tfName = tfName;
      continue;
    }
    auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorTy || dtypeOf(tensorTy.getElementType()).empty()) {
      ANNC_LOG_DEBUG("op-fusion-v2")
          << "boundary skip: promoted constant unranked/unknown dtype\n";
      return failure();
    }
    ProbeArg arg;
    arg.type = tensorTy;
    arg.value = value;
    arg.tfName = tfName;
    Operation *pdef = value.getDefiningOp();
    arg.role =
        (pdef && isa<stablehlo::ConstantOp>(pdef)) ? "constant" : "dynamic";
    args.push_back(arg);
  }

  return success();
}

// Data-chain member closure (semantics in the header). Among atir residues
// only Reshape is a site member (its operand is a column/data boundary);
// other residues (placeholders / carriers) are boundaries.

// Whitelist closure over a legalize-expansion chain; boundary = block args,
// unnamed atir carrier results and constants; any other dialect or a
// definition-less value yields false (drop the site, fail-closed).
bool collectArithClosure(Value root, SmallPtrSetImpl<Operation *> &closure,
                         DenseSet<Value> &boundary,
                         ArrayRef<StringRef> atirInteriorOps) {
  SmallVector<Value, 16> work{root};
  DenseSet<Value> visited;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    if (!visited.insert(v).second) continue;
    if (auto arg = llvm::dyn_cast<BlockArgument>(v)) {
      boundary.insert(arg);
      continue;
    }
    Operation *def = v.getDefiningOp();
    if (def == nullptr) return false;
    if (llvm::is_contained(atirInteriorOps, def->getName().getStringRef())) {
      closure.insert(def);
      for (Value o : def->getOperands()) work.push_back(o);
      continue;
    }
    StringRef ns =
        def->getDialect() ? def->getDialect()->getNamespace() : StringRef();
    if (ns == "atir") {
      boundary.insert(v);
      continue;
    }
    if (ns != "stablehlo" && ns != "shape" && ns != "arith" && ns != "tensor")
      return false;
    closure.insert(def);
    for (Value o : def->getOperands()) work.push_back(o);
  }
  return true;
}

// Members of a data-producing chain in post-order; constants, reshape
// residues and block args terminate the walk.
void collectDataMembers(Value root, SmallPtrSetImpl<Operation *> &seen,
                        SmallVectorImpl<Operation *> &out) {
  Operation *def = root.getDefiningOp();
  if (def == nullptr) return;
  if (isa<stablehlo::ConstantOp>(def) || isa<arith::ConstantOp>(def)) return;
  if (auto residue = dyn_cast<atir::UnsupportedOp>(def))
    if (!fusionv2::ReshapeLike::match(def)) return;
  if (!seen.insert(def).second) return;
  SmallVector<Value> deps(def->getOperands().begin(), def->getOperands().end());
  def->walk([&](Operation *inner) {
    if (inner != def)
      for (Value o : inner->getOperands()) deps.push_back(o);
  });
  for (Value dep : deps) collectDataMembers(dep, seen, out);
  out.push_back(def);
}

// Validates the outlineSite contract: non-empty tails/outputs, matched
// anchors, ranked outputs, and an enclosing module. Returns it on success.
static ModuleOp validateSiteInputs(func::FuncOp main,
                                   ArrayRef<Operation *> tailOps,
                                   ArrayRef<Value> outputs,
                                   ArrayRef<std::string> tailAnchors) {
  if (tailOps.empty() || outputs.empty() ||
      tailAnchors.size() != outputs.size())
    return ModuleOp();
  for (Value out : outputs)
    if (!isa<RankedTensorType>(out.getType())) return ModuleOp();
  ModuleOp module = main->getParentOfType<ModuleOp>();
  return module;
}

// Match-side aux closure: walks member (and region-inner) operands,
// whitelisting pure-compute deps as aux and promoting boundary block args
// / anchored defs into ABI parameters. auxOk=false marks a violation.
struct AuxClosure {
  bool auxOk = true;
  SmallVector<std::pair<Value, std::string>> promotions;
  SmallPtrSet<Operation *, 16> auxSet;
  SmallPtrSet<Operation *, 32> siteOps;
  SmallPtrSet<Value, 8> promotedVals;
  SmallPtrSet<Value, 8> boundaryVals;
  SmallVector<std::string> entryInputs;
  func::FuncOp main;

  AuxClosure(func::FuncOp mainIn, ArrayRef<Operation *> members,
             ArrayRef<Value> argValues)
      : siteOps(members.begin(), members.end()), main(mainIn) {
    for (Value v : argValues) boundaryVals.insert(v);
    entryInputs = entryInputNames(main);
    SmallVector<Operation *, 8> work(members.begin(), members.end());
    while (!work.empty()) {
      Operation *op = work.pop_back_val();
      op->walk([&](Operation *inner) {
        for (Value o : inner->getOperands()) seeOperand(o, work);
      });
    }
  }

  // Memoized canInternalize rejections: shared subexpressions must not be
  // re-walked (diamond chains would otherwise cost exponential time).
  llvm::SmallDenseSet<Operation *, 32> notInternalizable;

  void seeOperand(Value o, SmallVectorImpl<Operation *> &work);

  // A pure def whose operand closure resolves to site values, constants, or
  // boundary values is internalized instead of promoted, keeping the ABI at
  // the pattern's declared arity. Depth-bounded; fail-closed.
  bool canInternalize(Operation *def, unsigned depth) {
    if (depth > 32) return false;
    if (notInternalizable.count(def)) return false;
    if (siteOps.contains(def)) return true;
    // Same criterion as isPureDoRegion: isPure() can be false for
    // conditionally-speculatable stablehlo ops with dynamic dims.
    if (!mlir::isMemoryEffectFree(def) &&
        !def->hasTrait<OpTrait::AlwaysSpeculatableImplTrait>()) {
      notInternalizable.insert(def);
      return false;
    }
    for (Value o : def->getOperands()) {
      if (boundaryVals.contains(o)) continue;
      Operation *d = o.getDefiningOp();
      if (!d) return false;
      if (isa<stablehlo::ConstantOp, arith::ConstantOp>(d)) continue;
      if (!canInternalize(d, depth + 1)) {
        notInternalizable.insert(def);
        return false;
      }
    }
    return true;
  }
};

void AuxClosure::seeOperand(Value o, SmallVectorImpl<Operation *> &work) {
  if (!auxOk || boundaryVals.contains(o)) return;
  Operation *def = o.getDefiningOp();
  if (def == nullptr) {
    auto arg = llvm::cast<BlockArgument>(o);
    if (arg.getOwner()->getParentOp() != main) return;
    auto tensorTy = dyn_cast<RankedTensorType>(o.getType());
    if (arg.getArgNumber() >= entryInputs.size() || tensorTy == nullptr ||
        dtypeOf(tensorTy.getElementType()).empty()) {
      auxOk = false;
      return;
    }
    promotions.emplace_back(o, entryInputs[arg.getArgNumber()]);
    return;
  }
  if (siteOps.contains(def)) return;
  for (Operation *ancestor = def->getParentOp(); ancestor != nullptr;
       ancestor = ancestor->getParentOp())
    if (siteOps.contains(ancestor)) return;
  StringRef ns =
      def->getDialect() ? def->getDialect()->getNamespace() : StringRef();
  if (!isa<stablehlo::ConstantOp>(def) && ns != "shape" && ns != "arith" &&
      ns != "tensor") {
    if (canInternalize(def, 0)) {
      auxSet.insert(def);
      siteOps.insert(def);
      work.push_back(def);
      return;
    }
    auto tensorTy = dyn_cast<RankedTensorType>(o.getType());
    std::string anchor = anchorFromLoc(def->getLoc());
    if (tensorTy == nullptr || anchor.empty() ||
        dtypeOf(tensorTy.getElementType()).empty()) {
      auxOk = false;
      return;
    }
    if (promotedVals.insert(o).second) promotions.emplace_back(o, anchor);
    return;
  }
  auxSet.insert(def);
  siteOps.insert(def);
  work.push_back(def);
}

// Stay set: ops with out-of-site users (plus their in-site dependency
// chains) are cloned into the kernel; originals stay in main.
static SmallPtrSet<Operation *, 32> collectStaySet(
    const SmallPtrSet<Operation *, 32> &siteOps,
    const SmallPtrSet<Operation *, 2> &tailSet) {
  SmallPtrSet<Operation *, 32> stays;
  SmallVector<Operation *, 8> work;
  for (Operation *op : siteOps) {
    if (tailSet.contains(op)) continue;
    for (Value r : op->getResults())
      for (Operation *u : r.getUsers())
        if (!siteOps.contains(u)) {
          stays.insert(op);
          work.push_back(op);
          break;
        }
  }
  while (!work.empty()) {
    Operation *op = work.pop_back_val();
    for (Value o : op->getOperands()) {
      Operation *def = o.getDefiningOp();
      if (def != nullptr && siteOps.contains(def) && !stays.contains(def)) {
        stays.insert(def);
        work.push_back(def);
      }
    }
  }
  return stays;
}

// Attaches fusion.metadata + the top-level prune-contract attrs and prints
// the probe log. Pure assembly over already-built values.
static void attachFusionMetadata(PatternRewriter &rewriter, func::FuncOp fn,
                                 ArrayRef<ProbeArg> args,
                                 ArrayRef<Value> outputs,
                                 ArrayRef<std::string> tailAnchors,
                                 ArrayRef<Operation *> members,
                                 const std::string &name,
                                 StringRef patternName,
                                 ArrayRef<int64_t> contextI64Attrs,
                                 ArrayRef<std::string> outputAliases) {
  SmallVector<Attribute> argAttrs;
  for (const ProbeArg &arg : args)
    argAttrs.push_back(probeArgAttr(rewriter, arg));
  SmallVector<Attribute> outputAttrs;
  for (auto [out, anchor] : llvm::zip(outputs, tailAnchors)) {
    auto ty = cast<RankedTensorType>(out.getType());
    outputAttrs.push_back(DictionaryAttr::get(
        rewriter.getContext(),
        {rewriter.getNamedAttr("tf_name", rewriter.getStringAttr(anchor)),
         rewriter.getNamedAttr("role", rewriter.getStringAttr("output")),
         rewriter.getNamedAttr("shape",
                               rewriter.getI64ArrayAttr(ty.getShape())),
         rewriter.getNamedAttr("rank",
                               rewriter.getI64IntegerAttr(ty.getRank())),
         rewriter.getNamedAttr(
             "dtype", rewriter.getStringAttr(dtypeOf(ty.getElementType())))}));
  }

  SmallVector<NamedAttribute, 16> metadataAttrs{
      rewriter.getNamedAttr("tf.name", rewriter.getStringAttr(name)),
      rewriter.getNamedAttr("fusion.pattern",
                            rewriter.getStringAttr(patternName)),
      rewriter.getNamedAttr("kernel_name", rewriter.getStringAttr(name)),
      rewriter.getNamedAttr("execution_mode", rewriter.getStringAttr("aot")),
      rewriter.getNamedAttr("abi", rewriter.getStringAttr("annc_context_v3")),
      rewriter.getNamedAttr("args", rewriter.getArrayAttr(argAttrs)),
      rewriter.getNamedAttr("outputs", rewriter.getArrayAttr(outputAttrs)),
      // No GEMM-style dynamic-dim marker here: real dynamic dims surface
      // as -1 in the arg/output shapes; the field is read but not consumed
      // by the host op and passed through verbatim by the converter.
      rewriter.getNamedAttr("dynamic_dims", rewriter.getI64ArrayAttr({})),
      rewriter.getNamedAttr("fallback_function",
                            rewriter.getStringAttr("original_subgraph")),
      rewriter.getNamedAttr("symbolic_signature", rewriter.getStringAttr(""))};
  if (!contextI64Attrs.empty())
    metadataAttrs.push_back(rewriter.getNamedAttr(
        "context_i64_attrs", rewriter.getI64ArrayAttr(contextI64Attrs)));
  if (!outputAliases.empty()) {
    SmallVector<Attribute> aliasAttrs;
    for (const auto &alias : outputAliases)
      aliasAttrs.push_back(rewriter.getStringAttr(alias));
    metadataAttrs.push_back(rewriter.getNamedAttr(
        "output_aliases", rewriter.getArrayAttr(aliasAttrs)));
  }
  fn->setAttr("fusion.metadata",
              DictionaryAttr::get(rewriter.getContext(), metadataAttrs));
  fn->setAttr("fusion.pattern", rewriter.getStringAttr(patternName));

  std::string membersStr, anchorsStr;
  llvm::raw_string_ostream membersOs(membersStr);
  llvm::interleaveComma(members, membersOs, [&](Operation *op) {
    membersOs << op->getName().getStringRef();
  });
  membersOs.flush();
  llvm::raw_string_ostream anchorsOs(anchorsStr);
  llvm::interleaveComma(tailAnchors, anchorsOs);
  anchorsOs.flush();
  ANNC_LOG_DEBUG("op-fusion-v2")
      << patternName.str() << " probe matched: members=" << membersStr
      << " output_anchor=" << anchorsStr << "\n";
}

LogicalResult outlineSite(func::FuncOp main, PatternRewriter &rewriter,
                          SmallVectorImpl<ProbeArg> &args,
                          ArrayRef<Operation *> members,
                          ArrayRef<Operation *> tailOps,
                          ArrayRef<Value> outputs, const std::string &name,
                          StringRef patternName,
                          ArrayRef<std::string> tailAnchors,
                          ArrayRef<int64_t> contextI64Attrs,
                          CallPortTable &portTable,
                          ArrayRef<std::string> outputAliases) {
  ModuleOp module =
      validateSiteInputs(main, tailOps, outputs, tailAnchors);
  if (!module) return failure();

  SmallVector<Value> argValues;
  for (const ProbeArg &arg : args) argValues.push_back(arg.value);
  AuxClosure aux(main, members, argValues);
  if (!aux.auxOk) {
    tailOps.front()->emitWarning()
        << "[op-fusion-v2] site " << name
        << ": aux closure invalid; site dropped";
    return failure();
  }

  SmallVector<Type> tailTypes;
  for (Value out : outputs) tailTypes.push_back(out.getType());
  SmallVector<Type> argTypes;
  for (const ProbeArg &arg : args) argTypes.push_back(arg.type);
  for (auto &[value, tfName] : aux.promotions) {
    auto ty = dyn_cast<RankedTensorType>(value.getType());
    if (!ty) return failure();
    argTypes.push_back(ty);
  }
  rewriter.setInsertionPointToEnd(module.getBody());
  auto fn = rewriter.create<func::FuncOp>(
      module.getLoc(), name,
      FunctionType::get(rewriter.getContext(), argTypes, tailTypes));
  fn.setPrivate();
  fn->setAttr("annc.kernel", UnitAttr::get(rewriter.getContext()));
  fn->setAttr("annc.execution_mode", rewriter.getStringAttr("aot"));
  fn->setAttr("llvm.emit_c_interface", UnitAttr::get(rewriter.getContext()));
  Block *body = fn.addEntryBlock();
  IRMapping mapping;
  unsigned argIdx = 0;
  for (const ProbeArg &arg : args)
    mapping.map(arg.value, body->getArgument(argIdx++));
  for (auto &[value, tfName] : aux.promotions) {
    ProbeArg extra;
    extra.type = cast<RankedTensorType>(value.getType());
    extra.value = value;
    extra.tfName = tfName;
    extra.role = "dynamic";
    args.push_back(extra);
    mapping.map(value, body->getArgument(argIdx++));
  }

  SmallPtrSet<Operation *, 2> tailSet(tailOps.begin(), tailOps.end());
  SmallPtrSet<Operation *, 32> stays = collectStaySet(aux.siteOps, tailSet);

  // Outputs whose host-side users precede the tails cannot take call results
  // (the call needs those very chains as arguments); keep their producers in
  // the host via a stays clone and leave the uses as-is.
  SmallPtrSet<Value, 2> hostKept;
  {
    Block *hostBlock = tailOps.front()->getBlock();
    for (Value out : outputs) {
      for (Operation *user : out.getUsers()) {
        if (user->getBlock() != hostBlock) continue;
        if (!user->isBeforeInBlock(tailOps.front())) continue;
        hostKept.insert(out);
        if (Operation *def = out.getDefiningOp()) stays.insert(def);
        break;
      }
    }
  }

  SmallPtrSet<Operation *, 32> emittedM;
  SmallPtrSet<Operation *, 16> emittedA;
  std::function<LogicalResult(Operation *)> emitOp;
  std::function<LogicalResult(Value)> emitValue =
      [&](Value o) -> LogicalResult {
    if (mapping.contains(o)) return success();
    Operation *def = o.getDefiningOp();
    if (def == nullptr) return success();
    if (aux.siteOps.contains(def)) return emitOp(def);
    return success();
  };
  emitOp = [&](Operation *op) -> LogicalResult {
    if (llvm::is_contained(members, op)) {
      if (!emittedM.insert(op).second) return success();
    } else if (!emittedA.insert(op).second) {
      return success();
    }
    SmallVector<Value> deps(op->getOperands().begin(), op->getOperands().end());
    op->walk([&](Operation *inner) {
      if (inner == op) return;
      for (Value o : inner->getOperands()) deps.push_back(o);
    });
    for (Value o : deps)
      if (failed(emitValue(o))) return failure();
    if (stays.contains(op)) {
      rewriter.setInsertionPointToEnd(body);
      Operation *cloned = op->clone(mapping);
      rewriter.insert(cloned);
      for (auto [oldR, newR] :
           llvm::zip(op->getResults(), cloned->getResults()))
        mapping.map(oldR, newR);
    } else {
      rewriter.moveOpBefore(op, body, body->end());
    }
    return success();
  };

  SmallVector<Value> callArgs;
  for (const ProbeArg &arg : args) callArgs.push_back(arg.value);
  Block *hostBlock = tailOps.front()->getBlock();
  Operation *callBefore = tailOps.front();
  auto consider = [&](Operation *owner) {
    if (owner == callBefore || owner->getBlock() != hostBlock) return;
    if (owner->isBeforeInBlock(callBefore)) callBefore = owner;
  };
  for (Value out : outputs) {
    if (hostKept.contains(out)) continue;
    for (OpOperand &use : out.getUses()) consider(use.getOwner());
  }
  rewriter.setInsertionPoint(callBefore);
  auto call = rewriter.create<func::CallOp>(tailOps.front()->getLoc(), name,
                                            tailTypes, callArgs);
  portTable.entries[call.getOperation()] = {
      name, SmallVector<std::string>(tailAnchors)};
  for (Operation *tail : tailOps) {
    SmallVector<Value> tailDeps(tail->getOperands().begin(),
                                tail->getOperands().end());
    tail->walk([&](Operation *inner) {
      if (inner == tail) return;
      for (Value o : inner->getOperands()) tailDeps.push_back(o);
    });
    for (Value o : tailDeps) {
      if (failed(emitValue(o))) return failure();
    }
    // A tail may already have been emitted through another member's
    // dependency walk; re-emitting would insert a second clone (or a
    // duplicate move) and invert the definition order.
    if (!emittedM.insert(tail).second) continue;
    if (stays.contains(tail)) {
      // Host-kept tail: clone into the kernel, forward the results.
      rewriter.setInsertionPointToEnd(body);
      Operation *cloned = tail->clone(mapping);
      rewriter.insert(cloned);
      for (auto [oldR, newR] :
           llvm::zip(tail->getResults(), cloned->getResults()))
        mapping.map(oldR, newR);
    } else {
      rewriter.moveOpBefore(tail, body, body->end());
    }
  }
  rewriter.setInsertionPointToEnd(body);
  SmallVector<Value> returnVals;
  for (Value out : outputs) returnVals.push_back(mapping.lookupOrDefault(out));
  rewriter.create<func::ReturnOp>(tailOps.front()->getLoc(), returnVals);
  for (auto &it : mapping.getValueMap())
    rewriter.replaceUsesWithIf(it.first, it.second, [&](OpOperand &operand) {
      return operand.getOwner()->getParentOfType<func::FuncOp>() == fn;
    });
  for (auto [out, callResult] : llvm::zip(outputs, call->getResults())) {
    if (hostKept.contains(out)) continue;
    rewriter.replaceUsesWithIf(out, callResult, [&](OpOperand &operand) {
      return operand.getOwner()->getParentOfType<func::FuncOp>() != fn;
    });
  }

  // Defensive re-anchor: consumers can be rewritten after the call was
  // placed (greedy rounds re-entry), so re-anchor to the earliest in-block
  // result user. No-op when ordering is already correct.
  {
    Operation *before = nullptr;
    for (Value r : call->getResults())
      for (OpOperand &use : r.getUses()) {
        Operation *owner = use.getOwner();
        if (owner->getBlock() != hostBlock) continue;
        if (!before || owner->isBeforeInBlock(before)) before = owner;
      }
    if (before && before->isBeforeInBlock(call.getOperation()))
      call.getOperation()->moveBefore(before);
  }


  attachFusionMetadata(rewriter, fn, args, outputs, tailAnchors, members,
                       name, patternName, contextI64Attrs, outputAliases);

  return success();
}

}  // namespace fusionv2
}  // namespace atir
