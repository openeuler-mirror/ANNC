#include <functional>
#include <string>
#include <utility>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/FusionV2/CarrierMatchers.h"
#include "Dialect/Atir/Passes/FusionV2/FusionV2Outline.h"
#include "Dialect/Atir/Passes/FusionV2/Naming.h"
#include "Dialect/Atir/Passes/FusionV2/Patterns.h"
#include "Support/Log.h"
#include "Dialect/Atir/Passes/FusionV2/Provenance.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

// The atir dialect and the pass live in the top-level ::atir; fusionv2
// view references in this file go through the alias.
namespace fusionv2 = atir::fusionv2;
using namespace atir::fusionv2;

namespace atir {
namespace {

struct SparseSegmentReduceMatch {
  Operation *sliceOut;  // anchor: the slice_output def (reshape, i32 scalar)
  Operation *ssOp;      // atir.sparse_segment_{mean,sum}_v2
  int64_t combiner;     // 0=Sum, 1=Mean
  Type indicesType;     // kernel specialization source (I32/I64)
  SmallPtrSet<Operation *, 32> keysClosure;   // segment_ids allowlist closure
  SmallPtrSet<Operation *, 32> sliceClosure;  // ss->slice shape/slice closure
  SmallVector<Operation *, 16> members;       // keysClosure + sliceClosure + ss
  // Form A: begin_1 block argument; empty on Form B (synthesized from
  // begin1Static).
  Value begin1;
  int64_t begin1Static = -1;            // Form B: static tail slice start
  Operation *shapeTailSlice = nullptr;  // Form B: the start-extract slice
  Value beginStackConst;  // Form B: keys slice begin stack constant
  Operation *keysSliceOp = nullptr;  // Form A': keys real_dynamic_slice
  Value indices;  // the indices contract value (see the rewrite for naming)
  DenseSet<Operation *> beginChain;  // Form A': begin ops kept out of members
  Value keys;     // the keys contract value (carrier, block arg, or boundary)
  Value data;     // the data contract value (f32 2D)
  Value begin;    // Form A: the begin block arg (empty on Form B)
};

// Matches the V1 specialization name (kernel specializes by combiner and
// indices dtype, no attrs channel; see V1 getKpSparseSegmentReduceOpName);
// fast-codegen selects the registered builtin by this name.
static std::string sparseSegmentReduceKernelName(int64_t combiner,
                                                 Type indicesType) {
  auto tensorType = dyn_cast<RankedTensorType>(indicesType);
  std::string name = "KPFusedSparseSegmentReduce";
  name +=
      (tensorType && tensorType.getElementType().isInteger(64)) ? "I64" : "I32";
  name += (combiner == 1) ? "Mean" : "Sum";
  return name;
}

// Forward escape check: member results must stay in the closure or hit an
// exemption - constants, ssOp, or the site tail anchor sliceOut.
static bool closureInternal(const SmallPtrSetImpl<Operation *> &closure,
                            Operation *ssOp, Operation *sliceOut) {
  for (Operation *op : closure) {
    if (isa<stablehlo::ConstantOp>(op) || isa<arith::ConstantOp>(op)) continue;
    if (op == sliceOut) continue;  // the tail anchor is the site output
    for (Value r : op->getResults())
      for (Operation *u : r.getUsers())
        if (!closure.contains(u) && u != ssOp) return false;
  }
  return true;
}

// Anchor = the slice-scalar reshape; two allowlist closures fold backwards
// with dual-form boundaries, the site contract matching the V1 kernel:
// (keys, begin, data, indices, begin_1) -> (output, slice_output).
static bool matchStaticShapeTail(const SmallPtrSetImpl<Operation *> &closure,
                                 Operation *ssOp, int64_t &beginValue,
                                 Operation *&sliceOut) {
  Value ssResult = ssOp->getResult(0);
  Operation *hit = nullptr;
  int64_t start0 = -1;
  for (Operation *op : closure) {
    auto slice = dyn_cast<stablehlo::SliceOp>(op);
    if (!slice) continue;
    auto indexCast = slice.getOperand().getDefiningOp<arith::IndexCastOp>();
    if (!indexCast) continue;
    auto shapeOf = indexCast.getIn().getDefiningOp<shape::ShapeOfOp>();
    if (!shapeOf || shapeOf.getArg() != ssResult) continue;
    ArrayRef<int64_t> start = slice.getStartIndices();
    ArrayRef<int64_t> limit = slice.getLimitIndices();
    ArrayRef<int64_t> strides = slice.getStrides();
    if (start.size() != 1 || limit.size() != 1 || strides.size() != 1)
      return false;
    if (strides[0] != 1 || limit[0] != start[0] + 1) return false;
    if (hit && (start0 != start[0] || hit != op))
      return false;  // multiple candidates: pointer-order nondeterminism
    hit = op;
    start0 = start[0];
  }
  if (!hit) return false;
  beginValue = start0;
  sliceOut = hit;
  return true;
}

// Form-B keys chain: the batch-4 shrink removed the cast_v2/strided_slice_v2
// carriers, and the FE now leaves a `tf.StridedSlice` fallback residue
// (optionally convert-wrapped; bare i64 on cvr_4).
static bool matchCarrierKeysChain(Value segmentIds, Value &keys, Value &begin,
                                  SmallPtrSetImpl<Operation *> &closure) {
  auto why = [&](const char *r) {
    ANNC_LOG_DEBUG("op-fusion-v2")
        << "kp_sparse_segment_reduce: B-chain reject(" << r << ")\n";
    return false;
  };
  Value residueSrc = segmentIds;
  if (auto convert = segmentIds.getDefiningOp<stablehlo::ConvertOp>()) {
    residueSrc = convert.getOperand();
    closure.insert(convert.getOperation());
  } else {
    auto segTy = dyn_cast<RankedTensorType>(segmentIds.getType());
    if (!segTy || segTy.getRank() != 1 ||
        !segTy.getElementType().isSignlessInteger(64))
      return why("bare-form");
  }
  fusionv2::SliceLike slice =
      fusionv2::SliceLike::match(residueSrc.getDefiningOp());
  if (!slice) return why("slice-carrier");
  if (slice.isResidue()) {
    if (slice.op->getNumOperands() != 4)
      return why("residue-operands");  // input/begin/end/strides
    auto meta = slice.op->getAttrOfType<DictionaryAttr>("metadata");
    if (!meta) return why("residue-meta");
    auto maskIs = [&](llvm::StringRef name, int64_t want) {
      auto attr = meta.getAs<IntegerAttr>(name);
      return attr && attr.getValue().getSExtValue() == want;
    };
    if (!maskIs("shrink_axis_mask", 2) || !maskIs("begin_mask", 1) ||
        !maskIs("end_mask", 1) || !maskIs("ellipsis_mask", 0) ||
        !maskIs("new_axis_mask", 0))
      return why("residue-mask");
    // strides constant [1,1] (the TF full-dim stride).
    auto stridesDef =
        slice.op->getOperand(3).getDefiningOp<stablehlo::ConstantOp>();
    if (!stridesDef) return why("strides-def");
    auto stridesAttr = stridesDef.getValue().dyn_cast<DenseElementsAttr>();
    if (!stridesAttr || stridesAttr.getNumElements() != 2)
      return why("strides-shape");
    for (APInt v : stridesAttr.getValues<APInt>())
      if (v != 1) return why("strides-value");
  } else {
    if (isa<stablehlo::SliceOp>(slice.op)) return false;
    if (auto rds = dyn_cast<stablehlo::RealDynamicSliceOp>(slice.op)) {
      if (rds->getNumOperands() != 4) return false;
      auto stridesDef =
          rds->getOperand(3).getDefiningOp<stablehlo::ConstantOp>();
      if (!stridesDef) return false;
      auto stridesAttr = stridesDef.getValue().dyn_cast<DenseElementsAttr>();
      if (!stridesAttr || stridesAttr.getNumElements() != 2) return false;
      for (APInt v : stridesAttr.getValues<APInt>())
        if (v != 1) return false;
    } else if (!isa<stablehlo::DynamicSliceOp>(slice.op)) {
      return false;
    }
    // The result must be 1-D (the segment_ids contract).
    auto resTy = dyn_cast<RankedTensorType>(slice.result().getType());
    if (!resTy || resTy.getRank() != 1) return false;
  }
  begin = slice.start();
  auto beginDef = begin.getDefiningOp<stablehlo::ConstantOp>();
  if (!beginDef) return why("begin-def");
  auto beginTy = dyn_cast<RankedTensorType>(begin.getType());
  if (!beginTy || !beginTy.getElementType().isInteger(32) ||
      beginTy.getRank() != 1 || beginTy.getShape()[0] != 2)
    return why("begin-type");
  keys = slice.source();
  auto keysTy = dyn_cast<RankedTensorType>(keys.getType());
  if (!keysTy || !keysTy.getElementType().isInteger(64) ||
      keysTy.getRank() != 2)
    return why("keys-type");
  closure.insert(slice.op);
  return true;
}


// Shape/slice closure: boundaries converge to Form A {ss result, begin_1
// block arg} or Form B {ss result} with a static tail chain.
static bool matchShapeClosure(stablehlo::ReshapeOp anchor,
                              SparseSegmentReduceMatch &m, Operation *&ssOp,
                              Value &begin1, int64_t &begin1Static) {
  auto reject = [loc = anchor.getLoc()](const char *why) {
    std::string locStr;
    llvm::raw_string_ostream(locStr) << loc;
    ANNC_LOG_DEBUG("op-fusion-v2")
        << "kp_sparse_segment_reduce: reject(" << why << ") " << locStr
        << "\n";
    return false;
  };
  SmallPtrSet<Operation *, 32> sliceClosure;
  DenseSet<Value> sliceBoundary;
  if (!collectArithClosure(anchor.getResult(), sliceClosure, sliceBoundary,
                           {"atir.reshape"}))
    return reject("slice-closure");
  for (Value v : sliceBoundary) {
    if (llvm::isa<BlockArgument>(v)) {
      if (begin1) return reject("begin1-dup");
      begin1 = v;
      continue;
    }
    if (ssOp) return reject("ss-dup");
    ssOp = v.getDefiningOp();
  }
  if (!ssOp) return reject("ss-missing");
  if (begin1) {
    auto b1Ty = dyn_cast<RankedTensorType>(begin1.getType());
    if (!b1Ty || !b1Ty.getElementType().isInteger(32) || b1Ty.getRank() != 1 ||
        b1Ty.getShape()[0] != 1)
      return reject("begin1-dtype");
  } else if (!matchStaticShapeTail(sliceClosure, ssOp, begin1Static,
                                   m.shapeTailSlice)) {
    return reject("static-tail");
  }
  m.sliceClosure = std::move(sliceClosure);
  return true;
}

// Dispatches the ss op and records the data/indices/segment_ids/
// num_segments operand roles (combiner: 0=Sum, 1=Mean).
static bool dispatchSegmentOp(Operation *ssOp, int64_t &combiner,
                              Value &data, Value &indices, Value &segmentIds,
                              Value &numSegments) {
  if (auto mean = dyn_cast<atir::SparseSegmentMeanV2Op>(ssOp)) {
    combiner = 1;
    data = mean.getData();
    indices = mean.getIndices();
    segmentIds = mean.getSegmentIds();
    numSegments = mean.getNumSegments();
    return true;
  }
  if (auto sum = dyn_cast<atir::SparseSegmentSumV2Op>(ssOp)) {
    combiner = 0;
    data = sum.getInput();
    indices = sum.getIndices();
    segmentIds = sum.getSegmentIds();
    numSegments = sum.getNumSegments();
    return true;
  }
  return false;
}

// num_segments must be the constant 0; data f32 2D, indices i32/i64 1D.
static bool checkContract(Value numSegments, Value data, Value indices,
                          Type &indicesType) {
  Operation *numDef = numSegments.getDefiningOp();
  auto numSplat =
      numDef ? numDef->getAttrOfType<DenseElementsAttr>("value") : nullptr;
  if (!numSplat || !numSplat.isSplat() ||
      !numSplat.getSplatValue<APInt>().isZero())
    return false;
  auto dataTy = dyn_cast<RankedTensorType>(data.getType());
  if (!dataTy || !dataTy.getElementType().isF32() || dataTy.getRank() != 2)
    return false;
  auto idxTy = dyn_cast<RankedTensorType>(indices.getType());
  if (!idxTy || idxTy.getRank() != 1 ||
      !(idxTy.getElementType().isInteger(32) ||
        idxTy.getElementType().isInteger(64)))
    return false;
  indicesType = idxTy;
  return true;
}

// Form A/A' over the arithmetic keys closure: exactly one keys-consuming
// real_dynamic_slice with a column-width-1 result; the FE begin tag names
// the runtime begin stack.
static bool matchArithmeticKeys(SmallPtrSetImpl<Operation *> &keysClosure,
                                DenseSet<Value> &keysBoundary, Value &keys,
                                Value &begin, Value &beginStackConst,
                                Operation *&keysSliceOp) {
  bool boundaryOk = true;
  for (Value v : keysBoundary) {
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    bool isI64R2 =
        ty && ty.getRank() == 2 && ty.getElementType().isInteger(64);
    bool isI32V2 = ty && ty.getRank() == 1 &&
                   ty.getElementType().isInteger(32) && ty.getShape()[0] == 2;
    if (isI64R2) {
      if (keys) return false;  // duplicate keys boundary
      keys = v;
    } else if (isI32V2 && llvm::isa<BlockArgument>(v)) {
      if (begin) return false;  // duplicate begin boundary
      begin = v;
    } else {
      return false;
    }
  }
  auto keysTy = dyn_cast<RankedTensorType>(keys.getType());
  if (!keysTy || !keysTy.getElementType().isInteger(64) ||
      keysTy.getRank() != 2)
    return false;
  if (begin) {
    auto beginTy = dyn_cast<RankedTensorType>(begin.getType());
    if (!beginTy || !beginTy.getElementType().isInteger(32) ||
        beginTy.getRank() != 1 || beginTy.getShape()[0] != 2)
      return false;
  }
  bool slicesKeys = false;
  int keysRdsCount = 0;
  for (Operation *op : keysClosure) {
    if (isa<stablehlo::DynamicSliceOp>(op) &&
        llvm::is_contained(op->getOperands(), keys))
      slicesKeys = true;
    if (auto rds = dyn_cast<stablehlo::RealDynamicSliceOp>(op)) {
      if (!llvm::is_contained(rds.getOperands(), keys)) continue;
      auto resTy = dyn_cast<RankedTensorType>(rds.getType());
      if (!resTy || resTy.getRank() != 2 || resTy.isDynamicDim(1) ||
          resTy.getDimSize(1) != 1)
        continue;
      slicesKeys = true;
      keysSliceOp = op;
      ++keysRdsCount;
    }
  }
  if (keysRdsCount > 1 || !slicesKeys) return false;
  if (keysRdsCount == 1) {
    // Form A': the FE begin tag names the runtime begin stack.
    if (!keysSliceOp->hasAttr("annc.begin_source")) return false;
    beginStackConst = keysSliceOp->getOperand(1);
  }
  return true;  // Form A (block-arg begin) or Form A'
}

// The Form A' begin stack stays in main: as a member, outline would move
// the chain into the kernel while args still reference the main-graph
// value, which the region-isolation verifier rejects (reproduced).
static void excludeBeginChain(SparseSegmentReduceMatch &m) {
  DenseSet<Operation *> beginChain;
  if (!m.beginStackConst) return;
  auto def = m.beginStackConst.getDefiningOp();
  if (!def) return;
  SmallVector<Operation *, 8> work{def};
  while (!work.empty()) {
    Operation *op = work.pop_back_val();
    if (!m.keysClosure.contains(op) || !beginChain.insert(op).second)
      continue;
    for (Value o : op->getOperands())
      if (auto d = o.getDefiningOp())
        if (m.keysClosure.contains(d)) work.push_back(d);
  }
  m.beginChain = std::move(beginChain);
}

LogicalResult matchKpSparseSegmentReduce(stablehlo::ReshapeOp anchor,
                                         SparseSegmentReduceMatch &m) {
  std::string locStr;
  llvm::raw_string_ostream(locStr) << anchor.getLoc();
  auto reject = [&](const char *why) {
    ANNC_LOG_DEBUG("op-fusion-v2")
        << "kp_sparse_segment_reduce: reject(" << why << ") " << locStr
        << "\n";
    return failure();
  };
  auto outTy = dyn_cast<RankedTensorType>(anchor.getResult().getType());
  if (!outTy || !outTy.getElementType().isInteger(32) || outTy.getRank() != 0) {
    return reject("anchor-type");
  }

  Operation *ssOp = nullptr;
  Value begin1 = nullptr;
  if (!matchShapeClosure(anchor, m, ssOp, begin1, m.begin1Static))
    return failure();

  int64_t combiner = -1;
  Value data, indices, segmentIds, numSegments;
  if (!dispatchSegmentOp(ssOp, combiner, data, indices, segmentIds,
                         numSegments))
    return reject("ss-dispatch");

  if (!checkContract(numSegments, data, indices, m.indicesType))
    return reject("contract");

  SmallPtrSet<Operation *, 32> keysClosure;
  DenseSet<Value> keysBoundary;
  Value keys, begin, beginStackConst;
  Operation *keysSliceOp = nullptr;
  if (collectArithClosure(segmentIds, keysClosure, keysBoundary, {}) &&
      matchArithmeticKeys(keysClosure, keysBoundary, keys, begin,
                          beginStackConst, keysSliceOp)) {
    // Form A/A' matched.
  } else if (matchCarrierKeysChain(segmentIds, keys, begin, keysClosure)) {
    beginStackConst = begin;  // Form B
  } else {
    return reject("keys-chain");
  }
  m.begin1 = begin1;
  m.beginStackConst = beginStackConst;
  m.keysSliceOp = keysSliceOp;
  if (!closureInternal(keysClosure, ssOp, anchor.getOperation()) ||
      !closureInternal(m.sliceClosure, ssOp, anchor.getOperation()))
    return reject("closure-escape");

  m.sliceOut = anchor.getOperation();
  m.keys = keys;
  m.data = data;
  m.begin = begin;
  m.ssOp = ssOp;
  m.combiner = combiner;
  m.indices = indices;
  m.keysClosure = std::move(keysClosure);
  excludeBeginChain(m);
  // Members are assembled in func-body order: metadata/args order is
  // deterministic (SmallPtrSet iteration order wobbles with pointer
  // addresses and would infect metadata stability).
  auto func = anchor->getParentOfType<func::FuncOp>();
  for (Operation &op : func.getBody().front().getOperations())
    if ((m.keysClosure.contains(&op) || m.sliceClosure.contains(&op) ||
         &op == ssOp) &&
        !m.beginChain.contains(&op))
      m.members.push_back(&op);
  return success();
}

class FuseKpSparseSegmentReduceV2Pattern
    : public OpRewritePattern<stablehlo::ReshapeOp> {
 public:
  FuseKpSparseSegmentReduceV2Pattern(
      MLIRContext *ctx, fusionv2::CallPortTable *portTableArg)
      : OpRewritePattern(ctx), portTable(portTableArg) {
    assert(portTable && "kp_sparse_segment_reduce requires a port table");
  }

  LogicalResult orderSegmentReduceArgs(
      SmallVectorImpl<ProbeArg> &args, const SparseSegmentReduceMatch &m,
      Value begin1Arg) const {
    if (args.size() < 5) return failure();
    SmallVector<ProbeArg> ordered;
    auto take = [&](Value want) -> LogicalResult {
      for (auto it = args.begin(); it != args.end(); ++it) {
        if (it->value != want) continue;
        ordered.push_back(*it);
        args.erase(it);
        return success();
      }
      return failure();
    };
    Value begin = m.beginStackConst ? Value(m.beginStackConst) : Value(m.begin);
    if (failed(take(m.keys))) return failure();      // keys
    if (failed(take(begin))) return failure();       // begin
    if (failed(take(m.data))) return failure();      // data
    if (failed(take(m.indices))) return failure();   // indices
    if (failed(take(begin1Arg))) return failure();   // begin_1
    args = std::move(ordered);
    return success();
  }

  LogicalResult matchAndRewrite(stablehlo::ReshapeOp anchor,
                                PatternRewriter &rewriter) const override {
    SparseSegmentReduceMatch m;
    if (failed(matchKpSparseSegmentReduce(anchor, m))) return failure();

    func::FuncOp main = anchor->getParentOfType<func::FuncOp>();
    Operation *tailOps[] = {m.ssOp,
                            m.sliceOut};  // V1 order: output, slice_output
    Value outputs[] = {m.ssOp->getResult(0), m.sliceOut->getResult(0)};

    // begin/begin_1: block args on Form A; on Form B promoted/synthesized
    // constants keeping the V1 5-arg ABI. Synthesis failures clean up
    // uniformly without leaving dead code.
    SmallVector<std::pair<Value, std::string>> promoted;
    Value begin1Arg = m.begin1;
    arith::ConstantOp synthesizedBegin1;
    if (!begin1Arg) {
      // The synthesized begin_1 must name a real pb node; prefer the
      // annc.begin_source tag, falling back to the anchor spelling.
      std::string stackName;
      if (m.shapeTailSlice)
        if (auto src = m.shapeTailSlice->getAttrOfType<StringAttr>(
                "annc.begin_source"))
          stackName = src.getValue().str();
      if (stackName.empty())
        stackName = fusionv2::anchorFromLoc(anchor.getLoc()) + "_stack";
      auto b1Type = RankedTensorType::get({1}, rewriter.getI32Type());
      rewriter.setInsertionPoint(m.ssOp);  // must dominate the call site
      synthesizedBegin1 = rewriter.create<arith::ConstantOp>(
          NameLoc::get(rewriter.getStringAttr(stackName)),
          DenseElementsAttr::get(
              b1Type, ArrayRef<int32_t>{static_cast<int32_t>(m.begin1Static)}));
      begin1Arg = synthesizedBegin1.getResult();
      promoted.emplace_back(begin1Arg, stackName);
    }
    if (m.beginStackConst) {
      // Name source priority: annc.begin_source (CSE-immune) beats the
      // loc anchor name; the synthesized fallback gets dropped by the
      // converter (inputResolves), see the KNOWN GAP note.
      std::string beginName;
      if (m.keysSliceOp)
        if (auto src =
                m.keysSliceOp->getAttrOfType<StringAttr>("annc.begin_source"))
          beginName = src.getValue().str();
      if (beginName.empty())
        beginName = fusionv2::anchorFromLoc(
            m.beginStackConst.getDefiningOp()->getLoc());
      if (beginName.empty())
        beginName = fusionv2::anchorFromLoc(anchor.getLoc()) + "_begin";
      promoted.emplace_back(m.beginStackConst, beginName);
    }
    if (!isa<BlockArgument>(m.indices)) {
      // indices rides port 1 of a multi-result carrier (Unique:1); an auto
      // name would wire :0 (deduped values) into the segment indices port.
      // Producer provenance (source_node + IR port) wins over loc names.
      auto idxRes = llvm::dyn_cast<OpResult>(m.indices);
      std::string idxName = fusionv2::sourceNameOf(m.indices);
      if (idxName.empty())
        idxName =
            fusionv2::anchorFromLoc(m.indices.getDefiningOp()->getLoc());
      if (idxRes && !idxName.empty() && idxRes.getResultNumber() != 0)
        promoted.emplace_back(
            m.indices,
            idxName + ":" + std::to_string(idxRes.getResultNumber()));
    }

    SmallVector<ProbeArg> args;
    if (failed(collectBoundary(main, m.members, args, promoted))) {
      ANNC_LOG_DEBUG("op-fusion-v2")
          << "kp_sparse_segment_reduce: collect-boundary failed\n";
      if (synthesizedBegin1) rewriter.eraseOp(synthesizedBegin1);
      return failure();
    }
    SmallVector<std::string> tailAnchors;
    for (Value out : outputs)
      tailAnchors.push_back(fusionv2::anchorFromLoc(out.getLoc()));
    if (failed(orderSegmentReduceArgs(args, m, begin1Arg))) {
      ANNC_LOG_DEBUG("op-fusion-v2")
          << "kp_sparse_segment_reduce: order-args failed\n";
      if (synthesizedBegin1) rewriter.eraseOp(synthesizedBegin1);
      return failure();
    }

    auto module = anchor->getParentOfType<mlir::ModuleOp>();
    std::string name = uniquifySymbolName(
        module, sparseSegmentReduceKernelName(m.combiner, m.indicesType));
    if (failed(outlineSite(main, rewriter, args, m.members, tailOps, outputs,
                           name, "kp_sparse_segment_reduce", tailAnchors, {},
                           *portTable))) {
      ANNC_LOG_DEBUG("op-fusion-v2")
          << "kp_sparse_segment_reduce: outline failed\n";
      if (synthesizedBegin1) rewriter.eraseOp(synthesizedBegin1);
      return failure();
    }
    return success();
  }

 private:
  fusionv2::CallPortTable *portTable;
};

}  // namespace

void addSparseSegmentReduceV2Pattern(RewritePatternSet &patterns,
                                     fusionv2::CallPortTable *portTable) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<FuseKpSparseSegmentReduceV2Pattern>(ctx, portTable);
}
}  // namespace atir
