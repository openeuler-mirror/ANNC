#include "common/pb2_common.h"

// Dynamic-input StridedSlice translation: tf2xla's hasStaticShape gates
// strand every ranked-dynamic instance as residue. Normalize the bounds
// (masks / negatives / clamping), emit real_dynamic_slice + shrink
// reshape; the undecidable skip with a bucketed reason. Runs before the
// carrier rename.

#include <algorithm>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace pb2 {
namespace {

using llvm::SmallVector;
using llvm::StringRef;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::RankedTensorType;
using mlir::Type;
using mlir::Value;

bool readIntConst(Value v, llvm::SmallVectorImpl<int64_t>& out) {
  Operation* def = v.getDefiningOp();
  if (def == nullptr) return false;
  auto dense = def->getAttrOfType<mlir::DenseIntElementsAttr>("value");
  if (!dense || !dense.getElementType().isSignlessInteger()) return false;
  auto ty = llvm::dyn_cast<RankedTensorType>(dense.getType());
  if (!ty || ty.getRank() != 1) return false;
  for (mlir::APInt v : dense.getValues<mlir::APInt>())
    out.push_back(v.getSExtValue());
  return true;
}

struct SliceDim {
  int64_t begin = 0;
  int64_t stride = 1;
  bool limitIsRuntime = false;  // limit needs get_dimension_size/minimum
  int64_t limitStatic = 0;      // valid when !limitIsRuntime
  bool limitToDimEnd = false;   // runtime limit is the raw dim size (no clamp)
  bool shrink = false;
};

class StridedSliceDyn {
 public:
  explicit StridedSliceDyn(OpBuilder& b) : b_(b) {}

  // Returns "" when translated, else the skip reason.
  std::string rewrite(Operation* op);

  int translated() const { return translated_; }

 private:
  Value emitI32Const(mlir::Location loc, int64_t v) {
    auto scalarTy = RankedTensorType::get({}, b_.getI32Type());
    OperationState st(loc, mlir::OperationName("stablehlo.constant",
                                               b_.getContext()));
    st.addTypes(scalarTy);
    st.addAttribute("value", mlir::DenseIntElementsAttr::get(
                                 scalarTy, llvm::ArrayRef<int64_t>{v}));
    return b_.create(st)->getResult(0);
  }

  Value emitDimSize(mlir::Location loc, Value v, int64_t dim) {
    OperationState st(
        loc, mlir::OperationName("stablehlo.get_dimension_size",
                                 b_.getContext()));
    st.addOperands(v);
    st.addTypes(RankedTensorType::get({}, b_.getI32Type()));
    st.addAttribute("dimension", b_.getI64IntegerAttr(dim));
    return b_.create(st)->getResult(0);
  }

  Value emitMinimumI32(mlir::Location loc, Value a, Value b) {
    OperationState st(loc, mlir::OperationName("stablehlo.minimum",
                                               b_.getContext()));
    st.addOperands({a, b});
    st.addTypes(RankedTensorType::get({}, b_.getI32Type()));
    return b_.create(st)->getResult(0);
  }

  // Vector assembly via reshape+concatenate, NOT tensor.from_elements: the
  // wheel's FromElementsOp::fold segfaults on non-Attribute operands (our
  // runtime scalars). The concatenate chain is canonicalizer-proven safe.
  Value emitI32Vector(mlir::Location loc, llvm::SmallVectorImpl<Value>& scalars) {
    Value vec;
    for (Value s : scalars) {
      OperationState rst(loc, mlir::OperationName("stablehlo.reshape",
                                               b_.getContext()));
      rst.addOperands(s);
      rst.addTypes(RankedTensorType::get({1}, b_.getI32Type()));
      Value r = b_.create(rst)->getResult(0);
      if (vec == nullptr) {
        vec = r;
        continue;
      }
      int64_t n = llvm::cast<RankedTensorType>(vec.getType()).getDimSize(0) + 1;
      OperationState cst(
          loc, mlir::OperationName("stablehlo.concatenate", b_.getContext()));
      cst.addOperands({vec, r});
      cst.addTypes(RankedTensorType::get({n}, b_.getI32Type()));
      cst.addAttribute("dimension", b_.getI64IntegerAttr(0));
      vec = b_.create(cst)->getResult(0);
    }
    return vec;
  }

  OpBuilder& b_;
  int translated_ = 0;
};

std::string StridedSliceDyn::rewrite(Operation* op) {
  if (op->getNumOperands() != 4 || op->getNumResults() != 1) return "arity";
  auto inTy = llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto resTy = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!inTy || !resTy) return "unranked";
  if (llvm::isa<mlir::tf_type::StringType>(inTy.getElementType()))
    return "string";  // string slices belong to the encode/decode transport
  if (!llvm::isa<mlir::tf_type::StringType>(resTy.getElementType()) &&
      resTy.getElementType() != inTy.getElementType())
    return "elem-mismatch";

  SmallVector<int64_t, 4> begins, ends, strides;
  if (!readIntConst(op->getOperand(1), begins)) return "begin-not-const";
  if (!readIntConst(op->getOperand(2), ends)) return "end-not-const";
  if (!readIntConst(op->getOperand(3), strides)) return "strides-not-const";
  const int64_t rank = inTy.getRank();
  if (rank == 0) return "rank0";  // empty start/limit/stride would crash
  // TF allows a spec shorter than the input rank: trailing dimensions are
  // taken in full (begin=0, end=dim, stride=1). Pad instead of rejecting.
  const int64_t specSize = static_cast<int64_t>(begins.size());
  if (ends.size() != begins.size() || strides.size() != begins.size() ||
      specSize > rank)
    return "spec-size";
  // "Taken to the dimension end" rides a parallel flag, not a sentinel in
  // ends: kDynamic overlaps TF's legal negative indices (x[0:-1]).
  llvm::SmallVector<bool, 4> takeToEnd(begins.size(), false);
  while ((int64_t)begins.size() < rank) {
    begins.push_back(0);
    ends.push_back(0);
    takeToEnd.push_back(true);
    strides.push_back(1);
  }

  auto maskOf = [&](llvm::StringRef name) -> int64_t {
    auto a = op->getAttrOfType<mlir::IntegerAttr>(name);
    return a ? a.getInt() : 0;
  };
  const int64_t beginMask = maskOf("begin_mask");
  const int64_t endMask = maskOf("end_mask");
  const int64_t shrinkMask = maskOf("shrink_axis_mask");
  if (maskOf("ellipsis_mask") != 0) return "ellipsis";
  if (maskOf("new_axis_mask") != 0) return "new-axis";

  SmallVector<SliceDim, 4> dims;
  dims.reserve(rank);
  for (int64_t d = 0; d < rank; ++d) {
    SliceDim dim;
    dim.begin = begins[d];
    dim.stride = strides[d];
    dim.shrink = (shrinkMask >> d) & 1;
    if (dim.stride <= 0) return "stride";
    const bool dimStatic = !inTy.isDynamicDim(d);
    const int64_t dimSize = inTy.getDimSize(d);
    if ((beginMask >> d) & 1) dim.begin = 0;
    if (dim.begin < 0) {
      if (!dimStatic) return "neg-begin-dynamic-dim";
      dim.begin = std::max(dim.begin + dimSize, int64_t{0});  // clamp
    }
    if (dim.shrink) {
      // A runtime-clamped start with a constant begin+1 limit can place
      // the window past the dimension end; shrink needs a static dim.
      if (!dimStatic) return "shrink-dynamic";
      if (dimSize <= 0) return "shrink-empty-dim";
      // Clamp begin into the dimension (x[10] on a dim of 4 is empty-ish
      // in TF; clamp keeps the [begin, begin+1] window in range).
      dim.begin = std::min(std::max(dim.begin, int64_t{0}), dimSize - 1);
      dim.limitStatic = dim.begin + 1;
      dims.push_back(dim);
      continue;
    }
    int64_t end = ends[d];
    const bool toEnd = takeToEnd[d] || ((endMask >> d) & 1);
    if (toEnd) {
      // Padded spec entry or end_mask: taken to the dimension end. begin
      // still needs the constant-end clamp, or x[10:] on a shorter dim
      // yields a negative span.
      if (dimStatic) {
        dim.begin = std::min(std::max(dim.begin, int64_t{0}), dimSize);
        dim.limitStatic = dimSize;
      } else {
        dim.limitIsRuntime = true;
        dim.limitToDimEnd = true;
      }
    } else if (end < 0) {
      if (!dimStatic) return "neg-end-dynamic-dim";
      // x[:-100] must be empty, not a negative span: normalize and clamp
      // exactly like the constant-end path below.
      dim.begin = std::min(std::max(dim.begin, int64_t{0}), dimSize);
      end = std::min(std::max(end + dimSize, int64_t{0}), dimSize);
      end = std::max(end, dim.begin);
      dim.limitStatic = end;
    } else if (dimStatic) {
      // Static clamp, mirroring the tf2xla constant-begin path.
      dim.begin = std::min(std::max(dim.begin, int64_t{0}), dimSize);
      end = std::min(std::max(end, dim.begin), dimSize);
      dim.limitStatic = end;
    } else {
      // Constant bound on a dynamic dimension: runtime clamp.
      dim.limitIsRuntime = true;
    }
    dims.push_back(dim);
  }

  mlir::Location loc = op->getLoc();
  b_.setInsertionPoint(op);

  // start / strides are pure constants; limits mix constants with
  // get_dimension_size / minimum for runtime bounds.
  SmallVector<Value, 4> startScalars, limitScalars, strideScalars;
  for (int64_t d = 0; d < rank; ++d) {
    Value in = op->getOperand(0);
    // One dim-size query per dynamic dim, shared by the begin clamp and
    // the runtime limit below.
    Value gds = inTy.isDynamicDim(d) ? emitDimSize(loc, in, d) : nullptr;
    // Constant begin on a dynamic dim must clamp at runtime, or a shorter
    // runtime dim would yield start > limit (undefined for the slice op).
    if (!inTy.isDynamicDim(d)) {
      startScalars.push_back(emitI32Const(loc, dims[d].begin));
    } else {
      startScalars.push_back(
          emitMinimumI32(loc, emitI32Const(loc, dims[d].begin), gds));
    }
    strideScalars.push_back(emitI32Const(loc, dims[d].stride));
    if (dims[d].limitIsRuntime) {
      if (!dims[d].shrink && !dims[d].limitToDimEnd) {
        // Constant end on a dynamic dim: clamp at runtime.
        gds = emitMinimumI32(loc, emitI32Const(loc, ends[d]), gds);
      }
      limitScalars.push_back(gds);
    } else {
      limitScalars.push_back(emitI32Const(loc, dims[d].limitStatic));
    }
  }
  Value startVec = emitI32Vector(loc, startScalars);
  Value limitVec = emitI32Vector(loc, limitScalars);
  Value strideVec = emitI32Vector(loc, strideScalars);

  SmallVector<int64_t, 4> midShape;
  midShape.reserve(rank);
  for (int64_t d = 0; d < rank; ++d) {
    if (dims[d].shrink) {
      midShape.push_back(1);
    } else if (dims[d].limitIsRuntime) {
      midShape.push_back(mlir::ShapedType::kDynamic);
    } else {
      int64_t span = dims[d].limitStatic - dims[d].begin;
      midShape.push_back((span + dims[d].stride - 1) / dims[d].stride);
    }
  }
  auto midTy = RankedTensorType::get(midShape, inTy.getElementType());

  OperationState sliceSt(
      loc, mlir::OperationName("stablehlo.real_dynamic_slice",
                               b_.getContext()));
  sliceSt.addOperands({op->getOperand(0), startVec, limitVec, strideVec});
  sliceSt.addTypes(midTy);
  Value sliced = b_.create(sliceSt)->getResult(0);

  Value result = sliced;
  if (shrinkMask != 0) {
    if (llvm::none_of(dims, [](const SliceDim& d) { return !d.shrink; })) {
      // Every dimension shrank (e.g. x[0] on a 1-D tensor): the target is a
      // scalar and the [1] -> scalar hop is fully static.
      OperationState reshapeSt(
          loc, mlir::OperationName("stablehlo.reshape", b_.getContext()));
      reshapeSt.addOperands(sliced);
      reshapeSt.addTypes(resTy);
      result = b_.create(reshapeSt)->getResult(0);
    } else {
      // Fold unit shrink dimensions away: target shape vector over the
      // non-shrink mid dimensions (runtime dims read off the slice result).
      SmallVector<Value, 4> outScalars;
      for (int64_t d = 0; d < rank; ++d) {
        if (dims[d].shrink) continue;
        if (midShape[d] == mlir::ShapedType::kDynamic)
          outScalars.push_back(emitDimSize(loc, sliced, d));
        else
          outScalars.push_back(emitI32Const(loc, midShape[d]));
      }
      Value outVec = emitI32Vector(loc, outScalars);
      OperationState reshapeSt(
          loc, mlir::OperationName("stablehlo.dynamic_reshape",
                                   b_.getContext()));
      reshapeSt.addOperands({sliced, outVec});
      reshapeSt.addTypes(resTy);
      result = b_.create(reshapeSt)->getResult(0);
    }
  }

  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  ++translated_;
  return "";
}

struct StridedSliceDynPass
    : public mlir::PassWrapper<StridedSliceDynPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StridedSliceDynPass)

  llvm::StringRef getArgument() const override { return "strided-slice-dyn"; }
  llvm::StringRef getDescription() const override {
    return "translate dynamic-input tf.StridedSlice with constant bounds to "
           "stablehlo.real_dynamic_slice (+ dynamic_reshape for shrink axes)";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    OpBuilder b(&getContext());
    StridedSliceDyn t(b);

    llvm::SmallVector<Operation*, 8> targets;
    module.walk([&](Operation* op) {
      llvm::StringRef nm = op->getName().getStringRef();
      if (nm == "tf.StridedSlice") {
        targets.push_back(op);
        return;
      }
      if (nm == "tf.residue" || nm == "annc.residue") {
        auto ct = op->getAttrOfType<mlir::StringAttr>("call_target_name");
        if (ct && ct.getValue() == "tf.StridedSlice") targets.push_back(op);
      }
    });

    int skipped = 0;
    llvm::StringMap<int> skipWhy;
    for (Operation* op : targets) {
      if (op->getBlock() == nullptr) continue;
      std::string why = t.rewrite(op);
      if (why.empty()) continue;
      ++skipped;
      ++skipWhy[why];
    }
    if (t.translated() || skipped) {
      llvm::errs() << "[strided-slice-dyn] translated: " << t.translated()
                   << " skipped: " << skipped << "\n";
      for (auto& w : skipWhy)
        llvm::errs() << "[strided-slice-dyn-skip] " << w.getKey().str() << ": "
                     << w.getValue() << "\n";
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createStridedSliceDynPass() {
  return std::make_unique<StridedSliceDynPass>();
}

}  // namespace pb2
