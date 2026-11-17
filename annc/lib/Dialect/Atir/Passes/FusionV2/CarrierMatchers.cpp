#include "Dialect/Atir/Passes/FusionV2/CarrierMatchers.h"

#include "Dialect/Atir/AtirOps.h"
#include "llvm/ADT/APInt.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace atir {
namespace fusionv2 {

// Matches a reshape over the carriers the current frontend emits: static
// stablehlo.reshape (target shape in the result type) and the batch-4
// residue fallback (opType="Reshape", operand 1 = the runtime shape).
ReshapeLike ReshapeLike::match(Operation *op) {
  ReshapeLike view;
  if (op && isa<stablehlo::ReshapeOp>(op)) {
    view.op = op;
    return view;
  }
  if (auto residue = dyn_cast_or_null<atir::UnsupportedOp>(op))
    if (residue.getOpType() == "Reshape" && op->getNumOperands() == 2)
      view.op = op;
  return view;
}

Value ReshapeLike::src() const { return op->getOperand(0); }

Value ReshapeLike::targetShape() const {
  if (isa<stablehlo::ReshapeOp>(op)) return Value();
  return op->getOperand(1);
}

namespace {

// Value of a single-element integer constant; the def is checked by dialect
// namespace (op-class isa<> would pull stablehlo TypeID symbols into this
// library's link interface). arith.constant scalars carry IntegerAttr.
std::optional<int64_t> constantIntScalar(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def) return std::nullopt;
  StringRef ns =
      def->getDialect() ? def->getDialect()->getNamespace() : StringRef();
  if (ns != "stablehlo" && ns != "arith") return std::nullopt;
  if (auto attr = dyn_cast_or_null<DenseElementsAttr>(def->getAttr("value"))) {
    if (attr.getNumElements() != 1) return std::nullopt;
    if (!attr.getElementType().isIntOrIndex()) return std::nullopt;
    return (*attr.getValues<APInt>().begin()).getSExtValue();
  }
  if (auto attr = dyn_cast_or_null<IntegerAttr>(def->getAttr("value")))
    return attr.getInt();
  return std::nullopt;
}

bool isZeroIntConstant(Value v) {
  auto c = constantIntScalar(v);
  return c && *c == 0;
}

}  // namespace

// torch_index_select (2 operands, axis as attrs) or residue GatherV2 (axis
// as the 3rd operand); the kernel contract fixes axis=0, non-zero or
// non-constant is rejected here.
GatherLike GatherLike::match(Operation *op) {
  GatherLike view;
  if (!op) return view;
  if (isa<stablehlo::TorchIndexSelectOp>(op) && op->getNumOperands() == 2) {
    view.op = op;
    return view;
  }
  if (auto residue = dyn_cast_or_null<atir::UnsupportedOp>(op))
    if (residue.getOpType() == "GatherV2" && op->getNumOperands() == 3 &&
        isZeroIntConstant(op->getOperand(2)))
      view.op = op;
  return view;
}

Value GatherLike::params() const { return op->getOperand(0); }

Value GatherLike::indices() const { return op->getOperand(1); }

bool GatherLike::axisIsZero() const {
  if (auto tis = dyn_cast<stablehlo::TorchIndexSelectOp>(op))
    return tis.getDim() == 0 && tis.getBatchDims() == 0;
  auto axis = constantIntScalar(op->getOperand(2));
  return axis && *axis == 0;
}

// stablehlo slice family or residue StridedSlice (begin/end/strides as
// operands, masks as attrs).
SliceLike SliceLike::match(Operation *op) {
  SliceLike view;
  if (!op) return view;
  if (isa<stablehlo::SliceOp, stablehlo::RealDynamicSliceOp,
          stablehlo::DynamicSliceOp>(op)) {
    view.op = op;
    return view;
  }
  if (auto residue = dyn_cast_or_null<atir::UnsupportedOp>(op))
    if (residue.getOpType() == "StridedSlice" && op->getNumOperands() == 4)
      view.op = op;
  return view;
}

Value SliceLike::source() const { return op->getOperand(0); }

Value SliceLike::start() const {
  if (isa<stablehlo::SliceOp>(op)) return Value();
  return op->getNumOperands() > 1 ? op->getOperand(1) : Value();
}

bool SliceLike::isResidue() const { return isa<atir::UnsupportedOp>(op); }

// stablehlo.concatenate or residue ConcatV2 (axis as the trailing operand,
// at least 3 operands: 2 values + axis).
ConcatLike ConcatLike::match(Operation *op) {
  ConcatLike view;
  if (!op) return view;
  if (isa<stablehlo::ConcatenateOp>(op)) {
    view.op = op;
    return view;
  }
  if (auto residue = dyn_cast_or_null<atir::UnsupportedOp>(op))
    if (residue.getOpType() == "ConcatV2" && op->getNumOperands() >= 3)
      view.op = op;
  return view;
}

SmallVector<Value, 4> ConcatLike::inputs() const {
  SmallVector<Value, 4> values;
  unsigned n = op->getNumOperands();
  if (isa<atir::UnsupportedOp>(op) && n > 0) --n;
  for (unsigned i = 0; i < n; ++i) values.push_back(op->getOperand(i));
  return values;
}

std::optional<int64_t> ConcatLike::dimension() const {
  int64_t dim = 0;
  if (isa<stablehlo::ConcatenateOp>(op)) {
    auto attr = op->getAttrOfType<IntegerAttr>("dimension");
    if (!attr) return std::nullopt;
    dim = attr.getInt();
  } else {
    auto c = constantIntScalar(op->getOperand(op->getNumOperands() - 1));
    if (!c) return std::nullopt;
    dim = *c;
  }
  if (dim < 0) {
    auto ty = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!ty) return std::nullopt;
    dim += ty.getRank();
  }
  return dim;
}

FillLike FillLike::match(Operation *op) {
  FillLike view;
  if (!op) return view;
  if (isa<stablehlo::DynamicBroadcastInDimOp, stablehlo::BroadcastInDimOp>(
          op)) {
    view.op = op;
    return view;
  }
  if (auto residue = dyn_cast_or_null<atir::UnsupportedOp>(op))
    if (residue.getOpType() == "Fill" && op->getNumOperands() == 2)
      view.op = op;
  return view;
}

Value FillLike::value() const {
  return isa<atir::UnsupportedOp>(op) ? op->getOperand(1) : op->getOperand(0);
}

Value FillLike::shape() const {
  if (isa<atir::UnsupportedOp>(op)) return op->getOperand(0);
  if (op->getNumOperands() < 2) return Value();
  return op->getOperand(1);
}

namespace {

// Walk single-operand defs inside the group back to the first out-of-group
// value (piercing reshape/convert/cast wrappers); bail on multi-operand ops
// or overly deep walks - prefer giving up over guessing.
Value externalSource(Value v, ArrayRef<Operation *> members) {
  SmallPtrSet<Operation *, 16> group(members.begin(), members.end());
  for (int hop = 0; hop < 8; ++hop) {
    Operation *def = v.getDefiningOp();
    if (!def || !group.contains(def)) return v;
    if (def->getNumOperands() != 1) return Value();
    v = def->getOperand(0);
  }
  return Value();
}

// Splat integer constant value; empty for anything else.
std::optional<int64_t> splatIntConst(Value v) {
  auto k = v.getDefiningOp<stablehlo::ConstantOp>();
  if (!k) return std::nullopt;
  auto d = dyn_cast<DenseElementsAttr>(k.getValue());
  if (!d || !d.isSplat() || !d.getElementType().isInteger())
    return std::nullopt;
  return d.getSplatValue<APInt>().getSExtValue();
}

// Bucket port of a 1:N DP expansion; the bucket-size chain is
// divide(reduce(convert(compare EQ, sortedParts, broadcast k))), 1) - read k
// from the set_dimension_size size operand.
std::optional<int64_t> bucketPortOf(Value bucket) {
  Value v = bucket;
  while (auto cast = v.getDefiningOp<tensor::CastOp>()) v = cast.getSource();
  auto setDim = v.getDefiningOp<stablehlo::SetDimensionSizeOp>();
  if (!setDim) return std::nullopt;
  Value size = setDim->getOperand(1);
  if (auto div = size.getDefiningOp<stablehlo::DivOp>()) {
    auto one = splatIntConst(div->getOperand(1));
    if (!one || *one != 1) return std::nullopt;
    size = div->getOperand(0);
  }
  auto reduce = size.getDefiningOp<stablehlo::ReduceOp>();
  if (!reduce) return std::nullopt;
  auto init = splatIntConst(reduce->getOperand(1));
  if (!init || *init != 0) return std::nullopt;
  Block &rbody = reduce.getBody().front();
  if (&rbody.front() != rbody.getTerminator() ||
      !isa<stablehlo::AddOp>(rbody.front()))
    return std::nullopt;
  Value counted = reduce->getOperand(0);
  if (auto convert = counted.getDefiningOp<stablehlo::ConvertOp>())
    counted = convert->getOperand(0);
  auto compare = counted.getDefiningOp<stablehlo::CompareOp>();
  if (!compare ||
      compare.getComparisonDirection() != stablehlo::ComparisonDirection::EQ)
    return std::nullopt;
  // Exactly one side must trace to a splat integer constant: the other
  // side is the sorted-partitions data; both-constant or both-data is a
  // degenerate shape and yields nullopt.
  auto constOf = [](Value side) -> std::optional<int64_t> {
    Value c = side;
    if (auto bcast = c.getDefiningOp<stablehlo::BroadcastInDimOp>())
      c = bcast->getOperand(0);
    else if (auto bcast =
                 c.getDefiningOp<stablehlo::DynamicBroadcastInDimOp>())
      c = bcast->getOperand(0);
    return splatIntConst(c);
  };
  auto c0 = constOf(compare->getOperand(0));
  auto c1 = constOf(compare->getOperand(1));
  if (c0.has_value() == c1.has_value()) return std::nullopt;
  return c0 ? *c0 : *c1;
}

}  // namespace

// Residue form (result index = port) or the 1:N DP expansion: the expanded
// sort takes (partitions, data); buckets are escaping 1-D integer results
// with ports read from the counting chain, exactly 0..n-1.
PartitionLike PartitionLike::match(Operation *op) {
  PartitionLike view;
  if (!op) return view;
  if (auto r = dyn_cast<atir::UnsupportedOp>(op)) {
    if (r.getOpType() != "DynamicPartition" || op->getNumOperands() != 2)
      return PartitionLike();
    view.members = {op};
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      view.buckets.push_back(op->getResult(i));
    view.op = op;
    view.data = op->getOperand(0);
    view.partitions = op->getOperand(1);
    view.residue = true;
    return view;
  }
  if (semanticOf(op) != "tf.DynamicPartition") return PartitionLike();
  view.members = semanticGroup(op);
  if (view.members.empty()) return PartitionLike();
  view.op = op;

  Operation *sort = nullptr;
  for (Operation *member : view.members)
    if (isa<stablehlo::SortOp>(member)) {
      sort = member;
      break;
    }
  if (!sort || sort->getNumOperands() < 2) return PartitionLike();
  view.partitions = externalSource(sort->getOperand(0), view.members);
  view.data = externalSource(sort->getOperand(1), view.members);
  if (!view.partitions || !view.data) return PartitionLike();

  SmallPtrSet<Operation *, 16> group(view.members.begin(), view.members.end());
  SmallVector<std::pair<int64_t, Value>, 8> numbered;
  for (Operation *member : view.members) {
    for (Value r : member->getResults()) {
      auto ty = dyn_cast<RankedTensorType>(r.getType());
      if (!ty || ty.getRank() != 1 || !ty.getElementType().isInteger())
        continue;
      bool escapes = llvm::any_of(r.getUsers(), [&](Operation *user) {
        return !group.contains(user);
      });
      if (!escapes) continue;
      auto port = bucketPortOf(r);
      if (!port) return PartitionLike();
      numbered.emplace_back(*port, r);
    }
  }
  if (numbered.empty()) return PartitionLike();
  llvm::sort(numbered,
             [](const auto &a, const auto &b) { return a.first < b.first; });
  for (auto [i, entry] : llvm::enumerate(numbered)) {
    if (entry.first != static_cast<int64_t>(i)) return PartitionLike();
    view.buckets.push_back(entry.second);
  }
  return view;
}

// ---- Semantic groups (1:N expansion) ----

StringRef semanticOf(Operation *op) {
  if (!op) return {};
  if (auto a = op->getAttrOfType<StringAttr>("annc.source_op"))
    return a.getValue();
  return {};
}

// All ops sharing the (annc.source_node, annc.source_op) pair; groups are
// func-scoped, members never cross funcs.
SmallVector<Operation *, 8> semanticGroup(Operation *op) {
  SmallVector<Operation *, 8> out;
  if (!op) return out;
  auto src = op->getAttrOfType<StringAttr>("annc.source_node");
  auto sem = op->getAttrOfType<StringAttr>("annc.source_op");
  if (!src || !sem) return out;
  Operation *scope = op->getParentOp();
  while (scope && !isa<func::FuncOp>(scope)) scope = scope->getParentOp();
  if (!scope) return out;
  scope->walk([&](Operation *o) {
    auto s = o->getAttrOfType<StringAttr>("annc.source_node");
    auto m = o->getAttrOfType<StringAttr>("annc.source_op");
    if (s && m && s.getValue() == src.getValue() &&
        m.getValue() == sem.getValue())
      out.push_back(o);
  });
  return out;
}

// The member result tagged with the given port; source_ports may be spelled
// DenseI32ArrayAttr or DenseIntElementsAttr. Multiple producers on one port
// yield nullopt (conflict).
std::optional<Value> groupPortValue(Operation *op, int64_t port) {
  std::optional<Value> found;
  for (Operation *member : semanticGroup(op)) {
    SmallVector<int64_t, 4> ports;
    if (auto arr =
            member->getAttrOfType<DenseI32ArrayAttr>("annc.source_ports")) {
      for (int32_t p : arr.asArrayRef()) ports.push_back(p);
    } else if (auto dense = member->getAttrOfType<DenseIntElementsAttr>(
                   "annc.source_ports")) {
      for (const APInt &p : dense.getValues<APInt>())
        ports.push_back(p.getSExtValue());
    } else {
      continue;
    }
    for (auto [i, p] : llvm::enumerate(ports)) {
      if (p != port) continue;
      if (i >= member->getNumResults()) return std::nullopt;
      if (found) return std::nullopt;
      found = member->getResult(i);
    }
  }
  return found;
}

// Residue form (result index = port) or 1:N expansion (FE tags; indices
// untagged when nothing consumes it). Input x = the group's only
// non-constant outside operand; more than one bails out.
UniqueLike UniqueLike::match(Operation *op) {
  UniqueLike view;
  if (!op) return view;
  if (auto r = dyn_cast<atir::UnsupportedOp>(op)) {
    if (r.getOpType() != "Unique" || r->getNumOperands() != 1 ||
        r->getNumResults() != 2)
      return view;
    view.members = {op};
    view.values = r.getResult(0);
    view.indices = r.getResult(1);
    view.input = r.getOperand(0);
    view.residue = true;
    return view;
  }
  if (semanticOf(op) != "tf.Unique") return view;
  auto values = groupPortValue(op, 0);
  if (!values) return view;
  view.members = semanticGroup(op);
  view.values = *values;
  if (auto indices = groupPortValue(op, 1)) view.indices = *indices;
  std::optional<Value> external;
  for (Operation *member : view.members) {
    for (Value operand : member->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (def && llvm::is_contained(view.members, def)) continue;
      if (def && def->hasTrait<OpTrait::ConstantLike>()) continue;
      if (external && *external != operand) return UniqueLike();
      external = operand;
    }
  }
  if (!external) return UniqueLike();
  view.input = *external;
  return view;
}

}  // namespace fusionv2
}  // namespace atir
