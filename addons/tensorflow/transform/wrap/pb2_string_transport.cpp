#include "common/pb2_common.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace pb2 {
namespace {

using llvm::StringRef;
using mlir::DenseElementsAttr;
using mlir::NamedAttribute;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::RankedTensorType;
using mlir::Type;
using mlir::Value;

bool inAllowlist(const std::string& cls) {
  static const std::set<std::string> kAllow = {
      "Identity", "StopGradient", "GuaranteeConst", "EnsureShape",
      "Reshape", "ExpandDims", "Squeeze", "Transpose", "Concat", 
      "ConcatV2", "Pack", "Pad", "Gather", "GatherV2", "GatherNd",       
      "Slice", "StridedSlice", "DynamicStitch",
      "ParallelDynamicStitch", "Select", "SelectV2",
  };
  return kAllow.count(cls) != 0;
}

bool dataOperandIndices(const std::string& cls, unsigned numOperands,
                        llvm::SmallVectorImpl<unsigned>& out) {
  if (!inAllowlist(cls)) {
    return false;
  }
  if (cls == "Concat") {
    for (unsigned i = 1; i < numOperands; ++i) {
      out.push_back(i);
    }
  } else if (cls == "ConcatV2") {
    for (unsigned i = 0; i + 1 < numOperands; ++i) {
      out.push_back(i);
    }
  } else if (cls == "DynamicStitch" || cls == "ParallelDynamicStitch") {
    // Layout is [indices_0..N-1, data_0..N-1], NOT interleaved pairs.
    const unsigned n = numOperands / 2;
    for (unsigned i = n; i < 2 * n && i < numOperands; ++i) {
      out.push_back(i);
    }
  } else if (cls == "Select" || cls == "SelectV2") {
    out.push_back(1);
    out.push_back(2);
  } else if (cls == "Pack") {
    for (unsigned i = 0; i < numOperands; ++i) {
      out.push_back(i);
    }
  } else {
    out.push_back(0);
  }
  return true;
}

bool isStringElem(Type t) { return llvm::isa<mlir::tf_type::StringType>(t); }

enum class DimAttrKind { kDimListConst, kAxisInputConst, kAxisAttr };
struct DimRule {
  const char* op;
  const char* attr;
  DimAttrKind kind;
};
constexpr DimRule kDimRules[] = {
    {"Transpose", "perm", DimAttrKind::kDimListConst},
    {"Slice", "begin", DimAttrKind::kDimListConst},
    {"Slice", "size", DimAttrKind::kDimListConst},
    {"Concat", "axis", DimAttrKind::kAxisInputConst},
    {"ConcatV2", "axis", DimAttrKind::kAxisInputConst},
    {"GatherV2", "axis", DimAttrKind::kAxisInputConst},
    {"ExpandDims", "axis", DimAttrKind::kAxisAttr},
    {"Squeeze", "squeeze_dims", DimAttrKind::kAxisAttr},
};

bool readIntConst(Value v, llvm::SmallVectorImpl<int64_t>& out) {
  Operation* def = v.getDefiningOp();
  if (def == nullptr) {
    return false;
  }
  auto dense = def->getAttrOfType<DenseElementsAttr>("value");
  if (!dense || !dense.getElementType().isSignlessInteger()) {
    return false;
  }
  const int64_t rank = dense.getType().getRank();
  if (rank > 1) {
    return false;
  }
  for (mlir::APInt v : dense.getValues<mlir::APInt>()) {
    out.push_back(v.getSExtValue());
  }
  return !out.empty();
}

int64_t origRank(Operation* op,
                 const llvm::SmallVectorImpl<unsigned>& dataIdx) {
  auto t =
      llvm::dyn_cast<mlir::TensorType>(op->getOperand(dataIdx[0]).getType());
  return (t && t.hasRank()) ? (int64_t)t.getRank() : -1;
}

bool applyDimAdapter(Operation* op, const std::string& cls,
                     const llvm::SmallVectorImpl<unsigned>& dataIdx) {
  mlir::OpBuilder b(op);
  for (const DimRule& r : kDimRules) {
    if (cls != r.op) {
      continue;
    }
    int64_t orank = origRank(op, dataIdx);
    if (r.kind == DimAttrKind::kAxisInputConst) {
      if (orank <= 0 || op->getNumOperands() == 0) {
        return true;
      }
      // Concat (v1) carries axis as the FIRST operand; ConcatV2/GatherV2
      // as a rank-0 scalar on the LAST one.
      const unsigned axisPos =
          cls == "Concat" ? 0 : op->getNumOperands() - 1;
      llvm::SmallVector<int64_t, 2> ax;
      if (!readIntConst(op->getOperand(axisPos), ax) || ax.size() != 1)
        return false;  // unknown axis: skip, never guess
      int64_t norm = ax[0] < 0 ? ax[0] + orank : ax[0];
      if (norm == orank - 1) {
        return false;
      }
      continue;
    }
    if (r.kind == DimAttrKind::kAxisAttr) {
      auto ia = op->getAttrOfType<mlir::IntegerAttr>(r.attr);
      if (!ia) {
        // Squeeze without squeeze_dims drops every unit dim -- a width-1
        // payload dim would be one of them.
        if (cls == "Squeeze") return false;
        continue;
      }
      int64_t axis = ia.getInt();
      if (orank <= 0) {
        continue;
      }
      int64_t norm = axis < 0 ? axis + orank : axis;
      if (norm == orank - 1) {
        return false;
      }
      continue;
    }
    int64_t append;
    llvm::StringRef a(r.attr);
    if (a == "perm") {
      append = orank;
    } else if (a == "begin") {
      append = 0;
    } else if (a == "size") {
      append = -1;
    } else
      return true;
    int pos = -1;
    if (a == "perm") {
      pos = 1;
    } else if (a == "begin") {
      pos = 1;
    } else if (a == "size") {
      pos = 2;
    } else
      return true;
    Value cv = op->getOperand(pos);
    llvm::SmallVector<int64_t, 8> vals;
    if (!readIntConst(cv, vals)) {
      return true;
    }
    vals.push_back(append);
    auto oldTy = llvm::cast<mlir::RankedTensorType>(cv.getType());
    auto newTy =
        RankedTensorType::get({(int64_t)vals.size()}, oldTy.getElementType());
    OperationState st(
        mlir::UnknownLoc::get(b.getContext()),
        mlir::OperationName("stablehlo.constant", b.getContext()));
    st.addTypes(newTy);
    st.addAttribute("value", mlir::DenseIntElementsAttr::get(
                                 newTy, llvm::ArrayRef<int64_t>(vals)));
    auto cst = b.create(st);
    op->setOperand(pos, cst->getResult(0));
  }
  return true;
}

bool isStringTensorValue(Type t) {
  auto tt = llvm::dyn_cast<mlir::TensorType>(t);
  return tt && isStringElem(tt.getElementType());
}

bool readStringConst(Value v, llvm::SmallVectorImpl<StringRef>& out) {
  Operation* def = v.getDefiningOp();
  if (def == nullptr) {
    return false;
  }
  auto dense = def->getAttrOfType<DenseElementsAttr>("value");
  if (!dense || !isStringElem(dense.getElementType())) {
    return false;
  }
  for (StringRef s : dense.getValues<StringRef>()) {
    out.push_back(s);
  }
  return true;
}

class StringTransport {
 public:
  explicit StringTransport(OpBuilder& b) : b_(b) {}

  bool rewrite(Operation* op, const std::string& cls);

  int transfers() const { return transfers_; }
  int folded() const { return folded_; }

 private:
  Value emitOp(StringRef mnemonic, mlir::Location loc, mlir::ValueRange ops,
               Type result,
               llvm::function_ref<void(OperationState&)> addAttrs = nullptr) {
    OperationState st(loc, mlir::OperationName(mnemonic, b_.getContext()));
    st.addOperands(ops);
    st.addTypes(result);
    if (addAttrs) {
      addAttrs(st);
    }
    return b_.create(st)->getResult(0);
  }

  mlir::Operation* emitBoundary(mlir::Location loc, mlir::ValueRange ops,
                                llvm::ArrayRef<Type> results,
                                StringRef target) {
    OperationState st(loc, mlir::OperationName("tf.residue", b_.getContext()));
    st.addOperands(ops);
    st.addTypes(results);
    st.addAttribute("call_target_name", b_.getStringAttr(target));
    return b_.create(st);
  }

  Value emitGetDimView(Value v) {
    auto flatTy = RankedTensorType::get(
        {mlir::ShapedType::kDynamic},
        llvm::cast<mlir::TensorType>(v.getType()).getElementType());
    OperationState st(mlir::UnknownLoc::get(b_.getContext()),
                      mlir::OperationName("tf.residue", b_.getContext()));
    st.addOperands(v);
    st.addTypes(flatTy);
    st.addTypes(
        RankedTensorType::get({mlir::ShapedType::kDynamic}, b_.getI64Type()));
    st.addAttribute("call_target_name", b_.getStringAttr(kGetDim));
    return b_.create(st)->getResult(0);
  }

  Value emitSetDimView(Value v, mlir::Type elem) {
    OperationState st(mlir::UnknownLoc::get(b_.getContext()),
                      mlir::OperationName("tf.residue", b_.getContext()));
    st.addOperands(v);
    st.addTypes(mlir::UnrankedTensorType::get(elem));
    st.addAttribute("call_target_name", b_.getStringAttr(kSetDim));
    return b_.create(st)->getResult(0);
  }

  Value emitDimSize(Value v, int64_t dim) {
    OperationState st(
        mlir::UnknownLoc::get(b_.getContext()),
        mlir::OperationName("stablehlo.get_dimension_size", b_.getContext()));
    st.addOperands(v);
    st.addTypes(RankedTensorType::get({}, b_.getI32Type()));
    st.addAttribute("dimension", b_.getI64IntegerAttr(dim));
    return b_.create(st)->getResult(0);
  }

  Value emitBinary(StringRef mnemonic, Value a, Value b, Type ty) {
    OperationState st(mlir::UnknownLoc::get(b_.getContext()),
                      mlir::OperationName(mnemonic, b_.getContext()));
    st.addOperands({a, b});
    st.addTypes(ty);
    return b_.create(st)->getResult(0);
  }

  Value emitShapeVector(llvm::SmallVectorImpl<Value>& dims) {
    Value shape = nullptr;
    for (Value d : dims) {
      OperationState rst(
          mlir::UnknownLoc::get(b_.getContext()),
          mlir::OperationName("stablehlo.reshape", b_.getContext()));
      rst.addOperands(d);
      rst.addTypes(RankedTensorType::get({1}, b_.getI32Type()));
      Value r = b_.create(rst)->getResult(0);
      if (shape == nullptr) {
        shape = r;
        continue;
      }
      int64_t n =
          llvm::cast<RankedTensorType>(shape.getType()).getDimSize(0) + 1;
      OperationState cst(
          mlir::UnknownLoc::get(b_.getContext()),
          mlir::OperationName("stablehlo.concatenate", b_.getContext()));
      cst.addOperands({shape, r});
      cst.addTypes(RankedTensorType::get({n}, b_.getI32Type()));
      cst.addAttribute("dimension", b_.getI64IntegerAttr(0));
      shape = b_.create(cst)->getResult(0);
    }
    return shape;
  }

  void emitNormalizeToMax(llvm::SmallVectorImpl<Value>& packeds) {
    const int rank =
        llvm::cast<RankedTensorType>(packeds[0].getType()).getRank();
    mlir::Location loc = mlir::UnknownLoc::get(b_.getContext());
    auto scalarI32 = RankedTensorType::get({}, b_.getI32Type());
    auto scalarZeroTy = RankedTensorType::get({}, b_.getI8Type());

    llvm::SmallVector<Value, 4> widths;
    for (Value p : packeds) {
      widths.push_back(emitDimSize(p, rank - 1));
    }
    Value wMax = widths[0];
    for (size_t k = 1; k < widths.size(); ++k) {
      wMax = emitBinary("stablehlo.max", wMax, widths[k], scalarI32);
    }

    for (size_t k = 0; k < packeds.size(); ++k) {
      Value wi = widths[k];
      Value diff = emitBinary("stablehlo.subtract", wMax, wi, scalarI32);
      llvm::SmallVector<Value, 6> dims;
      for (int d = 0; d + 1 < rank; ++d) {
        dims.push_back(emitDimSize(packeds[k], d));
      }
      dims.push_back(diff);
      Value shape = emitShapeVector(dims);
      Value zero = emitOp(
          "stablehlo.constant", loc, {}, scalarZeroTy, [&](OperationState& st) {
            st.addAttribute("value", mlir::DenseElementsAttr::get(
                                         scalarZeroTy, llvm::APInt(8, 0)));
          });
      OperationState rz(
          loc, mlir::OperationName("stablehlo.reshape", b_.getContext()));
      rz.addOperands({zero});
      rz.addTypes(RankedTensorType::get({1}, b_.getI8Type()));
      Value zero1 = b_.create(rz)->getResult(0);
      Value zeros = emitOp(
          "stablehlo.dynamic_broadcast_in_dim", loc, {zero1, shape},
          packeds[k].getType(), [&](OperationState& st) {
            st.addAttribute("broadcast_dimensions",
                            mlir::DenseIntElementsAttr::get(
                                RankedTensorType::get({1}, b_.getI64Type()),
                                llvm::ArrayRef<int64_t>{0}));
          });
      OperationState cst(
          loc, mlir::OperationName("stablehlo.concatenate", b_.getContext()));
      cst.addOperands({packeds[k], zeros});
      cst.addTypes(packeds[k].getType());
      cst.addAttribute("dimension", b_.getI64IntegerAttr(rank - 1));
      packeds[k] = b_.create(cst)->getResult(0);
    }
  }

  Value emitPackedConst(mlir::Location loc, llvm::ArrayRef<StringRef> strs,
                        RankedTensorType packedTy) {
    const int64_t w = packedTy.getShape().back();
    std::string flat;
    flat.reserve(static_cast<size_t>(strs.size()) * w);
    for (StringRef s : strs) {
      flat.append(static_cast<size_t>(w), '\0');
      char* row = &flat[flat.size() - static_cast<size_t>(w)];
      uint32_t len = static_cast<uint32_t>(s.size());
      // Host-endian length: fine for same-target builds; the row layout
      // is a target ABI contract (see the sink's decodeString).
      std::memcpy(row, &len, 4);
      std::memcpy(row + 4, s.data(), s.size());
    }
    llvm::SmallVector<mlir::APInt, 64> vals;
    vals.reserve(flat.size());
    for (unsigned char c : flat) {
      vals.push_back(mlir::APInt(8, c));
    }
    return emitOp(
        "stablehlo.constant", loc, {}, packedTy, [&](OperationState& st) {
          st.addAttribute("value",
                          DenseElementsAttr::get(
                              packedTy, llvm::ArrayRef<mlir::APInt>(vals)));
        });
  }

  Operation* rebuildTfOp(Operation* carrier, const std::string& cls,
                         mlir::ValueRange operands,
                         llvm::ArrayRef<Type> results) {
    OperationState st(carrier->getLoc(),
                      mlir::OperationName("tf." + cls, b_.getContext()));
    st.addOperands(operands);
    st.addTypes(results);
    for (NamedAttribute a : carrier->getAttrs()) {
      llvm::StringRef n = a.getName();
      if (n == "call_target_name" || n == "residue_reason" ||
          n == "custom_op" || n == "custom_attrs" || n == "dtype_assumed")
        continue;
      auto ta = llvm::dyn_cast<mlir::TypeAttr>(a.getValue());
      if (ta && isStringElem(ta.getValue())) {
        st.addAttribute(n, mlir::TypeAttr::get(b_.getI8Type()));
        continue;
      }
      st.addAttribute(n, a.getValue());
    }
    return b_.create(st);
  }

  OpBuilder& b_;
  int transfers_ = 0;
  int folded_ = 0;
};

bool StringTransport::rewrite(Operation* op, const std::string& cls) {
  llvm::SmallVector<unsigned, 4> dataIdx;
  if (!dataOperandIndices(cls, op->getNumOperands(), dataIdx)) {
    return false;
  }
  if (op->getNumResults() != 1) {
    return false;
  }
  for (unsigned i : dataIdx) {
    if (!isStringTensorValue(op->getOperand(i).getType())) {
      return false;
    }
  }
  if (!applyDimAdapter(op, cls, dataIdx)) {
    return false;
  }
  auto resTyT = llvm::dyn_cast<mlir::TensorType>(op->getResult(0).getType());
  if (!resTyT || !isStringElem(resTyT.getElementType())) {
    return false;
  }
  const bool resUnranked = !resTyT.hasRank();
  auto resTy = resUnranked ? RankedTensorType::get({mlir::ShapedType::kDynamic},
                                                   resTyT.getElementType())
                           : llvm::cast<RankedTensorType>(resTyT);

  const bool singleData = dataIdx.size() == 1;
  b_.setInsertionPoint(op);
  llvm::SmallVector<Value, 4> newOperands(op->getNumOperands(), Value());
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    newOperands[i] = op->getOperand(i);
  }
  llvm::SmallVector<Operation*, 4> foldedCarriers;
  int64_t resW = mlir::ShapedType::kDynamic;
  for (unsigned i : dataIdx) {
    Value orig = op->getOperand(i);
    llvm::SmallVector<StringRef, 16> strs;
    const bool isConst = readStringConst(orig, strs);
    Value v = orig;
    const bool foldConst = isConst && singleData;
    if (!foldConst) {
      auto inTy0 = llvm::cast<mlir::TensorType>(v.getType());
      if (!inTy0.hasRank()) {
        v = emitGetDimView(v);
      }
    }
    auto inTy = llvm::cast<RankedTensorType>(v.getType());
    llvm::SmallVector<int64_t, 6> dims(inTy.getShape().begin(),
                                       inTy.getShape().end());
    if (foldConst && isConst) {
      int64_t w = 4;
      for (StringRef s : strs) {
        w = std::max<int64_t>(w, (int64_t)s.size() + 4);
      }
      dims.push_back(w);
      resW = w;
      auto packedTy = RankedTensorType::get(dims, b_.getI8Type());
      newOperands[i] = emitPackedConst(op->getLoc(), strs, packedTy);
      ++folded_;
      Operation* def = orig.getDefiningOp();
      if (def != nullptr && def != op && def->hasAttr("value")) {
        llvm::StringRef dn = def->getName().getStringRef();
        if (dn == "tf.residue" || dn == "tf.Const") {
          foldedCarriers.push_back(def);
        }
      }
    } else {
      dims.push_back(mlir::ShapedType::kDynamic);
      auto packedTy = RankedTensorType::get(dims, b_.getI8Type());
      newOperands[i] = emitBoundary(mlir::UnknownLoc::get(b_.getContext()), {v},
                                    {packedTy}, kEncodeString)
                           ->getResult(0);
      ++transfers_;
    }
  }

  if (dataIdx.size() >= 2) {
    llvm::SmallVector<Value, 4> packeds;
    for (unsigned i : dataIdx) {
      packeds.push_back(newOperands[i]);
    }
    emitNormalizeToMax(packeds);
    for (size_t k = 0; k < dataIdx.size(); ++k) {
      newOperands[dataIdx[k]] = packeds[k];
    }
  }

  llvm::SmallVector<int64_t, 6> rdims(resTy.getShape().begin(),
                                      resTy.getShape().end());
  rdims.push_back(resW);
  auto packedResTy = RankedTensorType::get(rdims, b_.getI8Type());

  if (cls == "Pack" &&
      (!op->hasAttr("axis") ||
       op->getAttrOfType<mlir::IntegerAttr>("axis").getInt() == 0)) {
    llvm::SmallVector<Value, 4> rows;
    llvm::SmallVector<int64_t, 6> catShape;
    for (unsigned i : dataIdx) {
      Value pv = newOperands[i];
      auto pTy = llvm::cast<RankedTensorType>(pv.getType());
      llvm::SmallVector<int64_t, 6> r1(pTy.getShape().begin(),
                                       pTy.getShape().end());
      r1.insert(r1.begin(), 1);
      auto r1Ty = RankedTensorType::get(r1, b_.getI8Type());
      llvm::SmallVector<Value, 6> dimVals;
      OperationState one(op->getLoc(), "stablehlo.constant");
      auto s32Ty = RankedTensorType::get({}, b_.getI32Type());
      one.addTypes(s32Ty);
      one.addAttribute("value", mlir::DenseIntElementsAttr::get(
                                    s32Ty, llvm::ArrayRef<int64_t>{1}));
      dimVals.push_back(b_.create(one)->getResult(0));
      for (int64_t d = 0; d < pTy.getRank(); ++d) {
        if (pTy.isDynamicDim(d)) {
          dimVals.push_back(emitDimSize(pv, d));
        } else {
          OperationState cd(op->getLoc(), "stablehlo.constant");
          auto s32Ty2 = RankedTensorType::get({}, b_.getI32Type());
          cd.addTypes(s32Ty2);
          cd.addAttribute(
              "value", mlir::DenseIntElementsAttr::get(
                           s32Ty2, llvm::ArrayRef<int64_t>{pTy.getDimSize(d)}));
          dimVals.push_back(b_.create(cd)->getResult(0));
        }
      }
      Value shapeVec = emitShapeVector(dimVals);
      OperationState drst(
          op->getLoc(),
          mlir::OperationName("stablehlo.dynamic_reshape", b_.getContext()));
      drst.addOperands({pv, shapeVec});
      drst.addTypes(r1Ty);
      rows.push_back(b_.create(drst)->getResult(0));
      if (catShape.empty()) {
        catShape = r1;
        catShape[0] = (int64_t)dataIdx.size();
      }
    }
    auto catTy = RankedTensorType::get(catShape, b_.getI8Type());
    Value cat;
    if (rows.size() == 1) {
      cat = rows[0];
    } else {
      OperationState cst(
          op->getLoc(),
          mlir::OperationName("stablehlo.concatenate", b_.getContext()));
      cst.addOperands(rows);
      cst.addTypes({catTy});
      cst.addAttribute("dimension", b_.getI64IntegerAttr(0));
      cat = b_.create(cst)->getResult(0);
    }
    llvm::SmallVector<int64_t, 6> outShape(catShape.begin(),
                                           catShape.end() - 1);
    auto decTy = RankedTensorType::get(outShape, resTyT.getElementType());
    b_.setInsertionPointAfter(cat.getDefiningOp());
    Value decoded = emitBoundary(mlir::UnknownLoc::get(b_.getContext()), {cat},
                                 {decTy}, kDecodeString)
                        ->getResult(0);
    Value finalV = decoded;
    if (resUnranked) {
      finalV = emitSetDimView(decoded, resTyT.getElementType());
    }
    op->getResult(0).replaceAllUsesWith(finalV);
    op->erase();
    for (Operation* c : foldedCarriers) {
      if (c->use_empty()) {
        c->erase();
      }
    }
    return true;
  }

  Operation* tfop = rebuildTfOp(op, cls, newOperands, {packedResTy});

  b_.setInsertionPointAfter(tfop);
  Value decoded = emitBoundary(mlir::UnknownLoc::get(b_.getContext()),
                               {tfop->getResult(0)}, {resTy}, kDecodeString)
                      ->getResult(0);
  Value finalV = decoded;
  if (resUnranked) {
    finalV = emitSetDimView(decoded, resTyT.getElementType());
  }
  op->getResult(0).replaceAllUsesWith(finalV);
  op->erase();

  for (Operation* c : foldedCarriers) {
    if (c->use_empty()) {
      c->erase();
    }
  }
  return true;
}

struct EncodeStringTransportPass
    : public mlir::PassWrapper<EncodeStringTransportPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EncodeStringTransportPass)

  llvm::StringRef getArgument() const override {
    return "encode-string-transport";
  }
  llvm::StringRef getDescription() const override {
    return "wrap string-typed transport ops with "
           "annc.encodeString/decodeString "
           "boundaries and retype them to i8 so the official legalize emits "
           "stablehlo.op3";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    OpBuilder b(&getContext());
    StringTransport st(b);

    struct Target {
      Operation* op;
      std::string cls;
    };
    llvm::SmallVector<Target, 8> targets;
    module.walk([&](Operation* op) {
      llvm::StringRef nm = op->getName().getStringRef();
      std::string cls;
      if (nm == "tf.residue" || nm == "annc.residue") {
        auto ct = op->getAttrOfType<mlir::StringAttr>("call_target_name");
        if (!ct || !ct.getValue().starts_with("tf.")) {
          return;
        }
        cls = ct.getValue().drop_front(3).str();
      } else if (nm.starts_with("tf.")) {
        cls = nm.drop_front(3).str();
      } else {
        return;
      }
      if (!inAllowlist(cls)) {
        return;
      }
      targets.push_back({op, cls});
    });

    int rewritten = 0, skipped = 0;
    for (Target& t : targets) {
      if (t.op->getBlock() == nullptr) {
        continue;
      }
      if (st.rewrite(t.op, t.cls)) {
        ++rewritten;
      } else {
        ++skipped;
      }
    }
    if (rewritten || skipped)
      llvm::errs() << "[string-transport] wrapped=" << rewritten
                   << " skipped=" << skipped
                   << " constFolded=" << st.folded() << "\n";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createStringTransportPass() {
  return std::make_unique<EncodeStringTransportPass>();
}

}  // namespace pb2
