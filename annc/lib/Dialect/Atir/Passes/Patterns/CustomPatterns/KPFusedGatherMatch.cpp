#include "Dialect/Atir/Passes/Patterns/CustomPatterns/KPFusedGatherMatch.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace atir {
namespace {

bool constantInts(mlir::Value value, llvm::ArrayRef<int64_t> expected) {
  auto tensorType = mlir::dyn_cast<TensorType>(value.getType());
  auto constant = value.getDefiningOp<ConstantOp>();
  if (!tensorType || !constant) return false;
  auto data = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
      tensorType.getCacheData());
  if (!data || data.getNumElements() != static_cast<int64_t>(expected.size()) ||
      !data.getElementType().isIntOrIndex())
    return false;
  size_t index = 0;
  for (const mlir::APInt &value : data.getValues<mlir::APInt>()) {
    if (value.getSExtValue() != expected[index++]) return false;
  }
  return true;
}

bool rankAndElement(mlir::Type type, unsigned rank, unsigned width) {
  auto tensor = mlir::dyn_cast<TensorType>(type);
  if (!tensor || tensor.getShape().size() != rank) return false;
  auto integer = mlir::dyn_cast<mlir::IntegerType>(tensor.getElementType());
  if (width == 0) return mlir::isa<mlir::Float32Type>(tensor.getElementType());
  return integer && integer.getWidth() == width;
}

bool onlyAllowedUsers(mlir::Value value,
                      llvm::ArrayRef<mlir::Operation *> allowed) {
  for (mlir::Operation *user : value.getUsers())
    if (!llvm::is_contained(allowed, user) &&
        !mlir::isa<mlir::func::ReturnOp>(user))
      return false;
  return true;
}

}  // namespace

mlir::FailureOr<KPFusedGatherMatch> matchKPFusedGather(GatherOp outerGather) {
  KPFusedGatherMatch match;
  match.outerGather = outerGather;
  match.innerGather = outerGather.getParams().getDefiningOp<GatherOp>();
  match.secondUnique = outerGather.getIndices().getDefiningOp<UniqueOp>();
  if (!match.innerGather || !match.secondUnique ||
      outerGather.getIndices() != match.secondUnique.getIdx())
    return mlir::failure();
  if (!constantInts(outerGather.getAxis(), {0}) ||
      outerGather.getBatchDims() != 0)
    return mlir::failure();

  match.firstUnique = match.secondUnique.getX().getDefiningOp<UniqueOp>();
  match.slice = match.firstUnique
                    ? match.firstUnique.getX().getDefiningOp<StridedSliceOp>()
                    : nullptr;
  if (!match.firstUnique || !match.slice ||
      match.secondUnique.getX() != match.firstUnique.getY() ||
      match.innerGather.getIndices() != match.secondUnique.getY() ||
      !constantInts(match.innerGather.getAxis(), {0}) ||
      match.innerGather.getBatchDims() != 0)
    return mlir::failure();

  match.keys = match.slice.getInput();
  match.begin = match.slice.getBegin();
  match.data = match.innerGather.getParams();
  // end 值不查: end_mask=1 时 TF 语义忽略 end, kernel 也不用 (实测 hmv_930
  // 的 ss.end=[0,1] 与 pattern 原要求的 [0,0] 语义等价)。strides 值检查
  // 保留——无 mask, 决定列提取语义。
  if (!constantInts(match.slice.getStrides(), {1, 1}) ||
      match.slice.getBeginMask() != 1 || match.slice.getEndMask() != 1 ||
      match.slice.getShrinkAxisMask() != 2 ||
      match.slice.getEllipsisMask() != 0 || match.slice.getNewAxisMask() != 0)
    return mlir::failure();

  if (!rankAndElement(match.data.getType(), 2, 0) ||
      !rankAndElement(match.keys.getType(), 2, 64) ||
      !rankAndElement(match.begin.getType(), 1, 32) ||
      !rankAndElement(match.firstUnique.getY().getType(), 1, 64) ||
      !rankAndElement(match.firstUnique.getIdx().getType(), 1, 32) ||
      !rankAndElement(match.outerGather.getResult().getType(), 2, 0))
    return mlir::failure();

  match.boundaryOutputs = {match.firstUnique.getY(), match.firstUnique.getIdx(),
                           match.outerGather.getResult()};
  if (!onlyAllowedUsers(match.secondUnique.getY(), {match.innerGather}) ||
      !onlyAllowedUsers(match.secondUnique.getIdx(), {match.outerGather}) ||
      !onlyAllowedUsers(match.innerGather.getResult(), {match.outerGather}) ||
      !onlyAllowedUsers(match.slice.getResult(), {match.firstUnique}))
    return mlir::failure();
  return match;
}

}  // namespace atir
