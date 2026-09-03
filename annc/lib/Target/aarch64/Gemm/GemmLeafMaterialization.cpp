#include "GemmPlan.h"
#include "GemmTilingUtils.h"
#include "Target/aarch64/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

namespace annc {
namespace {

func::FuncOp getOrCreateLeafDeclaration(ModuleOp module, StringRef name,
                                        FunctionType type,
                                        bool isPrivate = true) {
  if (auto declaration = module.lookupSymbol<func::FuncOp>(name))
    return declaration;
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto declaration = builder.create<func::FuncOp>(module.getLoc(), name, type);
  if (isPrivate) declaration.setPrivate();
  return declaration;
}

using WorkspaceMap = llvm::DenseMap<Operation *, Value>;

FailureOr<int64_t> getStaticDimension(Value value, unsigned dimension) {
  auto type = llvm::dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() != 2 || type.isDynamicDim(dimension))
    return failure();
  return type.getShape()[dimension];
}

Value callSvePackedBHelper(ModuleOp module, OpBuilder &builder, Location loc,
                           StringRef name, Value first, Value second) {
  Type index = builder.getIndexType();
  getOrCreateLeafDeclaration(module, name,
                             builder.getFunctionType({index, index}, {index}));
  return builder
      .create<func::CallOp>(loc, name, TypeRange{index},
                            ValueRange{first, second})
      .getResult(0);
}

Value getOrCreateWorkspace(ModuleOp module, Operation *op,
                           const aarch64::gemm::GemmPlan &plan,
                           WorkspaceMap &workspaces) {
  func::FuncOp function = op->getParentOfType<func::FuncOp>();
  Operation *functionOp = function.getOperation();
  if (auto existing = workspaces.find(functionOp); existing != workspaces.end())
    return existing->second;

  OpBuilder builder(function.getContext());
  builder.setInsertionPointToStart(&function.front());
  MemRefType type;
  ValueRange dynamicSizes;
  if (plan.isa == aarch64::gemm::GemmIsa::kSve) {
    Value k = builder.create<arith::ConstantIndexOp>(op->getLoc(),
                                                      plan.cacheTile.kc);
    Value n = builder.create<arith::ConstantIndexOp>(op->getLoc(),
                                                      plan.cacheTile.nc);
    Value elements =
        callSvePackedBHelper(module, builder, op->getLoc(),
                             aarch64::gemm::kSvePackedBElementsAsmSymbol, k, n);
    type = MemRefType::get({ShapedType::kDynamic}, builder.getF32Type());
    dynamicSizes = ValueRange{elements};
  } else {
    type = MemRefType::get({plan.cacheTile.kc * plan.cacheTile.nc},
                           builder.getF32Type());
  }
  auto alloca =
      builder.create<memref::AllocaOp>(op->getLoc(), type, dynamicSizes);
  alloca->setAttr("alignment", builder.getI64IntegerAttr(64));
  workspaces.try_emplace(functionOp, alloca);
  return alloca;
}

Value materializeIndex(OpBuilder &builder, Location loc, OpFoldResult value) {
  if (auto dynamicValue = llvm::dyn_cast<Value>(value)) return dynamicValue;
  auto attr = llvm::dyn_cast<IntegerAttr>(llvm::cast<Attribute>(value));
  return builder.create<arith::ConstantIndexOp>(loc, attr.getInt());
}

bool isStaticZero(OpFoldResult value) {
  if (auto dynamicValue = llvm::dyn_cast<Value>(value)) {
    if (auto constant = dynamicValue.getDefiningOp<arith::ConstantIndexOp>())
      return constant.value() == 0;
    return false;
  }
  auto attr = llvm::dyn_cast<Attribute>(value);
  if (!attr) return false;
  if (auto integer = llvm::dyn_cast<IntegerAttr>(attr))
    return integer.getInt() == 0;
  return false;
}

void copyScheduleAttrs(Operation *source, func::CallOp target,
                       OpBuilder &builder) {
  for (StringRef name :
       {aarch64::gemm::kPlanAttrName, aarch64::gemm::kKcModeAttrName,
        aarch64::gemm::kEpilogueAttrName}) {
    if (Attribute attr = source->getAttr(name))
      target->setDiscardableAttr(name, attr);
  }
  aarch64::gemm::setStage(target, builder, aarch64::gemm::kPackedStage);
}

Value materializePackCall(ModuleOp module, OpBuilder &builder, Location loc,
                          Value rhsBase, Value rhsOffset, Value workspace,
                          Value kSize, Value nSize, Value ldb,
                          aarch64::gemm::GemmTarget target,
                          aarch64::gemm::GemmIsa isa) {
  Value packedBase =
      aarch64::gemm::castToUnrankedF32MemRef(builder, loc, workspace);
  Value rhsUnranked =
      aarch64::gemm::castToUnrankedF32MemRef(builder, loc, rhsBase);
  auto unranked = UnrankedMemRefType::get(builder.getF32Type(), 0);
  auto index = builder.getIndexType();
  getOrCreateLeafDeclaration(
      module, aarch64::gemm::kPackBLeafName,
      builder.getFunctionType({unranked, unranked, index, index, index, index},
                              {}));
  auto packCall = builder.create<func::CallOp>(
      loc, aarch64::gemm::kPackBLeafName, TypeRange{},
      ValueRange{rhsUnranked, packedBase, rhsOffset, kSize, nSize, ldb});
  aarch64::gemm::setStage(packCall, builder, aarch64::gemm::kPackedStage);
  packCall->setDiscardableAttr(
      aarch64::gemm::kAsmSymbolAttrName,
      builder.getStringAttr(aarch64::gemm::getPackBAsmSymbol(target, isa)));
  return packedBase;
}

Value getPackedBPanelOffset(OpBuilder &builder, Location loc,
                            Value columnOffset, Value kSize) {
  // Kernel tiling advances JR by NR, which is exactly one packed-B panel.
  // Therefore every generated microtile starts at a panel boundary and its
  // packed element offset simplifies from the general SVE layout formula to
  // columnOffset * kSize.
  return builder.create<arith::MulIOp>(loc, columnOffset, kSize);
}

struct PackedBBlock {
  Value base;
  Value offset;
};

struct PackedBWorkspace {
  Value base;
  Value kSize;
};

using PackedBBlockMap = llvm::DenseMap<Value, PackedBWorkspace>;

FailureOr<PackedBBlock> materializePackedBBlock(
    ModuleOp module, Operation *op, const aarch64::gemm::GemmPlan &plan,
    WorkspaceMap &workspaces, PackedBBlockMap &packedBlocks) {
  Value rhsInput = aarch64::gemm::getGemmInput(op, 1);
  auto rhsTile = rhsInput.getDefiningOp<memref::SubViewOp>();

  // Kernel tiling creates rhsTile from the (PC, JC) RHS block. Pack that
  // cache block once so every IR/JR microtile reuses its packed panels.
  if (rhsTile && rhsTile.getMixedOffsets().size() == 2) {
    if (!isStaticZero(rhsTile.getMixedOffsets()[0])) {
      return op->emitOpError(
          "requires kernel RHS microtiles with a zero K offset");
    }
    Value rhsBlock = rhsTile.getSource();
    Operation *rhsBlockDef = rhsBlock.getDefiningOp();
    Operation *hoistPoint = op;
    while (rhsBlockDef && hoistPoint &&
           hoistPoint->getBlock() != rhsBlockDef->getBlock())
      hoistPoint = hoistPoint->getParentOp();
    if (hoistPoint && rhsBlockDef->isBeforeInBlock(hoistPoint)) {
      OpBuilder microBuilder(op);
      Location loc = op->getLoc();
      Value jr =
          materializeIndex(microBuilder, loc, rhsTile.getMixedOffsets()[1]);

      if (auto existing = packedBlocks.find(rhsBlock);
          existing != packedBlocks.end()) {
        Value packedOffset = getPackedBPanelOffset(microBuilder, loc, jr,
                                                   existing->second.kSize);
        return PackedBBlock{existing->second.base, packedOffset};
      }

      OpBuilder packBuilder(hoistPoint);
      FailureOr<aarch64::gemm::MemRefBaseAndOffset> rhs =
          aarch64::gemm::getMemRefBaseAndOffset(packBuilder, loc, rhsBlock);
      if (failed(rhs)) {
        return op->emitOpError(
            "requires identity-layout bases with statically strided "
            "(PC, JC) RHS subviews");
      }

      Value workspace = getOrCreateWorkspace(module, op, plan, workspaces);
      FailureOr<int64_t> staticK = getStaticDimension(rhsBlock, 0);
      FailureOr<int64_t> staticN = getStaticDimension(rhsBlock, 1);
      if (failed(staticK) || failed(staticN))
        return op->emitOpError("requires statically shaped RHS cache blocks");
      Value kSize = packBuilder.create<arith::ConstantIndexOp>(loc, *staticK);
      Value nSize = packBuilder.create<arith::ConstantIndexOp>(loc, *staticN);
      Value ldbValue =
          packBuilder.create<arith::ConstantIndexOp>(loc, plan.ldb);
      Value packedBase = materializePackCall(
          module, packBuilder, loc, rhs->base, rhs->offset, workspace, kSize,
          nSize, ldbValue, plan.target, plan.isa);
      packedBlocks.try_emplace(rhsBlock, PackedBWorkspace{packedBase, kSize});
      Value packedOffset = getPackedBPanelOffset(microBuilder, loc, jr, kSize);
      return PackedBBlock{packedBase, packedOffset};
    }
  }

  // Preserve standalone kernel_tiled lowering for pass-level users that have
  // not materialized the cache/kernel loop structure.
  OpBuilder builder(op);
  Location loc = op->getLoc();
  FailureOr<aarch64::gemm::MemRefBaseAndOffset> rhs =
      aarch64::gemm::getMemRefBaseAndOffset(builder, loc, rhsInput);
  if (failed(rhs)) {
    return op->emitOpError(
        "requires identity-layout bases with statically strided RHS subviews");
  }
  Value workspace = getOrCreateWorkspace(module, op, plan, workspaces);
  FailureOr<int64_t> staticK = getStaticDimension(rhsInput, 0);
  FailureOr<int64_t> staticN = getStaticDimension(rhsInput, 1);
  if (failed(staticK) || failed(staticN)) {
    return op->emitOpError("requires statically shaped RHS microtiles");
  }
  Value kSize = builder.create<arith::ConstantIndexOp>(loc, *staticK);
  Value nSize = builder.create<arith::ConstantIndexOp>(loc, *staticN);
  Value ldbValue = builder.create<arith::ConstantIndexOp>(loc, plan.ldb);
  Value packedBase = materializePackCall(module, builder, loc, rhs->base,
                                         rhs->offset, workspace, kSize, nSize,
                                         ldbValue, plan.target, plan.isa);
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  return PackedBBlock{packedBase, zero};
}

LogicalResult materializeLeafCalls(ModuleOp module, Operation *op,
                                   WorkspaceMap &workspaces,
                                   PackedBBlockMap &packedBlocks) {
  FailureOr<aarch64::gemm::GemmPlan> plan = aarch64::gemm::readPlan(op);
  if (failed(plan) || failed(aarch64::gemm::requireStage(
                          op, aarch64::gemm::kKernelTiledStage))) {
    return failure();
  }
  OpBuilder builder(op);
  Location loc = op->getLoc();
  Value lhsInput = aarch64::gemm::getGemmInput(op, 0);
  Value rhsInput = aarch64::gemm::getGemmInput(op, 1);
  Value output = aarch64::gemm::getGemmOutput(op, 0);
  FailureOr<aarch64::gemm::MemRefBaseAndOffset> lhs =
      aarch64::gemm::getMemRefBaseAndOffset(builder, loc, lhsInput);
  FailureOr<aarch64::gemm::MemRefBaseAndOffset> out =
      aarch64::gemm::getMemRefBaseAndOffset(builder, loc, output);
  if (failed(lhs) || failed(out)) {
    return op->emitOpError(
        "requires identity-layout bases with statically strided subviews");
  }

  const bool isDirectRhs =
      plan->rhsPacking == aarch64::gemm::RhsPacking::kDirect;
  const bool usesLdbAbi =
      aarch64::gemm::getGemmLeafAbi(plan->executionKind, plan->rhsPacking) ==
      aarch64::gemm::GemmLeafAbi::kRowMajor;
  const bool scalarK =
      usesLdbAbi &&
      plan->executionKind == aarch64::gemm::GemmExecutionKind::kGemm;
  Value rhsBase;
  Value rhsOffset;
  if (isDirectRhs) {
    FailureOr<aarch64::gemm::MemRefBaseAndOffset> rhs =
        aarch64::gemm::getMemRefBaseAndOffset(builder, loc, rhsInput);
    if (failed(rhs)) {
      return op->emitOpError(
          "requires identity-layout bases with statically strided direct RHS");
    }
    rhsBase = rhs->base;
    rhsOffset = rhs->offset;
  } else {
    FailureOr<PackedBBlock> packed =
        materializePackedBBlock(module, op, *plan, workspaces, packedBlocks);
    if (failed(packed)) return failure();
    rhsBase = packed->base;
    rhsOffset = packed->offset;
  }

  FailureOr<int64_t> staticM = getStaticDimension(lhsInput, 0);
  FailureOr<int64_t> staticK = getStaticDimension(lhsInput, 1);
  FailureOr<int64_t> staticN = getStaticDimension(rhsInput, 1);
  FailureOr<int64_t> nr = aarch64::gemm::getGemmNr(
      plan->kernelTile, plan->vectorLengthBytes, plan->dataType,
      plan->executionKind);
  if (failed(staticM) || failed(staticK) || failed(staticN) || *staticM < 1 ||
      *staticM > plan->kernelTile.mr) {
    return op->emitOpError(
        "requires static M/K microtiles within the selected kernel family");
  }
  if (failed(nr) || *staticN < 1 || *staticN > *nr)
    return op->emitOpError(
        "requires static N microtiles within the selected kernel family");

  Value lhsBase =
      aarch64::gemm::castToUnrankedF32MemRef(builder, loc, lhs->base);
  Value outBase =
      aarch64::gemm::castToUnrankedF32MemRef(builder, loc, out->base);

  Value lda = builder.create<arith::ConstantIndexOp>(loc, plan->lda);
  Value ldb = builder.create<arith::ConstantIndexOp>(loc, plan->ldb);
  Value ldc = builder.create<arith::ConstantIndexOp>(loc, plan->ldc);
  Value kSize = builder.create<arith::ConstantIndexOp>(loc, *staticK);
  Value familyArgument;
  if (plan->isa == aarch64::gemm::GemmIsa::kSve) {
    familyArgument = builder.create<arith::ConstantIndexOp>(loc, *staticN);
  } else {
    const aarch64::gemm::GemmKernelABI &abi =
        aarch64::gemm::getGemmKernelABI(plan->target, plan->isa,
                                        plan->dataType,
                                        plan->executionKind);
    FailureOr<int64_t> kScalarUnroll =
        aarch64::gemm::getGemmKScalarUnroll(abi, plan->dataType);
    if (failed(kScalarUnroll))
      return op->emitOpError(
          "requires a valid NEON kernel ABI vector K unroll");
    familyArgument = builder.create<arith::ConstantIndexOp>(
        loc, scalarK ? *staticK : *staticK / *kScalarUnroll);
  }

  auto unranked = UnrankedMemRefType::get(builder.getF32Type(), 0);
  auto index = builder.getIndexType();
  StringRef leafName = usesLdbAbi ? aarch64::gemm::kMicrokernelRmLeafName
                                  : aarch64::gemm::kMicrokernelLeafName;
  SmallVector<Type> leafTypes = {unranked, unranked, unranked};
  for (int64_t i = 0, count = usesLdbAbi ? 8 : 7; i < count; ++i)
    leafTypes.push_back(index);
  getOrCreateLeafDeclaration(module, leafName,
                             builder.getFunctionType(leafTypes, {}));

  auto kcMode = op->getAttrOfType<StringAttr>(aarch64::gemm::kKcModeAttrName);
  if (!kcMode ||
      (kcMode.getValue() != "overwrite" && kcMode.getValue() != "accumulate")) {
    return op->emitOpError(
        "requires a valid KC mode before GEMM leaf materialization");
  }
  SmallVector<Value> leafOperands = {
      lhsBase,
      isDirectRhs ? aarch64::gemm::castToUnrankedF32MemRef(builder, loc,
                                                            rhsBase)
                  : rhsBase,
      outBase,
      lhs->offset,
      rhsOffset,
      out->offset,
      lda};
  if (usesLdbAbi) leafOperands.push_back(ldb);
  leafOperands.push_back(ldc);
  leafOperands.push_back(kSize);
  leafOperands.push_back(familyArgument);
  auto kernelCall = builder.create<func::CallOp>(loc, leafName, TypeRange{},
                                                 leafOperands);
  copyScheduleAttrs(op, kernelCall, builder);
  kernelCall->setDiscardableAttr(aarch64::gemm::kMicrokernelMAttrName,
                                 builder.getI64IntegerAttr(*staticM));
  kernelCall->setDiscardableAttr(aarch64::gemm::kMicrokernelNAttrName,
                                 builder.getI64IntegerAttr(*staticN));
  kernelCall->setDiscardableAttr(aarch64::gemm::kMicrokernelKAttrName,
                                 builder.getI64IntegerAttr(*staticK));

  op->erase();
  return success();
}

class AArch64GemmLeafMaterialization
    : public AArch64GemmLeafMaterializationBase<
          AArch64GemmLeafMaterialization> {
 public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> candidates;
    module.walk([&](Operation *op) {
      if (aarch64::gemm::isGemmAnchor(op) &&
          aarch64::gemm::hasStage(op, aarch64::gemm::kKernelTiledStage)) {
        candidates.push_back(op);
      }
    });
    WorkspaceMap workspaces;
    PackedBBlockMap packedBlocks;
    for (Operation *op : candidates) {
      if (failed(materializeLeafCalls(module, op, workspaces, packedBlocks))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64GemmLeafMaterialization() {
  return std::make_unique<AArch64GemmLeafMaterialization>();
}

}  // namespace annc
