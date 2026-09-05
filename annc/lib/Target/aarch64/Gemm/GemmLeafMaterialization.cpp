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

// Weak reference to the host-side SVE packed-B sizing helper; resolved when
// the microkernel archive is linked in.
extern "C" int64_t annc_aarch64_sve_packed_b_elements_f32(int64_t k,
                                                          int64_t n)
    __attribute__((weak));

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

int64_t packedBElements(aarch64::gemm::GemmIsa isa, int64_t k, int64_t n) {
  return isa == aarch64::gemm::GemmIsa::kSve
             ? annc_aarch64_sve_packed_b_elements_f32(k, n)
             : k * ((n + 3) / 4) * 4;
}

// Elements covered by the first `count` column blocks of one KC row.
int64_t columnPrefix(aarch64::gemm::GemmIsa isa, int64_t k, int64_t n,
                     int64_t nc, int64_t count) {
  int64_t total = 0;
  for (int64_t jc = 0; jc < count; ++jc)
    total += packedBElements(isa, k, std::min(nc, n - jc * nc));
  return total;
}

// Elements before the pcIndex-th KC row of the whole packed RHS.
int64_t rowPrefix(aarch64::gemm::GemmIsa isa, int64_t k, int64_t n,
                  const aarch64::gemm::GemmCacheTile &tile, int64_t pcIndex) {
  int64_t total = 0;
  for (int64_t pc = 0; pc < pcIndex; ++pc)
    total += columnPrefix(isa, std::min(tile.kc, k - pc * tile.kc), n,
                          tile.nc, (n + tile.nc - 1) / tile.nc);
  return total;
}

FailureOr<Value> getPrepackedRhsGlobal(ModuleOp module, Operation *op,
                                       const aarch64::gemm::GemmPlan &plan,
                                       OpBuilder &builder, Location loc) {
  if (plan.prepackedRhsSymbol.empty() || plan.prepackedRhsElements <= 0)
    return op->emitOpError("has no valid prepacked RHS global reference");
  MemRefType type =
      MemRefType::get({plan.prepackedRhsElements}, builder.getF32Type());
  Operation *existing = module.lookupSymbol(plan.prepackedRhsSymbol);
  if (existing) {
    auto global = llvm::dyn_cast<memref::GlobalOp>(existing);
    if (!global || global.getType() != type || !global.isExternal())
      return op->emitOpError()
             << "prepacked RHS symbol '" << plan.prepackedRhsSymbol
             << "' conflicts with an incompatible module symbol";
  } else {
    OpBuilder moduleBuilder(module.getContext());
    moduleBuilder.setInsertionPointToStart(module.getBody());
    moduleBuilder.create<memref::GlobalOp>(
        loc, plan.prepackedRhsSymbol, StringAttr(), type, Attribute(), false,
        builder.getI64IntegerAttr(64));
  }
  return builder.create<memref::GetGlobalOp>(loc, type,
                                             plan.prepackedRhsSymbol)
      .getResult();
}

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

// Materialize the (PC, JC) offset for the pc-major / jc-minor packed RHS.
Value materializePrepackedCacheOffset(OpBuilder &builder, Location loc,
                                      const aarch64::gemm::GemmPlan &plan,
                                      int64_t kSize, Value pc, Value jc) {
  const int64_t kc = plan.cacheTile.kc;
  const int64_t nc = plan.cacheTile.nc;
  const int64_t kRows = (plan.k + kc - 1) / kc;
  const int64_t nCols = (plan.n + nc - 1) / nc;
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value rowIndex = builder.create<arith::DivUIOp>(
      loc, pc, builder.create<arith::ConstantIndexOp>(loc, kc));
  Value colIndex = builder.create<arith::DivUIOp>(
      loc, jc, builder.create<arith::ConstantIndexOp>(loc, nc));
  Value offset = zero;
  auto addContribution = [&](Value index, int64_t bound, int64_t width) {
    Value beyond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, index,
        builder.create<arith::ConstantIndexOp>(loc, bound));
    Value contribution = builder.create<arith::SelectOp>(
        loc, beyond, builder.create<arith::ConstantIndexOp>(loc, width), zero);
    offset = builder.create<arith::AddIOp>(loc, offset, contribution);
  };
  // Full rows before the current K row cover every N column.
  for (int64_t r = 0; r < kRows; ++r) {
    const int64_t kr = std::min(kc, plan.k - r * kc);
    int64_t rowWidth = 0;
    for (int64_t c = 0; c < nCols; ++c) {
      const int64_t blockN = std::min(nc, plan.n - c * nc);
      rowWidth += packedBElements(plan.isa, kr, blockN);
    }
    addContribution(rowIndex, r, rowWidth);
  }
  // Columns before JC inside the current K row.
  for (int64_t c = 0; c < nCols; ++c) {
    const int64_t blockN = std::min(nc, plan.n - c * nc);
    addContribution(colIndex, c, packedBElements(plan.isa, kSize, blockN));
  }
  return offset;
}

FailureOr<PackedBBlock> materializePrepackedBBlock(
    ModuleOp module, Operation *op, const aarch64::gemm::GemmPlan &plan,
    OpBuilder &builder, Value rhsBlock, Value jr) {
  auto rhsSubview = rhsBlock.getDefiningOp<memref::SubViewOp>();
  if (!rhsSubview || rhsSubview.getMixedOffsets().size() != 2)
    return op->emitOpError(
        "prepacked RHS requires cache-blocked RHS subviews");
  FailureOr<int64_t> kSize = getStaticDimension(rhsBlock, 0);
  if (failed(kSize))
    return op->emitOpError("prepacked RHS requires static cache K blocks");

  OpBuilder &microBuilder = builder;
  Location loc = op->getLoc();
  Value pc = materializeIndex(microBuilder, loc,
                              rhsSubview.getMixedOffsets()[0]);
  Value jc = materializeIndex(microBuilder, loc,
                              rhsSubview.getMixedOffsets()[1]);
  // packed offset = cache-block prefix (pc row + jc column) + panel offset.
  Value offset;
  auto pcConstant = pc.getDefiningOp<arith::ConstantIndexOp>();
  auto jcConstant = jc.getDefiningOp<arith::ConstantIndexOp>();
  if (pcConstant && jcConstant) {
    offset = microBuilder.create<arith::ConstantIndexOp>(
        loc, rowPrefix(plan.isa, plan.k, plan.n, plan.cacheTile,
                       pcConstant.value() / plan.cacheTile.kc) +
                 columnPrefix(plan.isa, *kSize, plan.n, plan.cacheTile.nc,
                              jcConstant.value() / plan.cacheTile.nc));
  } else {
    offset = materializePrepackedCacheOffset(microBuilder, loc, plan, *kSize,
                                             pc, jc);
  }
  if (plan.isa == aarch64::gemm::GemmIsa::kSve) {
    Value blockK = microBuilder.create<arith::ConstantIndexOp>(loc, *kSize);
    Value panel = callSvePackedBHelper(
        module, microBuilder, loc, aarch64::gemm::kSvePackedBOffsetAsmSymbol,
        blockK, jr);
    offset = microBuilder.create<arith::AddIOp>(loc, offset, panel);
  } else {
    Value group = microBuilder.create<arith::DivUIOp>(
        loc, jr, microBuilder.create<arith::ConstantIndexOp>(loc, 4));
    Value groupOffset = microBuilder.create<arith::MulIOp>(
        loc, group,
        microBuilder.create<arith::ConstantIndexOp>(loc, *kSize * 4));
    Value lane = microBuilder.create<arith::AndIOp>(
        loc, jr, microBuilder.create<arith::ConstantIndexOp>(loc, 3));
    offset = microBuilder.create<arith::AddIOp>(
        loc, offset, microBuilder.create<arith::AddIOp>(loc, groupOffset, lane));
  }

  FailureOr<Value> global =
      getPrepackedRhsGlobal(module, op, plan, microBuilder, loc);
  if (failed(global)) return failure();
  return PackedBBlock{
      aarch64::gemm::castToUnrankedF32MemRef(microBuilder, loc, *global),
      offset};
}

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

      if (plan.rhsPacking == aarch64::gemm::RhsPacking::kPrepacked)
        return materializePrepackedBBlock(module, op, plan, microBuilder,
                                          rhsBlock, jr);

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
  if (plan.rhsPacking == aarch64::gemm::RhsPacking::kPrepacked)
    return op->emitOpError(
        "prepacked RHS requires cache-blocked GEMM structure");
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
  const bool usesLdbAbi = aarch64::gemm::usesLdbAbi(
    aarch64::gemm::getGemmLeafKind(plan->executionKind, plan->rhsPacking));
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
