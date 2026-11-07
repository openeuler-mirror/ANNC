#include "Dialect/Atir/Passes/GemmEpilogueCandidate.h"
#include "GemmMicrokernelSelection.h"
#include "GemmPacking.h"
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

int64_t packedBElements(const aarch64::gemm::GemmExecutionContext &context, int64_t k,
                        int64_t n) {
  // The caller checks the optional archive sizing helper before materializing
  // SVE prepacked data; offsets follow the layout declared by the kernel ABI.
  const auto &abi = aarch64::gemm::getGemmKernelABI(
      context.strategy.target, context.strategy.isa, context.strategy.dataType,
      context.strategy.executionKind);
  return aarch64::gemm::getGemmPackedBElements(abi, k, n);
}

// Elements covered by the first `count` column blocks of one KC row.
int64_t columnPrefix(const aarch64::gemm::GemmExecutionContext &context, int64_t k,
                     int64_t n,
                     int64_t nc, int64_t count) {
  int64_t total = 0;
  for (int64_t jc = 0; jc < count; ++jc)
    total += packedBElements(context, k, std::min(nc, n - jc * nc));
  return total;
}

// Elements before the pcIndex-th KC row of the whole packed RHS.
int64_t rowPrefix(const aarch64::gemm::GemmExecutionContext &context, int64_t k, int64_t n,
                  const aarch64::gemm::FixedCacheParameters &tile, int64_t pcIndex) {
  int64_t total = 0;
  for (int64_t pc = 0; pc < pcIndex; ++pc)
    total += columnPrefix(context, std::min(tile.kc, k - pc * tile.kc), n,
                          tile.nc, (n + tile.nc - 1) / tile.nc);
  return total;
}

FailureOr<Value> getPrepackedRhsGlobal(ModuleOp module, Operation *op,
                                       const aarch64::gemm::GemmExecutionContext &context,
                                       OpBuilder &builder, Location loc) {
  if (context.prepackedRhsSymbol.empty() || context.prepackedRhsElements <= 0)
    return op->emitOpError("has no valid prepacked RHS global reference");
  MemRefType type =
      MemRefType::get({context.prepackedRhsElements}, builder.getF32Type());
  Operation *existing = module.lookupSymbol(context.prepackedRhsSymbol);
  if (existing) {
    auto global = llvm::dyn_cast<memref::GlobalOp>(existing);
    if (!global || global.getType() != type || !global.isExternal())
      return op->emitOpError()
             << "prepacked RHS symbol '" << context.prepackedRhsSymbol
             << "' conflicts with an incompatible module symbol";
  } else {
    OpBuilder moduleBuilder(module.getContext());
    moduleBuilder.setInsertionPointToStart(module.getBody());
    moduleBuilder.create<memref::GlobalOp>(
        loc, context.prepackedRhsSymbol, StringAttr(), type, Attribute(), false,
        builder.getI64IntegerAttr(64));
  }
  return builder.create<memref::GetGlobalOp>(loc, type,
                                             context.prepackedRhsSymbol)
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

struct WorkspaceRequirement {
  int64_t maxK = 0;
  int64_t maxN = 0;
  int64_t maxElements = 0;
};

using WorkspaceMap = llvm::DenseMap<Operation *, Value>;
using WorkspaceRequirements =
    llvm::DenseMap<Operation *, WorkspaceRequirement>;

Value getOrCreatePackedBWorkspace(
    ModuleOp module, Operation *op,
    const aarch64::gemm::GemmExecutionContext &context,
    const WorkspaceRequirements &requirements, WorkspaceMap &workspaces) {
  func::FuncOp function = op->getParentOfType<func::FuncOp>();
  Operation *functionOp = function.getOperation();
  if (auto existing = workspaces.find(functionOp); existing != workspaces.end())
    return existing->second;
  const auto &size = requirements.find(functionOp)->second;
  OpBuilder builder(function.getContext());
  builder.setInsertionPointToStart(&function.front());
  MemRefType type;
  ValueRange dynamicSizes;
  if (context.strategy.isa == aarch64::gemm::GemmIsa::kSve) {
    Value k = builder.create<arith::ConstantIndexOp>(op->getLoc(), size.maxK);
    Value n = builder.create<arith::ConstantIndexOp>(op->getLoc(), size.maxN);
    Value elements =
        callSvePackedBHelper(module, builder, op->getLoc(),
                             aarch64::gemm::kSvePackedBElementsAsmSymbol, k, n);
    type = MemRefType::get({ShapedType::kDynamic}, builder.getF32Type());
    dynamicSizes = ValueRange{elements};
  } else {
    type = MemRefType::get({size.maxElements}, builder.getF32Type());
  }
  auto alloca = builder.create<memref::AllocaOp>(op->getLoc(), type, dynamicSizes);
  alloca->setAttr("alignment", builder.getI64IntegerAttr(64));
  workspaces.try_emplace(functionOp, alloca);
  return alloca;
}

Value materializeIndex(OpBuilder &builder, Location loc, OpFoldResult value) {
  if (auto dynamicValue = llvm::dyn_cast<Value>(value)) return dynamicValue;
  auto attr = llvm::dyn_cast<IntegerAttr>(llvm::cast<Attribute>(value));
  return builder.create<arith::ConstantIndexOp>(loc, attr.getInt());
}

bool hasEpilogue(Operation *op) {
  if (!op->getAttrOfType<DictionaryAttr>(atir::kGemmEpilogueAttr))
    return false;
  if (auto role =
          op->getAttrOfType<StringAttr>(aarch64::gemm::kKcRoleAttrName)) {
    if (role.getValue() == "first" || role.getValue() == "middle") return false;
  }
  return true;
}

FailureOr<AffineMap> buildCarrierMap(OpBuilder &builder, MemRefType outputType,
                                     MemRefType inputType, AffineExpr m,
                                     AffineExpr n) {
  auto map = [&](ArrayRef<AffineExpr> results) {
    return AffineMap::get(2, 0, results, builder.getContext());
  };
  auto zero = builder.getAffineConstantExpr(0);
  if (inputType.getRank() == 0) return map({});
  if (inputType.getRank() == 1)
    return inputType.getDimSize(0) == 1 ? map({zero}) : map({n});
  if (inputType.getRank() != 2) return failure();
  if (inputType.getShape() == outputType.getShape()) return map({m, n});
  if (inputType.getDimSize(0) == 1 && inputType.getDimSize(1) == 1)
    return map({zero, zero});
  if (inputType.getDimSize(1) == 1) return map({m, zero});
  if (inputType.getDimSize(0) == 1) return map({zero, n});
  return failure();
}

FailureOr<linalg::GenericOp> materializeEpilogueCarrier(
    Operation *op, OpBuilder &builder, Value output,
    const atir::EpilogueProgram &program) {
  auto outputType = llvm::dyn_cast<MemRefType>(output.getType());
  auto generic = llvm::dyn_cast<linalg::GenericOp>(op);
  if (!outputType || !generic || outputType.getRank() != 2)
    return op->emitOpError("requires a rank-2 output for the epilogue carrier");

  SmallVector<Value> inputs{output};
  SmallVector<AffineMap> maps;
  auto m = builder.getAffineDimExpr(0);
  auto n = builder.getAffineDimExpr(1);
  maps.push_back(AffineMap::get(2, 0, {m, n}, builder.getContext()));
  unsigned binaryInput = 0;
  for (const atir::EpilogueStep &step : program.steps) {
    if (!atir::isBinaryEpilogueOpcode(step.opcode)) continue;
    if (2 + binaryInput >= generic.getInputs().size())
      return op->emitOpError("epilogue steps do not match GEMM inputs");
    Value input = generic.getInputs()[2 + binaryInput++];
    auto inputType = llvm::dyn_cast<MemRefType>(input.getType());
    if (!inputType) return op->emitOpError("epilogue input must be a memref");
    FailureOr<AffineMap> map =
        buildCarrierMap(builder, outputType, inputType, m, n);
    if (failed(map))
      return op->emitOpError("epilogue input has an invalid broadcast shape");
    inputs.push_back(input);
    maps.push_back(*map);
  }
  if (generic.getInputs().size() != 2 + binaryInput)
    return op->emitOpError("epilogue inputs do not match the candidate plan");
  maps.push_back(AffineMap::get(2, 0, {m, n}, builder.getContext()));

  auto carrier = builder.create<linalg::GenericOp>(
      op->getLoc(), TypeRange{}, inputs, ValueRange{output}, maps,
      SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                       utils::IteratorType::parallel},
      [](OpBuilder &nested, Location loc, ValueRange args) {
        // The carrier preserves operands and maps until epilogue lowering.
        nested.create<linalg::YieldOp>(loc, args.front());
      });
  carrier->setAttr(aarch64::gemm::kEpilogueCarrierAttrName,
                   UnitAttr::get(builder.getContext()));
  carrier->setAttr(atir::kGemmEpilogueAttr,
                   atir::buildEpilogueCandidateAttr(builder, program));
  for (StringRef name : {StringRef("withBias"), StringRef("do_relu")})
    if (Attribute attr = op->getAttr(name)) carrier->setAttr(name, attr);
  return carrier;
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
       {aarch64::gemm::kKcModeAttrName, aarch64::gemm::kKcRoleAttrName,
        atir::kGemmEpilogueAttr}) {
    if (Attribute attr = source->getAttr(name))
      target->setDiscardableAttr(name, attr);
  }
  aarch64::gemm::copyStrategyReference(source, target);
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
                                      const aarch64::gemm::GemmExecutionContext &context,
                                      int64_t fullN, int64_t kSize, Value pc,
                                      Value jc) {
  const auto &tile = aarch64::gemm::getFixedCacheParameters(context.strategy.cache);
  const int64_t kc = tile.kc;
  const int64_t nc = tile.nc;
  const int64_t kRows = (context.problem.k + kc - 1) / kc;
  const int64_t nCols = (fullN + nc - 1) / nc;
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
    const int64_t kr = std::min(kc, context.problem.k - r * kc);
    int64_t rowWidth = 0;
    for (int64_t c = 0; c < nCols; ++c) {
      const int64_t blockN = std::min(nc, fullN - c * nc);
      rowWidth += packedBElements(context, kr, blockN);
    }
    addContribution(rowIndex, r, rowWidth);
  }
  // Columns before JC inside the current K row.
  for (int64_t c = 0; c < nCols; ++c) {
    const int64_t blockN = std::min(nc, fullN - c * nc);
    addContribution(colIndex, c, packedBElements(context, kSize, blockN));
  }
  return offset;
}

// Absolute (K, N) origin of a cache block in the original RHS buffer: walk
// the subview chain to the root, accumulating each level's offsets.
FailureOr<std::pair<Value, Value>> absoluteBlockBase(Operation *op,
                                                     Value block) {
  OpBuilder builder(op);
  Location loc = op->getLoc();
  Value kAccum;
  Value nAccum;
  size_t levels = 0;
  Value current = block;
  while (auto subview = current.getDefiningOp<memref::SubViewOp>()) {
    ++levels;
    if (subview.getMixedOffsets().size() != 2) return failure();
    if (!isStaticZero(subview.getMixedOffsets()[0])) {
      Value kOffset =
          materializeIndex(builder, loc, subview.getMixedOffsets()[0]);
      kAccum = kAccum ? builder.create<arith::AddIOp>(loc, kAccum, kOffset)
                      : kOffset;
    }
    if (!isStaticZero(subview.getMixedOffsets()[1])) {
      Value nOffset =
          materializeIndex(builder, loc, subview.getMixedOffsets()[1]);
      nAccum = nAccum ? builder.create<arith::AddIOp>(loc, nAccum, nOffset)
                      : nOffset;
    }
    current = subview.getSource();
  }
  if (levels == 0) return failure();
  if (!kAccum) kAccum = builder.create<arith::ConstantIndexOp>(loc, 0);
  if (!nAccum) nAccum = builder.create<arith::ConstantIndexOp>(loc, 0);
  return std::make_pair(kAccum, nAccum);
}

FailureOr<PackedBBlock> materializePrepackedBBlock(
    ModuleOp module, Operation *op, const aarch64::gemm::GemmExecutionContext &context,
    OpBuilder &builder, Value rhsBlock, Value jr) {
  const auto &tile = aarch64::gemm::getFixedCacheParameters(context.strategy.cache);
  if (context.strategy.isa == aarch64::gemm::GemmIsa::kSve &&
      !annc_aarch64_sve_packed_b_elements_f32) {
    return op->emitOpError(
        "SVE prepacked RHS requires the AArch64 GEMM microkernel archive");
  }
  auto rhsSubview = rhsBlock.getDefiningOp<memref::SubViewOp>();
  if (!rhsSubview || rhsSubview.getMixedOffsets().size() != 2)
    return op->emitOpError(
        "prepacked RHS requires cache-blocked RHS subviews");
  FailureOr<int64_t> kSize = getStaticDimension(rhsBlock, 0);
  if (failed(kSize))
    return op->emitOpError("prepacked RHS requires static cache K blocks");

  OpBuilder &microBuilder = builder;
  Location loc = op->getLoc();
  FailureOr<std::pair<Value, Value>> base = absoluteBlockBase(op, rhsBlock);
  if (failed(base))
    return op->emitOpError("prepacked RHS requires a subview chain rooted at "
                           "the original RHS buffer");
  Value pc = base->first;
  Value blockColumn = base->second;

  // Thread tasks carry a tile-local N; recover the problem-level N that
  // shapes the packed data blocks.
  const int64_t fullN = context.problem.n;

  // packed offset = cache-block prefix (pc row + block column) + panel offset.
  Value offset;
  auto pcConstant = pc.getDefiningOp<arith::ConstantIndexOp>();
  auto columnConstant = blockColumn.getDefiningOp<arith::ConstantIndexOp>();
  if (pcConstant && columnConstant) {
    offset = microBuilder.create<arith::ConstantIndexOp>(
        loc, rowPrefix(context, context.problem.k, fullN, tile,
                       pcConstant.value() / tile.kc) +
                 columnPrefix(context, *kSize, fullN, tile.nc,
                              columnConstant.value() / tile.nc));
  } else {
    offset = materializePrepackedCacheOffset(microBuilder, loc, context, fullN,
                                             *kSize, pc, blockColumn);
  }
  // Panel numbering restarts at each NC-aligned packed block, so a task tile
  // starting mid-block must rebase JR into that block's column space.
  Value panelColumn = jr;
  if (!columnConstant || columnConstant.value() % tile.nc != 0) {
    Value ncValue = microBuilder.create<arith::ConstantIndexOp>(
        loc, tile.nc);
    Value blockStart = microBuilder.create<arith::MulIOp>(
        loc, microBuilder.create<arith::DivUIOp>(loc, blockColumn, ncValue),
        ncValue);
    panelColumn = microBuilder.create<arith::AddIOp>(
        loc, microBuilder.create<arith::SubIOp>(loc, blockColumn, blockStart),
        jr);
  }
  if (context.strategy.isa == aarch64::gemm::GemmIsa::kSve) {
    Value blockK = microBuilder.create<arith::ConstantIndexOp>(loc, *kSize);
    Value panel = callSvePackedBHelper(
        module, microBuilder, loc, aarch64::gemm::kSvePackedBOffsetAsmSymbol,
        blockK, panelColumn);
    offset = microBuilder.create<arith::AddIOp>(loc, offset, panel);
  } else {
    Value group = microBuilder.create<arith::DivUIOp>(
        loc, panelColumn, microBuilder.create<arith::ConstantIndexOp>(loc, 4));
    Value groupOffset = microBuilder.create<arith::MulIOp>(
        loc, group,
        microBuilder.create<arith::ConstantIndexOp>(loc, *kSize * 4));
    Value lane = microBuilder.create<arith::AndIOp>(
        loc, panelColumn, microBuilder.create<arith::ConstantIndexOp>(loc, 3));
    offset = microBuilder.create<arith::AddIOp>(
        loc, offset, microBuilder.create<arith::AddIOp>(loc, groupOffset, lane));
  }

  if (context.prepackedRhsFromInput) {
    Value root = rhsBlock;
    while (auto subview = root.getDefiningOp<memref::SubViewOp>())
      root = subview.getSource();
    // The thread dispatcher passes the original descriptor, not a sliced
    // packed pointer. Subviews above describe coordinates only.
    return PackedBBlock{
        aarch64::gemm::castToUnrankedF32MemRef(microBuilder, loc, root),
        offset};
  }
  FailureOr<Value> global =
      getPrepackedRhsGlobal(module, op, context, microBuilder, loc);
  if (failed(global)) return failure();
  return PackedBBlock{
      aarch64::gemm::castToUnrankedF32MemRef(microBuilder, loc, *global),
      offset};
}

FailureOr<PackedBBlock> materializePackedBBlock(
    ModuleOp module, Operation *op, const aarch64::gemm::GemmExecutionContext &context,
    const WorkspaceRequirements &requirements, WorkspaceMap &workspaces,
    PackedBBlockMap &packedBlocks) {
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

      if (context.strategy.rhsPacking == aarch64::gemm::RhsPacking::kPrepacked)
        return materializePrepackedBBlock(module, op, context, microBuilder,
                                          rhsBlock, jr);

      OpBuilder packBuilder(hoistPoint);
      FailureOr<aarch64::gemm::MemRefBaseAndOffset> rhs =
          aarch64::gemm::getMemRefBaseAndOffset(packBuilder, loc, rhsBlock);
      if (failed(rhs)) {
        return op->emitOpError(
            "requires identity-layout bases with statically strided "
            "(PC, JC) RHS subviews");
      }

      FailureOr<int64_t> staticK = getStaticDimension(rhsBlock, 0);
      FailureOr<int64_t> staticN = getStaticDimension(rhsBlock, 1);
      if (failed(staticK) || failed(staticN))
        return op->emitOpError("requires statically shaped RHS cache blocks");
      Value workspace = getOrCreatePackedBWorkspace(
          module, op, context, requirements, workspaces);
      Value kSize = packBuilder.create<arith::ConstantIndexOp>(loc, *staticK);
      Value nSize = packBuilder.create<arith::ConstantIndexOp>(loc, *staticN);
      Value ldbValue =
          packBuilder.create<arith::ConstantIndexOp>(loc, context.problem.ldb);
      Value packedBase = materializePackCall(
          module, packBuilder, loc, rhs->base, rhs->offset, workspace, kSize,
          nSize, ldbValue, context.strategy.target, context.strategy.isa);
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
  if (context.strategy.rhsPacking == aarch64::gemm::RhsPacking::kPrepacked)
    return op->emitOpError(
        "prepacked RHS requires cache-blocked GEMM structure");
  FailureOr<int64_t> staticK = getStaticDimension(rhsInput, 0);
  FailureOr<int64_t> staticN = getStaticDimension(rhsInput, 1);
  if (failed(staticK) || failed(staticN)) {
    return op->emitOpError("requires statically shaped RHS microtiles");
  }
  Value workspace = getOrCreatePackedBWorkspace(
      module, op, context, requirements, workspaces);
  Value kSize = builder.create<arith::ConstantIndexOp>(loc, *staticK);
  Value nSize = builder.create<arith::ConstantIndexOp>(loc, *staticN);
  Value ldbValue = builder.create<arith::ConstantIndexOp>(loc, context.problem.ldb);
  Value packedBase = materializePackCall(module, builder, loc, rhs->base,
                                         rhs->offset, workspace, kSize, nSize,
                                         ldbValue, context.strategy.target,
                                         context.strategy.isa);
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  return PackedBBlock{packedBase, zero};
}

LogicalResult materializeLeafCalls(ModuleOp module, Operation *op,
                                   const WorkspaceRequirements &requirements,
                                   WorkspaceMap &workspaces,
                                   PackedBBlockMap &packedBlocks) {
  FailureOr<aarch64::gemm::GemmExecutionContext> context =
      aarch64::gemm::readExecutionContext(op);
  if (failed(context) || failed(aarch64::gemm::requireStage(
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
      context->strategy.rhsPacking == aarch64::gemm::RhsPacking::kDirect;
  const bool usesLdbAbi = aarch64::gemm::usesLdbAbi(
    aarch64::gemm::getGemmLeafKind(context->strategy.executionKind,
                                   context->strategy.rhsPacking));
  const bool scalarK =
      usesLdbAbi &&
      context->strategy.executionKind == aarch64::gemm::GemmExecutionKind::kGemm;
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
        materializePackedBBlock(module, op, *context, requirements, workspaces,
                                packedBlocks);
    if (failed(packed)) return failure();
    rhsBase = packed->base;
    rhsOffset = packed->offset;
  }

  FailureOr<int64_t> staticM = getStaticDimension(lhsInput, 0);
  FailureOr<int64_t> staticK = getStaticDimension(lhsInput, 1);
  FailureOr<int64_t> staticN = getStaticDimension(rhsInput, 1);
  if (failed(staticM) || failed(staticK) || failed(staticN) || *staticM < 1 ||
      *staticK < 1 || *staticN < 1) {
    return op->emitOpError(
        "requires positive static M/N/K microtile dimensions");
  }

  Value lhsBase =
      aarch64::gemm::castToUnrankedF32MemRef(builder, loc, lhs->base);
  Value outBase =
      aarch64::gemm::castToUnrankedF32MemRef(builder, loc, out->base);

  const bool hasCarrier = hasEpilogue(op);

  Value lda = builder.create<arith::ConstantIndexOp>(loc, context->problem.lda);
  Value ldb = builder.create<arith::ConstantIndexOp>(loc, context->problem.ldb);
  Value ldc = builder.create<arith::ConstantIndexOp>(loc, context->problem.ldc);
  Value kSize = builder.create<arith::ConstantIndexOp>(loc, *staticK);
  Value familyArgument;
  if (context->strategy.isa == aarch64::gemm::GemmIsa::kSve) {
    familyArgument = builder.create<arith::ConstantIndexOp>(loc, *staticN);
  } else {
    const aarch64::gemm::GemmKernelABI &abi =
        aarch64::gemm::getGemmKernelABI(context->strategy);
    FailureOr<int64_t> kScalarUnroll =
        aarch64::gemm::getGemmKScalarUnroll(abi, context->strategy.dataType);
    if (failed(kScalarUnroll))
      return op->emitOpError(
          "requires a valid NEON kernel ABI vector K unroll");
    familyArgument = builder.create<arith::ConstantIndexOp>(
        loc, scalarK ? *staticK : *staticK / *kScalarUnroll);
  }

  auto unranked = UnrankedMemRefType::get(builder.getF32Type(), 0);
  auto index = builder.getIndexType();
  StringRef leafName =
      usesLdbAbi
          ? aarch64::gemm::kMicrokernelRmLeafName
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
  // The epilogue remains on the carrier until the selected leaf has been
  // inspected.  A pure GEMM call must not look fused to later lowering.
  kernelCall->removeAttr(atir::kGemmEpilogueAttr);
  if (hasCarrier)
    kernelCall->setAttr(aarch64::gemm::kEpilogueInsertionPointAttrName,
                        UnitAttr::get(builder.getContext()));
  kernelCall->setDiscardableAttr(aarch64::gemm::kMicrokernelMAttrName,
                                 builder.getI64IntegerAttr(*staticM));
  kernelCall->setDiscardableAttr(aarch64::gemm::kMicrokernelNAttrName,
                                 builder.getI64IntegerAttr(*staticN));
  kernelCall->setDiscardableAttr(aarch64::gemm::kMicrokernelKAttrName,
                                 builder.getI64IntegerAttr(*staticK));
  FailureOr<aarch64::gemm::FinalMicrokernelSelection> selection =
      aarch64::gemm::selectFinalMicrokernel(kernelCall);
  if (failed(selection) ||
      failed(aarch64::gemm::attachFinalMicrokernelSelection(kernelCall,
                                                            *selection)))
    return failure();

  if (hasCarrier) {
    auto program = atir::parseEpilogueCandidatePlan(
        op->getAttrOfType<DictionaryAttr>(atir::kGemmEpilogueAttr));
    if (failed(program))
      return op->emitOpError("has an invalid fixed epilogue plan");
    builder.setInsertionPointAfter(kernelCall);
    if (failed(materializeEpilogueCarrier(op, builder, output, *program)))
      return failure();
  }
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
    WorkspaceRequirements requirements;
    for (Operation *op : candidates) {
      FailureOr<aarch64::gemm::GemmExecutionContext> context =
          aarch64::gemm::readExecutionContext(op);
      if (failed(context)) {
        signalPassFailure();
        return;
      }
      if (context->strategy.rhsPacking == aarch64::gemm::RhsPacking::kDirect ||
          context->strategy.rhsPacking == aarch64::gemm::RhsPacking::kPrepacked)
        continue;
      Value rhs = aarch64::gemm::getGemmInput(op, 1);
      if (auto tile = rhs.getDefiningOp<memref::SubViewOp>())
        rhs = tile.getSource();
      FailureOr<int64_t> k = getStaticDimension(rhs, 0);
      FailureOr<int64_t> n = getStaticDimension(rhs, 1);
      if (failed(k) || failed(n)) {
        op->emitOpError("requires statically shaped packed RHS blocks");
        signalPassFailure();
        return;
      }
      auto &size = requirements[op->getParentOfType<func::FuncOp>()];
      size.maxK = std::max(size.maxK, *k);
      size.maxN = std::max(size.maxN, *n);
      const auto &abi = aarch64::gemm::getGemmKernelABI(context->strategy);
      size.maxElements = aarch64::gemm::updateGemmPackedBWorkspaceElements(
          size.maxElements, abi, *k, *n);
    }
    WorkspaceMap workspaces;
    PackedBBlockMap packedBlocks;
    for (Operation *op : candidates) {
      if (failed(materializeLeafCalls(module, op, requirements, workspaces,
                                      packedBlocks))) {
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
