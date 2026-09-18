#include <string>

#include "GemmPlan.h"
#include "GemmTilingUtils.h"
#include "Target/aarch64/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

namespace annc {
namespace {

constexpr llvm::StringLiteral kGemmParallelFor =
    "annc_threadpool_parallel_for_gemm";

func::FuncOp getOrCreateGemmParallelForDeclaration(ModuleOp module,
                                                   FunctionType taskType) {
  StringRef name = kGemmParallelFor;
  if (auto declaration = module.lookupSymbol<func::FuncOp>(name))
    return declaration;

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  Type i64 = builder.getI64Type();
  MemRefType dynamicF32MemRef = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, builder.getF32Type());
  SmallVector<Type> inputs = {i64, taskType, dynamicF32MemRef, dynamicF32MemRef,
                              dynamicF32MemRef};
  auto declaration = builder.create<func::FuncOp>(
      module.getLoc(), name, builder.getFunctionType(inputs, {}));
  declaration.setPrivate();
  declaration->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                       builder.getUnitAttr());
  return declaration;
}

std::string getUniqueTaskName(ModuleOp module) {
  for (unsigned ordinal = 0;; ++ordinal) {
    std::string name = "__annc_gemm_thread_task_" + std::to_string(ordinal);
    if (!module.lookupSymbol(name)) return name;
  }
}

LogicalResult validateThreadTiling(Operation *gemm,
                                   const aarch64::gemm::GemmTilingPlan &plan) {
  if (auto generic = llvm::dyn_cast<linalg::GenericOp>(gemm)) {
    if (failed(aarch64::gemm::validateGemmGeneric(generic))) return failure();
    if (generic.getInputs().size() != 2) {
      return generic.emitOpError(
          "thread tiling supports only ordinary GEMM with A and B inputs");
    }
  }
  if (gemm->getNumResults() != 0)
    return gemm->emitOpError("requires buffer-semantics GEMM anchors");

  auto lhsType = llvm::dyn_cast<MemRefType>(
      aarch64::gemm::getGemmInput(gemm, 0).getType());
  auto rhsType = llvm::dyn_cast<MemRefType>(
      aarch64::gemm::getGemmInput(gemm, 1).getType());
  auto outType = llvm::dyn_cast<MemRefType>(
      aarch64::gemm::getGemmOutput(gemm, 0).getType());
  if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || outType.getRank() != 2 ||
      !lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !outType.hasStaticShape()) {
    return gemm->emitOpError(
        "requires statically shaped bufferized memref operands");
  }
  if (lhsType.getShape() != ArrayRef<int64_t>{plan.m, plan.k} ||
      rhsType.getShape() != ArrayRef<int64_t>{plan.k, plan.n} ||
      outType.getShape() != ArrayRef<int64_t>{plan.m, plan.n}) {
    return gemm->emitOpError("has plan dimensions inconsistent with operands");
  }
  return success();
}

DictionaryAttr getTilePlan(Operation *gemm, OpBuilder &builder, int64_t mSize,
                           int64_t nSize) {
  auto originalPlan =
      gemm->getAttrOfType<DictionaryAttr>(aarch64::gemm::kPlanAttrName);
  NamedAttrList tilePlan(originalPlan);
  // M/N are local task dimensions. Leading dimensions remain those of the
  // original matrices because all subviews retain the original row strides.
  tilePlan.set("m", builder.getI64IntegerAttr(mSize));
  tilePlan.set("n", builder.getI64IntegerAttr(nSize));
  // Prepacked-RHS leaf materialization indexes packed data laid out over the
  // whole problem, so task-local GEMMs keep the problem-level N available.
  if (auto fullN = originalPlan.get("n")) tilePlan.set("full_n", fullN);
  tilePlan.set("thread_count", builder.getI64IntegerAttr(1));
  tilePlan.set("tasks_m", builder.getI64IntegerAttr(1));
  tilePlan.set("tasks_n", builder.getI64IntegerAttr(1));
  // The outer dispatcher is already materialized; every nested GEMM stays
  // single-task.
  tilePlan.set("shard_direction", builder.getStringAttr("rows"));
  tilePlan.set("thread_partition", builder.getStringAttr("static-2d"));
  return tilePlan.getDictionary(gemm->getContext());
}

LogicalResult createThreadTile(OpBuilder &builder, Operation *gemm,
                               const aarch64::gemm::GemmTilingPlan &plan,
                               Block *entry, Value mOffset, Value nOffset,
                               int64_t mSize, int64_t nSize) {
  Location loc = gemm->getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  if (auto generic = llvm::dyn_cast<linalg::GenericOp>(gemm)) {
    SmallVector<Value> operands;
    operands.append(generic.getInputs().begin(), generic.getInputs().end());
    operands.append(generic.getOutputs().begin(), generic.getOutputs().end());
    IRMapping taskMapping;
    for (auto [operand, argument] :
         llvm::zip_equal(operands, entry->getArguments().drop_front(1))) {
      taskMapping.map(operand, argument);
    }
    auto taskGeneric = llvm::cast<linalg::GenericOp>(
        builder.clone(*generic.getOperation(), taskMapping));
    SmallVector<OpFoldResult> offsets = {mOffset, nOffset, zero};
    SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(mSize),
                                       builder.getIndexAttr(nSize),
                                       builder.getIndexAttr(plan.k)};
    FailureOr<linalg::GenericOp> tiled = aarch64::gemm::cloneTiledGemmGeneric(
        builder, taskGeneric, offsets, sizes);
    taskGeneric.erase();
    if (failed(tiled)) return failure();
    (*tiled)->setAttr(aarch64::gemm::kPlanAttrName,
                      getTilePlan(gemm, builder, mSize, nSize));
    aarch64::gemm::setStage(*tiled, builder, aarch64::gemm::kThreadTiledStage);
    return success();
  }

  SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1),
                                       builder.getIndexAttr(1)};
  Value lhsTile = builder.create<memref::SubViewOp>(
      loc, entry->getArgument(1), SmallVector<OpFoldResult>{mOffset, zero},
      SmallVector<OpFoldResult>{builder.getIndexAttr(mSize),
                                builder.getIndexAttr(plan.k)},
      strides);
  Value rhsTile = builder.create<memref::SubViewOp>(
      loc, entry->getArgument(2), SmallVector<OpFoldResult>{zero, nOffset},
      SmallVector<OpFoldResult>{builder.getIndexAttr(plan.k),
                                builder.getIndexAttr(nSize)},
      strides);
  Value outTile = builder.create<memref::SubViewOp>(
      loc, entry->getArgument(3), SmallVector<OpFoldResult>{mOffset, nOffset},
      SmallVector<OpFoldResult>{builder.getIndexAttr(mSize),
                                builder.getIndexAttr(nSize)},
      strides);
  auto tiled = builder.create<linalg::MatmulOp>(
      loc, ValueRange{lhsTile, rhsTile}, ValueRange{outTile});
  tiled->setAttrs(gemm->getAttrs());
  tiled->setAttr(aarch64::gemm::kPlanAttrName,
                 getTilePlan(gemm, builder, mSize, nSize));
  aarch64::gemm::setStage(tiled, builder, aarch64::gemm::kThreadTiledStage);
  return success();
}

LogicalResult createThreadTask(ModuleOp module, Operation *gemm,
                               const aarch64::gemm::GemmTilingPlan &plan,
                               FunctionType taskType, func::FuncOp &task) {
  OpBuilder moduleBuilder(module.getContext());
  moduleBuilder.setInsertionPointToEnd(module.getBody());
  task = moduleBuilder.create<func::FuncOp>(
      gemm->getLoc(), getUniqueTaskName(module), taskType);
  task.setPrivate();
  Block *entry = task.addEntryBlock();

  OpBuilder builder(entry, entry->begin());
  Location loc = gemm->getLoc();
  Type index = builder.getIndexType();
  Value taskId =
      builder.create<arith::IndexCastOp>(loc, index, entry->getArgument(0));
  Value tasksM = builder.create<arith::ConstantIndexOp>(loc, plan.tasksM);
  Value tasksN = builder.create<arith::ConstantIndexOp>(loc, plan.tasksN);
  Value mTask;
  Value nTask;
  if (plan.shardByColumns) {
    mTask = builder.create<arith::RemSIOp>(loc, taskId, tasksM);
    nTask = builder.create<arith::DivSIOp>(loc, taskId, tasksM);
  } else {
    mTask = builder.create<arith::DivSIOp>(loc, taskId, tasksN);
    nTask = builder.create<arith::RemSIOp>(loc, taskId, tasksN);
  }

  // Task ranges shared with the planner: N extents truncate to sixteen
  // columns (one output-row cache line) and M extents truncate to whole
  // microkernel rows; the final task of each dimension absorbs the remainder.
  aarch64::gemm::GemmTaskSplit splitM;
  aarch64::gemm::GemmTaskSplit splitN;
  if (!aarch64::gemm::splitGemmTaskRange(plan.m, plan.tasksM,
                                         plan.kernelTile.mr, splitM) ||
      !aarch64::gemm::splitGemmTaskRange(plan.n, plan.tasksN,
                                         aarch64::gemm::kGemmTaskSplitAlignment,
                                         splitN)) {
    return gemm->emitOpError(
        "has a thread grid that cannot be split into aligned task extents");
  }

  auto makeRange = [&](Value taskIndex, std::int64_t taskCount,
                       std::int64_t base, Value &offset, Value &isLast) {
    Value baseValue = builder.create<arith::ConstantIndexOp>(loc, base);
    offset = builder.create<arith::MulIOp>(loc, taskIndex, baseValue);
    Value lastIndex =
        builder.create<arith::ConstantIndexOp>(loc, taskCount - 1);
    isLast = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           taskIndex, lastIndex);
  };

  Value mOffset, mIsLast;
  Value nOffset, nIsLast;
  makeRange(mTask, plan.tasksM, splitM.base, mOffset, mIsLast);
  makeRange(nTask, plan.tasksN, splitN.base, nOffset, nIsLast);

  const int64_t mBase = splitM.base;
  const int64_t nBase = splitN.base;
  const int64_t mTail = splitM.tail;
  const int64_t nTail = splitN.tail;
  auto emitNVariants = [&](OpBuilder &nestedBuilder, int64_t staticMSize,
                           Value nestedMOffset) -> LogicalResult {
    if (nTail == nBase)
      return createThreadTile(nestedBuilder, gemm, plan, entry, nestedMOffset,
                              nOffset, staticMSize, nBase);
    auto nIf = nestedBuilder.create<scf::IfOp>(loc, nIsLast,
                                               /*withElseRegion=*/true);
    {
      OpBuilder thenBuilder =
          OpBuilder::atBlockBegin(&nIf.getThenRegion().front());
      if (failed(createThreadTile(thenBuilder, gemm, plan, entry, nestedMOffset,
                                  nOffset, staticMSize, nTail)))
        return failure();
    }
    {
      OpBuilder elseBuilder =
          OpBuilder::atBlockBegin(&nIf.getElseRegion().front());
      if (failed(createThreadTile(elseBuilder, gemm, plan, entry, nestedMOffset,
                                  nOffset, staticMSize, nBase)))
        return failure();
    }
    return success();
  };

  if (mTail == mBase) {
    if (failed(emitNVariants(builder, mBase, mOffset))) return failure();
  } else {
    auto mIf = builder.create<scf::IfOp>(loc, mIsLast,
                                         /*withElseRegion=*/true);
    {
      OpBuilder thenBuilder =
          OpBuilder::atBlockBegin(&mIf.getThenRegion().front());
      if (failed(emitNVariants(thenBuilder, mTail, mOffset))) return failure();
    }
    {
      OpBuilder elseBuilder =
          OpBuilder::atBlockBegin(&mIf.getElseRegion().front());
      if (failed(emitNVariants(elseBuilder, mBase, mOffset))) return failure();
    }
  }

  builder.setInsertionPointToEnd(entry);
  builder.create<func::ReturnOp>(loc);
  return success();
}

LogicalResult materializeThreadTiling(Operation *gemm) {
  FailureOr<aarch64::gemm::GemmTilingPlan> plan =
      aarch64::gemm::readTilingPlan(gemm);
  if (failed(plan)) return failure();
  if (failed(validateThreadTiling(gemm, *plan))) return failure();

  // A finalized adaptive plan may select one thread even when the deployment
  // budget is larger. Keep that path entirely serial: cache blocking can
  // consume the original full GEMM once it is marked thread-tiled, while an
  // outlined one-item dispatcher only adds context and thread-pool overhead.
  if (plan->threadCount == 1) {
    OpBuilder builder(gemm);
    aarch64::gemm::setStage(gemm, builder, aarch64::gemm::kThreadTiledStage);
    return success();
  }

  ModuleOp module = gemm->getParentOfType<ModuleOp>();
  OpBuilder builder(gemm);
  Location loc = gemm->getLoc();
  Type i64 = builder.getI64Type();
  MemRefType dynamicF32MemRef = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, builder.getF32Type());
  SmallVector<Type> taskInputs = {i64, dynamicF32MemRef, dynamicF32MemRef,
                                  dynamicF32MemRef};
  FunctionType taskType = builder.getFunctionType(taskInputs, {});
  func::FuncOp task;
  if (failed(createThreadTask(module, gemm, *plan, taskType, task)))
    return failure();
  func::FuncOp dispatcher =
      getOrCreateGemmParallelForDeclaration(module, taskType);

  OperationState state(loc, func::ConstantOp::getOperationName());
  state.addTypes(taskType);
  state.addAttribute(
      "value", FlatSymbolRefAttr::get(module.getContext(), task.getName()));
  Value taskReference = builder.create(state)->getResult(0);
  Value workCount = builder.create<arith::ConstantIntOp>(
      loc, plan->tasksM * plan->tasksN, 64);
  Value lhs = builder.create<memref::CastOp>(
      loc, dynamicF32MemRef, aarch64::gemm::getGemmInput(gemm, 0));
  Value rhs = builder.create<memref::CastOp>(
      loc, dynamicF32MemRef, aarch64::gemm::getGemmInput(gemm, 1));
  SmallVector<Value> callOperands = {workCount, taskReference, lhs, rhs};
  Value out = builder.create<memref::CastOp>(
      loc, dynamicF32MemRef, aarch64::gemm::getGemmOutput(gemm, 0));
  callOperands.push_back(out);
  builder.create<func::CallOp>(loc, dispatcher, callOperands);

  gemm->erase();
  return success();
}

class AArch64GemmThreadTiling
    : public AArch64GemmThreadTilingBase<AArch64GemmThreadTiling> {
 public:
  using Base::Base;

  void runOnOperation() override {
    SmallVector<Operation *> candidates;
    getOperation().walk([&](Operation *op) {
      if (aarch64::gemm::isGemmAnchor(op) &&
          op->hasAttr(aarch64::gemm::kPlanAttrName) &&
          !op->hasAttr(aarch64::gemm::kStageAttrName)) {
        candidates.push_back(op);
      }
    });
    for (Operation *gemm : candidates) {
      if (failed(materializeThreadTiling(gemm))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64GemmThreadTiling() {
  return std::make_unique<AArch64GemmThreadTiling>();
}

}  // namespace annc
