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
constexpr llvm::StringLiteral kGemmParallelForEpilogue =
    "annc_threadpool_parallel_for_gemm_epilogue";

func::FuncOp getOrCreateGemmParallelForDeclaration(ModuleOp module,
                                                   FunctionType taskType,
                                                   bool withEpilogueInputs) {
  StringRef name =
      withEpilogueInputs ? kGemmParallelForEpilogue : kGemmParallelFor;
  if (auto declaration = module.lookupSymbol<func::FuncOp>(name))
    return declaration;

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  Type i64 = builder.getI64Type();
  MemRefType dynamicF32MemRef = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, builder.getF32Type());
  SmallVector<Type> inputs = {i64, taskType, dynamicF32MemRef, dynamicF32MemRef,
                              dynamicF32MemRef};
  if (withEpilogueInputs) {
    Type unrankedF32MemRef = UnrankedMemRefType::get(builder.getF32Type(), 0);
    inputs.append(aarch64::gemm::kEpilogueArgsPointerSlots,
                  unrankedF32MemRef);
  }
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
                                   const aarch64::gemm::GemmProblem &problem) {
  if (auto generic = llvm::dyn_cast<linalg::GenericOp>(gemm)) {
    if (failed(aarch64::gemm::validateGemmGeneric(generic))) return failure();
    if (generic.getInputs().size() >
        2 + aarch64::gemm::kEpilogueArgsPointerSlots) {
      return generic.emitOpError(
          "has more epilogue inputs than the thread task ABI supports");
    }
  }
  if (gemm->getNumResults() != 0)
    return gemm->emitOpError("requires buffer-semantics GEMM anchors");

  FailureOr<aarch64::gemm::GemmBlockShape> shape =
      aarch64::gemm::readGemmBlockShape(gemm);
  if (failed(shape)) return failure();
  if (shape->m != problem.m || shape->n != problem.n ||
      shape->k != problem.k)
    return gemm->emitOpError("has problem dimensions inconsistent with operands");
  return success();
}

LogicalResult createThreadTile(OpBuilder &builder, Operation *gemm,
                               const aarch64::gemm::GemmProblem &problem,
                               Block *entry, Value mOffset, Value nOffset,
                               int64_t mSize, int64_t nSize) {
  Location loc = gemm->getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  if (auto generic = llvm::dyn_cast<linalg::GenericOp>(gemm)) {
    IRMapping taskMapping;
    taskMapping.map(generic.getInputs()[0], entry->getArgument(1));
    taskMapping.map(generic.getInputs()[1], entry->getArgument(2));
    taskMapping.map(generic.getOutputs()[0], entry->getArgument(3));
    for (auto [index, input] :
         llvm::enumerate(generic.getInputs().drop_front(2))) {
      auto inputType = llvm::dyn_cast<MemRefType>(input.getType());
      if (!inputType || !inputType.getElementType().isF32())
        return generic.emitOpError(
            "threaded epilogue inputs must be ranked f32 memrefs");
      Value rankedInput = builder.create<memref::CastOp>(
          loc, inputType, entry->getArgument(4 + index));
      taskMapping.map(input, rankedInput);
    }
    auto taskGeneric = llvm::cast<linalg::GenericOp>(
        builder.clone(*generic.getOperation(), taskMapping));
    FailureOr<linalg::GenericOp> tiled = aarch64::gemm::createTiledGemmGeneric(
        builder, taskGeneric, mOffset, nOffset, zero, mSize, nSize, problem.k,
        aarch64::gemm::kThreadTiledStage);
    taskGeneric.erase();
    if (failed(tiled)) return failure();
    return success();
  }

  auto tiled = aarch64::gemm::createTiledMatmul(
      builder, loc, entry->getArgument(1), entry->getArgument(2),
      entry->getArgument(3), mOffset, nOffset, zero, mSize, nSize, problem.k,
      gemm->getAttrs());
  aarch64::gemm::setStage(tiled, builder, aarch64::gemm::kThreadTiledStage);
  return success();
}

LogicalResult createThreadTask(ModuleOp module, Operation *gemm,
                               const aarch64::gemm::GemmProblem &problem,
                               const aarch64::gemm::GemmCandidate &candidate,
                               FunctionType taskType, func::FuncOp &task) {
  OpBuilder moduleBuilder(module.getContext());
  moduleBuilder.setInsertionPointToEnd(module.getBody());
  task = moduleBuilder.create<func::FuncOp>(
      gemm->getLoc(), getUniqueTaskName(module), taskType);
  task.setPrivate();
  aarch64::gemm::copyGemmMetadata(gemm, task);
  Block *entry = task.addEntryBlock();

  OpBuilder builder(entry, entry->begin());
  Location loc = gemm->getLoc();
  Type index = builder.getIndexType();
  Value taskId =
      builder.create<arith::IndexCastOp>(loc, index, entry->getArgument(0));
  Value tasksM = builder.create<arith::ConstantIndexOp>(loc, candidate.tasksM);
  Value tasksN = builder.create<arith::ConstantIndexOp>(loc, candidate.tasksN);
  Value mTask;
  Value nTask;
  if (candidate.shardByColumns) {
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
  const auto &kernelStrategy = candidate.strategy.kernel;
  const int64_t rowAlignment =
      aarch64::gemm::getGemmKernelTilingStrategy(kernelStrategy.kind)
          .getTaskRowAlignment(kernelStrategy);
  if (!aarch64::gemm::splitGemmTaskRange(problem.m, candidate.tasksM, rowAlignment,
                                         splitM) ||
      !aarch64::gemm::splitGemmTaskRange(problem.n, candidate.tasksN,
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
  makeRange(mTask, candidate.tasksM, splitM.base, mOffset, mIsLast);
  makeRange(nTask, candidate.tasksN, splitN.base, nOffset, nIsLast);

  const int64_t mBase = splitM.base;
  const int64_t nBase = splitN.base;
  const int64_t mTail = splitM.tail;
  const int64_t nTail = splitN.tail;
  auto emitNVariants = [&](OpBuilder &nestedBuilder, int64_t staticMSize,
                           Value nestedMOffset) -> LogicalResult {
    if (nTail == nBase)
      return createThreadTile(nestedBuilder, gemm, problem, entry, nestedMOffset,
                              nOffset, staticMSize, nBase);
    auto nIf = nestedBuilder.create<scf::IfOp>(loc, nIsLast,
                                               /*withElseRegion=*/true);
    {
      OpBuilder thenBuilder =
          OpBuilder::atBlockBegin(&nIf.getThenRegion().front());
      if (failed(createThreadTile(thenBuilder, gemm, problem, entry, nestedMOffset,
                                  nOffset, staticMSize, nTail)))
        return failure();
    }
    {
      OpBuilder elseBuilder =
          OpBuilder::atBlockBegin(&nIf.getElseRegion().front());
      if (failed(createThreadTile(elseBuilder, gemm, problem, entry, nestedMOffset,
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
  FailureOr<aarch64::gemm::GemmProblem> problem =
      aarch64::gemm::readProblem(gemm);
  FailureOr<aarch64::gemm::GemmCandidate> candidate =
      aarch64::gemm::readCandidate(gemm);
  if (failed(problem) || failed(candidate)) return failure();
  if (failed(validateThreadTiling(gemm, *problem))) return failure();

  // Thread planning may select one thread even when the deployment
  // budget is larger. Keep that path entirely serial: cache blocking can
  // consume the original full GEMM once it is marked thread-tiled, while an
  // outlined one-item dispatcher only adds context and thread-pool overhead.
  if (candidate->threadCount == 1) {
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
  auto generic = llvm::dyn_cast<linalg::GenericOp>(gemm);
  const unsigned epilogueInputCount =
      generic ? generic.getInputs().size() - 2 : 0;
  const bool withEpilogueInputs = epilogueInputCount != 0;
  Type unrankedF32MemRef = UnrankedMemRefType::get(builder.getF32Type(), 0);
  if (withEpilogueInputs)
    taskInputs.append(aarch64::gemm::kEpilogueArgsPointerSlots,
                      unrankedF32MemRef);
  FunctionType taskType = builder.getFunctionType(taskInputs, {});
  func::FuncOp task;
  if (failed(createThreadTask(module, gemm, *problem, *candidate, taskType, task)))
    return failure();
  func::FuncOp dispatcher =
      getOrCreateGemmParallelForDeclaration(module, taskType,
                                             withEpilogueInputs);

  OperationState state(loc, func::ConstantOp::getOperationName());
  state.addTypes(taskType);
  state.addAttribute(
      "value", FlatSymbolRefAttr::get(module.getContext(), task.getName()));
  Value taskReference = builder.create(state)->getResult(0);
  Value workCount = builder.create<arith::ConstantIntOp>(
      loc, candidate->tasksM * candidate->tasksN, 64);
  Value lhs = builder.create<memref::CastOp>(
      loc, dynamicF32MemRef, aarch64::gemm::getGemmInput(gemm, 0));
  Value rhs = builder.create<memref::CastOp>(
      loc, dynamicF32MemRef, aarch64::gemm::getGemmInput(gemm, 1));
  SmallVector<Value> callOperands = {workCount, taskReference, lhs, rhs};
  Value out = builder.create<memref::CastOp>(
      loc, dynamicF32MemRef, aarch64::gemm::getGemmOutput(gemm, 0));
  callOperands.push_back(out);
  if (withEpilogueInputs) {
    for (Value input : generic.getInputs().drop_front(2))
      callOperands.push_back(aarch64::gemm::castToUnrankedF32MemRef(
          builder, loc, input));
    Value unused = aarch64::gemm::castToUnrankedF32MemRef(
        builder, loc, aarch64::gemm::getGemmInput(gemm, 0));
    callOperands.append(aarch64::gemm::kEpilogueArgsPointerSlots -
                            epilogueInputCount,
                        unused);
  }
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
          op->hasAttr(aarch64::gemm::kCandidateAttrName) &&
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
    aarch64::gemm::pruneGemmMetadata(getOperation());
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64GemmThreadTiling() {
  return std::make_unique<AArch64GemmThreadTiling>();
}

}  // namespace annc
