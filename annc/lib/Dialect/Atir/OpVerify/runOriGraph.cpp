#include "Dialect/Atir/OpVerify/runOriGraph.h"
#include "Dialect/Atir/AtirOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

// 定义宏：新增 OP
#define DISPATCH_ATIR_OP(OpTy) \
  .Case<atir::OpTy>([](auto op) { op.Interpret(); return true; })

using namespace llvm;
using namespace mlir;

#define DEBUG_TYPE "run-ori-graph"

// 算子分发函数
static bool dispatchOp(Operation &op) {
  return mlir::TypeSwitch<Operation *, bool>(&op)
      DISPATCH_ATIR_OP(AddOp)
      DISPATCH_ATIR_OP(SubOp)
      DISPATCH_ATIR_OP(MulOp)
      DISPATCH_ATIR_OP(BufferOp)
      DISPATCH_ATIR_OP(CustomizeOp)
      DISPATCH_ATIR_OP(ConcatOp)
      DISPATCH_ATIR_OP(ConcatV2Op)
      DISPATCH_ATIR_OP(ConstantOp)
      DISPATCH_ATIR_OP(ParameterOp)
      DISPATCH_ATIR_OP(ExpandDimsOp)
      DISPATCH_ATIR_OP(FillOp)
      DISPATCH_ATIR_OP(FloorDivOp)
      DISPATCH_ATIR_OP(FloorModOp)
      DISPATCH_ATIR_OP(GatherOp)
      DISPATCH_ATIR_OP(DotOp)
      DISPATCH_ATIR_OP(TransposeOp)
      DISPATCH_ATIR_OP(GreaterOp)
      DISPATCH_ATIR_OP(GreaterEqualOp)
      DISPATCH_ATIR_OP(IdentityOp)
      DISPATCH_ATIR_OP(LessOp)
      DISPATCH_ATIR_OP(LoadOp)
      DISPATCH_ATIR_OP(MatMulOp)
      DISPATCH_ATIR_OP(MaximumOp)
      DISPATCH_ATIR_OP(MinimumOp)
      DISPATCH_ATIR_OP(NotEqualOp)
      DISPATCH_ATIR_OP(AndOp)
      DISPATCH_ATIR_OP(PackOp)
      DISPATCH_ATIR_OP(ProdOp)
      DISPATCH_ATIR_OP(ReluOp)
      DISPATCH_ATIR_OP(AbsOp)
      DISPATCH_ATIR_OP(LogisticOp)
      DISPATCH_ATIR_OP(RealDivOp)
      DISPATCH_ATIR_OP(DivideOp)
      DISPATCH_ATIR_OP(RangeOp)
      DISPATCH_ATIR_OP(DynamicPartitionOp)
      DISPATCH_ATIR_OP(ParallelDynamicStitchOp)
      DISPATCH_ATIR_OP(ReshapeOp)
      DISPATCH_ATIR_OP(RsqrtOp)
      DISPATCH_ATIR_OP(SumOp)
      DISPATCH_ATIR_OP(StridedSliceOp)
      DISPATCH_ATIR_OP(CastOp)
      DISPATCH_ATIR_OP(ShapeOp)
      DISPATCH_ATIR_OP(SizeOp)
      DISPATCH_ATIR_OP(TileOp)
      DISPATCH_ATIR_OP(ZerosLikeOp)
      DISPATCH_ATIR_OP(BatchMatMulOp)
      DISPATCH_ATIR_OP(VariableOp)
      DISPATCH_ATIR_OP(WhereOp)
      DISPATCH_ATIR_OP(GatherNdOp)
      DISPATCH_ATIR_OP(SliceOp)
      DISPATCH_ATIR_OP(UniqueOp)
      DISPATCH_ATIR_OP(TopKOp)
      DISPATCH_ATIR_OP(UnsortedSegmentMinOp)
      DISPATCH_ATIR_OP(TensorScatterUpdateOp)
      DISPATCH_ATIR_OP(ScatterOp)
      DISPATCH_ATIR_OP(PadOp)
      DISPATCH_ATIR_OP(MergeOp)
      DISPATCH_ATIR_OP(BroadcastOp)
      DISPATCH_ATIR_OP(ResourceGatherOp)
      DISPATCH_ATIR_OP(SparseSegmentSumOp)
      DISPATCH_ATIR_OP(SparseSegmentMinOp)
      DISPATCH_ATIR_OP(SparseSegmentMeanOp)
      DISPATCH_ATIR_OP(SparseFillEmptyRowsOp)
      DISPATCH_ATIR_OP(StringToHashBucketFastOp)
      .Default([](Operation *op) {
        llvm::errs() << "Warning: No interpreter registered for op: " 
                     << op->getName() << "\n";
        return false;
      });
}

namespace atir {

// 运行原始子图
void runOriGraph(mlir::ModuleOp root, const std::string &funcName,
                 const IoTensorDef *inputs, IoTensorDef *outputs) {
  auto targetFuncOp = root.lookupSymbol<func::FuncOp>(funcName);
  if (!targetFuncOp) {
    llvm::errs() << "Error: Could not find target function: " << funcName << "\n";
    return;
  }

  LLVM_DEBUG({
    llvm::dbgs() << targetFuncOp.getName() << " inputs:\n";
    for (auto arg : targetFuncOp.getArguments()) {
      llvm::dbgs() << "  " << arg << " : " << arg.getType() << "\n";
    }
    llvm::dbgs() << "inputs size: " << targetFuncOp.getNumArguments() << "\n";
  });

  if (inputs->getInputCount() == 0) {
    llvm::errs() << "Error: inputs should be constructed by createIoDef first!\n";
    return;
  }

  Block &entryBlock = targetFuncOp.getBody().front();
  mlir::func::ReturnOp returnOp;

  for (Operation &op : entryBlock) {
    if (auto ret = dyn_cast<mlir::func::ReturnOp>(op)) {
      returnOp = ret;
      continue;
    }
    if (mlir::isa<atir::NoneOp>(op)) {
      continue;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "Executing Op: ";
      op.dump();
    });
    bool dispatched = dispatchOp(op);
    (void)dispatched;
  }

  if (!returnOp) {
    llvm::errs() << "Error: Could not find ReturnOp in function: " << funcName << "\n";
    return;
  }

  if (outputs->getOutputCount() == 0) {
    llvm::dbgs() << "Info: collecting results from block terminators to outputs.\n";
    int outputIndex = 0;
    for (auto result : returnOp.getOperands()) {
      std::string outputName = "output_" + std::to_string(outputIndex++);
      outputs->addOutput(outputName, result);
    }
    llvm::dbgs() << "Info: collected " << outputs->getOutputCount() << " output tensors.\n";
  }

  LLVM_DEBUG({
    if (outputs->getOutputCount()) {
      llvm::dbgs() << "\nFinal Outputs:\n";
      for (const auto &outputData : outputs->getOutputs()) {
        outputData.second.dump();
      }
    }
  });
}

} // namespace atir
