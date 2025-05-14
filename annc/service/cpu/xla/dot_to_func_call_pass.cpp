#include <cstdint>

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mhlo/IR/hlo_ops.h"
#include "transforms/passes.h"

namespace mlir {

#define GEN_PASS_DEF_DOTTOFUNCTIONCALLPASS
#include "transforms/passes.h.inc"

namespace {

class DotToFunctionCallPass
    : public impl::DotToFunctionCallPassBase<DotToFunctionCallPass> {

public:
  using DotToFunctionCallPassBase<DotToFunctionCallPass>::DotToFunctionCallPassBase;
  DotToFunctionCallPass() { 
    sgemv_nn_callee_initialized = false;
    matmul_idx = 0; 
  }
 
private:
  int matmul_idx;
  func::FuncOp sgemv_nn_callee;
  bool sgemv_nn_callee_initialized;
  void runOnOperation() override;

  enum MatmulType {
    // GEMM
    SGEMM_MLIRLIB_DEFAULT,          // 0
    SGEMM_MLIRLIB_LIBSHALOM_PACK,   // 1
    SGEMM_BLASKERNEL,               // 2

    // BatchMatmul
    BATCH_SGEMM_NN,                 // 3
    BATCH_SGEMM_NT,                 // 4
    BATCH_SGEMM_NT_FUSED_MUL_C,     // 5
    BATCH_SGEMM_NT_FUSED_MUL_C_CST, // 6
    // Currently not supported, so commenting this for now:
    // BATCH_SGEMM_TN,
    // BATCH_SGEMM_TT,

    BATCH_SGEMM_NN_BLASKERNEL,      // 7
    BATCH_SGEMM_NT_BLASKERNEL,      // 8

    BATCH_SGEMM_INVALID,            // 9
    SGEMM_INVALID                   // 10
  };

  struct MatmulMeta {
    std::string funcName;
    bool initialized;
    func::FuncOp callee;
  };

  std::map<int, MatmulMeta> BatchMatmulToMeta =
  {
    {BATCH_SGEMM_NN,                 MatmulMeta { "batch_matmul_nn",                 false, nullptr } },
    {BATCH_SGEMM_NT,                 MatmulMeta { "batch_matmul_nt",                 false, nullptr } },
    {BATCH_SGEMM_NT_FUSED_MUL_C,     MatmulMeta { "batch_matmul_nt_fused_mul_c",     false, nullptr } },
    {BATCH_SGEMM_NT_FUSED_MUL_C_CST, MatmulMeta { "batch_matmul_nt_fused_mul_c_cst", false, nullptr } },
    // Currently not supported, so commenting this for now:
    // {BATCH_SGEMM_TN, MatmulMeta { "batch_matmul_tn", false, nullptr } },
    // {BATCH_SGEMM_TT, MatmulMeta { "batch_matmul_tt", false, nullptr } },

    {BATCH_SGEMM_NN_BLASKERNEL,      MatmulMeta { "batch_matmul_nn_blaskernel",      false, nullptr } },
    {BATCH_SGEMM_NT_BLASKERNEL,      MatmulMeta { "batch_matmul_nt_blaskernel",      false, nullptr } },
  };

  std::map<int, MatmulMeta> MatmulToMeta =
  {
    {SGEMM_MLIRLIB_DEFAULT,        MatmulMeta { "sgemm_mlirlib",                false, nullptr } },
    {SGEMM_MLIRLIB_LIBSHALOM_PACK, MatmulMeta { "sgemm_mlirlib_libshalom_pack", false, nullptr } },
    {SGEMM_BLASKERNEL,             MatmulMeta { "sgemm_blaskernel",             false, nullptr } },
  };

  std::map<std::tuple<int, int , int> , int> MatmulSizesToMatmulType =
  {
    // DFFM
    { {6656,   8,    8}, SGEMM_MLIRLIB_LIBSHALOM_PACK},
    { {128, 1024,  416}, SGEMM_BLASKERNEL},
    { {128,  512, 1024}, SGEMM_BLASKERNEL},
    { {128,  256,  512}, SGEMM_BLASKERNEL}
  };

  std::map<std::tuple<int, int, int , int> , int> BatchMatmulSizesToBatchMatmulType =
  {
    // DFFM
    { {512, 26,   4,    4}, BATCH_SGEMM_NT_BLASKERNEL},
    { {512, 26,   4,   26}, BATCH_SGEMM_NN_BLASKERNEL}
  };

  bool isFusedMatmulType(int matmulType) {
    return matmulType == BATCH_SGEMM_NT_FUSED_MUL_C;
  }

  // These implementations are designed to be called from C/C++.
  // Because we are calling from MLIR (i.e., using memrefs instead
  // of plain pointers), we need to write a wrapper to convert
  // memrefs into C pointers.
  bool needsWrapper(int matmulType) {
    return matmulType == SGEMM_BLASKERNEL ||
           matmulType == BATCH_SGEMM_NN_BLASKERNEL ||
           matmulType == BATCH_SGEMM_NT_BLASKERNEL;
  }

  // Allow users override default kernel selector. This function
  // parses the env variable (passed by as "selector") and returns
  // the matmulType for the current matmul being processed.
  int getMatmulTypeFromSelector(std::string selector) {
    std::vector<int> result;
    std::stringstream ss(selector);
    std::string temp;

    while (std::getline(ss, temp, ','))
        result.push_back(std::stoi(temp));

    if (matmul_idx >= result.size()) {
      llvm::errs() << "FATAL: Kernel selector does not have enough values!\n";
      llvm::errs() << "Should have at least " << matmul_idx+1 << "but has " << result.size() << "\n";
    }

    int res = result[matmul_idx];
    llvm::errs() << "[Kernel Selector] Using MatmulType:" << res << " for matmul with index: " << matmul_idx << "\n";
    matmul_idx++;
    return res;
  }

  // Declare GEMM function if not already declared
  FailureOr<func::FuncOp> getOrInsertMatVecMulFunction(
    Operation *op, IRRewriter *rewriter, Builder builder, int matmulType,
    MemRefType matrixTypeMemref, MemRefType vectorTypeMemref) {

    Location loc = op->getLoc();
    func::FuncOp parentFuncOp = op->getParentOfType<func::FuncOp>();

    if (!sgemv_nn_callee_initialized) {
      rewriter->setInsertionPoint(parentFuncOp);
      SmallVector<Type> operandTypes(
          {matrixTypeMemref.getElementType(), vectorTypeMemref.getElementType(),
          matrixTypeMemref, vectorTypeMemref, vectorTypeMemref});

      FunctionType funcType = FunctionType::get(rewriter->getContext(),
                                                operandTypes, vectorTypeMemref);
      sgemv_nn_callee = rewriter->create<func::FuncOp>(loc, "sgemv", funcType);
      sgemv_nn_callee.setSymVisibilityAttr(builder.getStringAttr("private"));
      sgemv_nn_callee_initialized = true;
    }

    return sgemv_nn_callee;
  }

  // Declare GEMM function if not already declared
  FailureOr<func::FuncOp> getOrInsertMatmulFunction(Operation *op, IRRewriter *rewriter, Builder builder, int matmulType, MemRefType matrixTypeMemref) {
    Location loc = op->getLoc();
    func::FuncOp parentFuncOp = op->getParentOfType<func::FuncOp>();

    if (dyn_cast<mhlo::DotOp>(op)) {
      if (!MatmulToMeta[matmulType].initialized) {
        // AFAIK, mhlo::DotOp only represents non-transposed GEMM, so we do not check trA/trB
        rewriter->setInsertionPoint(parentFuncOp);

        SmallVector<Type> operandTypes({matrixTypeMemref.getElementType(), matrixTypeMemref.getElementType(), matrixTypeMemref, matrixTypeMemref, matrixTypeMemref});

        FunctionType funcType =
          FunctionType::get(rewriter->getContext(), operandTypes, matrixTypeMemref);
        MatmulToMeta[matmulType].callee = rewriter->create<func::FuncOp>(loc, MatmulToMeta[matmulType].funcName, funcType);
        MatmulToMeta[matmulType].callee.setSymVisibilityAttr(builder.getStringAttr("private"));

        if (needsWrapper(matmulType)) {
          // These implementations are written in C, so we need to call a C wrapper instead.
          // Generate the attribute to indicate this to MLIR.
          MatmulToMeta[matmulType].callee->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(), UnitAttr::get(rewriter->getContext()));
        }

        MatmulToMeta[matmulType].initialized = true;
      }

      return MatmulToMeta[matmulType].callee;
    }
    else if (dyn_cast<mhlo::DotGeneralOp>(op)) {
      assert(matmulType != BATCH_SGEMM_INVALID);

      if (!BatchMatmulToMeta[matmulType].initialized) {
        rewriter->setInsertionPoint(parentFuncOp);

        SmallVector<Type> operandTypes({matrixTypeMemref, matrixTypeMemref, matrixTypeMemref});
        if (isFusedMatmulType(matmulType))
          operandTypes.push_back(matrixTypeMemref);
        else if (matmulType == BATCH_SGEMM_NT_FUSED_MUL_C_CST)
          operandTypes.push_back(matrixTypeMemref.getElementType());

        FunctionType funcType = FunctionType::get(rewriter->getContext(), operandTypes, matrixTypeMemref);
        BatchMatmulToMeta[matmulType].callee = rewriter->create<func::FuncOp>(loc, BatchMatmulToMeta[matmulType].funcName, funcType);
        BatchMatmulToMeta[matmulType].callee.setSymVisibilityAttr(builder.getStringAttr("private"));

        if (needsWrapper(matmulType)) {
          BatchMatmulToMeta[matmulType].callee->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(), UnitAttr::get(rewriter->getContext()));
        }

        BatchMatmulToMeta[matmulType].initialized = true;
      }

      return BatchMatmulToMeta[matmulType].callee;
    }

    op->emitOpError("invalid mhlo op to get the matmul function for");
    return failure();
  }

  int getMamtulType(mhlo::DotOp op) {
    // Copied from getMatmulOperandType (TODO: Refactor)
    if (op->getNumOperands() != 2) {
      // VLOG(1) << "expected to find an op with exactly 2 operands";
      return SGEMM_INVALID;
    }

    RankedTensorType matATy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    RankedTensorType matBTy = dyn_cast<RankedTensorType>(op->getOperand(1).getType());

    if (!matATy || !matBTy) {
      // VLOG(1) << "expected to find an input operand of ranked tensor type";
      return SGEMM_INVALID;
    }

    if (matATy.getElementType() != matBTy.getElementType()) {
      // VLOG(1) << "expected to find the same element type on both input operands";
      return SGEMM_INVALID;
    }
    // end copy-pasta

    char * matmulSelector;
    if ((matmulSelector = getenv("XLA_MLIR_MATMUL_SELECTOR")) != NULL) {
      return getMatmulTypeFromSelector(matmulSelector);
    }

    int M = matATy.getShape()[0];
    int K = matATy.getShape()[1];
    int N = matBTy.getShape()[1];

    auto sizesMap = std::make_tuple(M, N, K);
    if (MatmulSizesToMatmulType.find(sizesMap) == MatmulSizesToMatmulType.end()) {
      // Our map does not have a value for this input size, report to the user and return the default implementation.
      llvm::errs() << "WARNING: getMamtulType: Input size not found: M=" << M << " N=" << N << " K=" << K << "\n";
      llvm::errs() << "         Please consider adding this size to the LUT. This will surely improve performance!\n";
      return SGEMM_BLASKERNEL;
    }
    return MatmulSizesToMatmulType[sizesMap];
  }

  int getBatchMamtulType(mhlo::DotGeneralOp op, Operation* toFuse, bool foldTensorConstant) {
    // From DotGeneralBatchMatMulOpConversion in:
    // xla/xla/mlir_hlo/mhlo/transforms/legalize_to_linalg/legalize_to_linalg.cc
    mhlo::DotDimensionNumbersAttr dimNumbers = op.getDotDimensionNumbers();
    auto lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
    auto rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
    auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
    auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();

    // Copied from getMatmulOperandType (TODO: Refactor)
    if (op->getNumOperands() != 2) {
      // VLOG(1) << "expected to find an op with exactly 2 operands";
      return BATCH_SGEMM_INVALID;
    }

    RankedTensorType matATy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    RankedTensorType matBTy = dyn_cast<RankedTensorType>(op->getOperand(1).getType());

    if (!matATy || !matBTy) {
      // VLOG(1) << "expected to find an input operand of ranked tensor type";
      return BATCH_SGEMM_INVALID;
    }

    if (matATy.getElementType() != matBTy.getElementType()) {
      // VLOG(1) << "expected to find the same element type on both input operands";
      return BATCH_SGEMM_INVALID;
    }
    // end copy-pasta

    char * matmulSelector;
    if ((matmulSelector = getenv("XLA_MLIR_MATMUL_SELECTOR")) != NULL) {
      return getMatmulTypeFromSelector(matmulSelector);
    }

    int B = matATy.getShape()[0];
    int M = matATy.getShape()[1];
    int K = matATy.getShape()[2];
    int N = matBTy.getShape()[2];

    // Detect the type of batch matmul.
    // We currently only support canonical batch matmul and trB.
    if (lhsBatchingDims.size() != 1 || lhsBatchingDims[0] != 0) {
      return BATCH_SGEMM_INVALID;
    }
    if (rhsBatchingDims.size() != 1 || rhsBatchingDims[0] != 0) {
      return BATCH_SGEMM_INVALID;
    }
    if (lhsContractingDims.size() != 1 || lhsContractingDims[0] != 2) {
      return BATCH_SGEMM_INVALID;
    }
    if (rhsContractingDims.size() != 1 || rhsContractingDims[0] != 1) {
      if (toFuse && dyn_cast<mhlo::MulOp>(toFuse)) {
        // If we fuse, we must call a custom function specifically generated for this.
        if (foldTensorConstant)
          return BATCH_SGEMM_NT_FUSED_MUL_C_CST;
        return BATCH_SGEMM_NT_FUSED_MUL_C;
      }
    }

    auto sizesMap = std::make_tuple(B, M, N, K);
    if (BatchMatmulSizesToBatchMatmulType.find(sizesMap) == BatchMatmulSizesToBatchMatmulType.end()) {
      // Our map does not have a value for this input size, report to the user and return the default implementation.
      llvm::errs() << "WARNING: getBatchMamtulType: Input size not found: B=" << B << " M=" << M << " N=" << N << " K=" << K << "\n";
      llvm::errs() << "         Please consider adding this size to the LUT. This will surely improve performance!\n";
      return BATCH_SGEMM_NN;
    }
    return BatchMatmulSizesToBatchMatmulType[sizesMap];
  }

  FailureOr<std::vector<RankedTensorType>> getOperandsType(Operation *op) {
    if (op->getNumOperands() != 2)
      return failure();

    Type operand0Type = op->getOperand(0).getType();
    Type operand1Type = op->getOperand(1).getType();
    RankedTensorType operand0 = dyn_cast<RankedTensorType>(operand0Type);
    RankedTensorType operand1 = dyn_cast<RankedTensorType>(operand1Type);
    if (!operand0 || !operand1)
      return failure();

    if (operand0.getElementType() != operand1.getElementType())
      return failure();

    std::vector<RankedTensorType> operandTypes{operand0, operand1};
    return operandTypes;
  }

  Value castRankedTensorToUnrankedMemref(Location loc, IRRewriter *rewriter, Value rankedTensor, MemRefType matrixTypeMemref, RankedTensorType matrixTypeTensor) {
    // readOnly indicates that the to_memref op will only be read and never writen to. We can be sure of this
    // since nobody else will use this value. Not setting this to true makes the bufferization much less
    // efficient since it inserts unnecesary memory allocations and copies.
    bool readOnly = true;
    Value dynTsr = rewriter->create<tensor::CastOp>(loc, matrixTypeTensor, rankedTensor);
    return rewriter->create<bufferization::ToMemrefOp>(loc, matrixTypeMemref, dynTsr, readOnly);
  }

  LogicalResult castMatVecMulInputs(Operation *op, IRRewriter *rewriter,
                                    Value &matA, Value &vec,
                                    MemRefType matrixTypeMemref,
                                    RankedTensorType matrixTypeTensor,
                                    MemRefType vectorTypeMemref,
                                    RankedTensorType vectorTypeTensor) {
    Location loc = op->getLoc();
    Value matAStaticTsr = op->getOperand(0);
    Value vecStaticTsr = op->getOperand(1);
    if (!dyn_cast<RankedTensorType>(matAStaticTsr.getType()) ||
        !dyn_cast<RankedTensorType>(vecStaticTsr.getType())) {
      op->emitOpError(
      "expected to find an input operand of ranked tensor type");
      return failure();
    }

    matA = castRankedTensorToUnrankedMemref(loc, rewriter, matAStaticTsr,
                matrixTypeMemref, matrixTypeTensor);
    vec = castRankedTensorToUnrankedMemref(loc, rewriter, vecStaticTsr,
              vectorTypeMemref, vectorTypeTensor);
    return success();
  }


  LogicalResult castMatmulInputs(Operation *op, IRRewriter *rewriter, Value &matA, Value &matB, MemRefType matrixTypeMemref, RankedTensorType matrixTypeTensor) {
    Location loc = op->getLoc();

    if (op->getNumOperands() != 2) {
      op->emitOpError("expected to find an op with exactly 2 operands");
      return failure();
    }

    Value matAStaticTsr = op->getOperand(0);
    Value matBStaticTsr = op->getOperand(1);

    if (!dyn_cast<RankedTensorType>(matAStaticTsr.getType()) ||
        !dyn_cast<RankedTensorType>(matBStaticTsr.getType())) {
      op->emitOpError("expected to find an input operand of ranked tensor type");
      return failure();
    }

    matA = castRankedTensorToUnrankedMemref(loc, rewriter, matAStaticTsr, matrixTypeMemref, matrixTypeTensor);
    matB = castRankedTensorToUnrankedMemref(loc, rewriter, matBStaticTsr, matrixTypeMemref, matrixTypeTensor);

    return success();
  }
};

FailureOr<SmallVector<Value>> getOutputDynamicShape(Operation *op, IRRewriter *rewriter, Builder builder) {
  Type opResultType = op->getResult(0).getType();
  if (!dyn_cast<RankedTensorType>(opResultType)) {
    op->emitOpError("expected to find a result of RankedTensorType");
    return failure();
  }

  Location loc = op->getLoc();
  SmallVector<Value> dynamicSizes;
  for (auto dim : dyn_cast<RankedTensorType>(opResultType).getShape())
    dynamicSizes.push_back(rewriter->create<arith::ConstantOp>(loc, builder.getIndexAttr(dim)));

  return dynamicSizes;
}

} // namespace

void DotToFunctionCallPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  Builder builder(context);

  // A list of ops that must be erased after the walk
  // has taken place.
  llvm::SmallVector<Operation*> opsToErase;
  bool enableFusion = false;

  getOperation()->walk([&](mhlo::DotOp op) {
    // 1. Get the element type of the input operands and create the matrix types
    // If the input op is not supported by this pass, return silently.
    auto maybeOperandTypes = getOperandsType(op);
    std::vector<RankedTensorType> operandTypes = *maybeOperandTypes;

    Type elemTy = operandTypes[0].getElementType();
    Location loc = op->getLoc();
    auto matrixTypeMemref =
        MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, elemTy);
    RankedTensorType matrixTypeTensor = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, elemTy);

    auto vectorTypeMemref = MemRefType::get({ShapedType::kDynamic}, elemTy);
    RankedTensorType vectorTypeTensor =
        RankedTensorType::get({ShapedType::kDynamic}, elemTy);

    // Get the best matmul implementation for the input size.
    int matmulType = getMamtulType(op);
    if (matmulType == SGEMM_INVALID)
      return;

    // 2. Get the GEMM callee
    if ((operandTypes[0].getRank() != 2))
      return;

    int64_t operand1Rank = operandTypes[1].getRank();
    if ((operand1Rank != 2) && (operand1Rank != 1))
      return;

    bool isGemm = (operand1Rank == 2);
    FailureOr<func::FuncOp> maybeCallee =
        isGemm
            ? this->getOrInsertMatmulFunction(op.getOperation(), &rewriter,
                                              builder, matmulType, matrixTypeMemref)
            : this->getOrInsertMatVecMulFunction(op.getOperation(), &rewriter,
                                                 builder, -1, matrixTypeMemref,
                                                 vectorTypeMemref);
    if (failed(maybeCallee)) {
      emitError(op.getLoc(), "unable to find the right function call to replace the op");
      signalPassFailure();
    }
    func::FuncOp sgemCallee = *maybeCallee;

    rewriter.setInsertionPoint(op.getOperation());

    // 3. Create the alpha and beta constants
    Value alpha, beta;
    if (elemTy.isF32()) {
      alpha = rewriter.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(1.0f));
      beta = rewriter.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(0.0f));
    }
    else if (elemTy.isF64()) {
      alpha = rewriter.create<arith::ConstantOp>(loc, builder.getF64FloatAttr(1.0f));
      beta = rewriter.create<arith::ConstantOp>(loc, builder.getF64FloatAttr(0.0f));
    }
    else {
      emitError(op.getLoc(), "expected to find a matmul with either F32/F64 element type");
      signalPassFailure();
    }

    // 4. Cast A and B (from static tensors to dynamic shape memrefs)
    Value matA;
    Value matBOrVector;
    LogicalResult castOperands =
        isGemm ? castMatmulInputs(op, &rewriter, matA, matBOrVector,
                                  matrixTypeMemref, matrixTypeTensor)
               : castMatVecMulInputs(op, &rewriter, matA, matBOrVector,
                                     matrixTypeMemref, matrixTypeTensor,
                                     vectorTypeMemref, vectorTypeTensor);
    if (failed(castOperands))
      signalPassFailure();

    // 5. Create the buffer for C matrix (the shape must be the same as the output from the mhlo op)
    FailureOr<SmallVector<Value>> maybeDynSizes = getOutputDynamicShape(op, &rewriter, builder);
    if (failed(maybeDynSizes))
      signalPassFailure();
    SmallVector<Value> dynamicSizes = *maybeDynSizes;
    auto resultTypeMemref = isGemm ? matrixTypeMemref : vectorTypeMemref;
    Value matC = rewriter.create<memref::AllocOp>(loc, resultTypeMemref, dynamicSizes); // , builder.getI32IntegerAttr(64));

    // 6. Call the GEMM function
    ValueRange funcOperands({alpha, beta, matA, matBOrVector, matC});
    Operation *funcCall = rewriter.create<func::CallOp>(loc, sgemCallee, funcOperands);
    assert(funcCall->getNumResults() == 1);
    Value funcOutput = funcCall->getResult(0);

    // 7. Replace the dot operation and its result with the result from the library
    // call, which needs to be casted (from dynamic memref to static tensor)
    if (op->getNumResults() != 1) {
      emitError(op.getLoc(), "expected to find an op with exactly one result");
      signalPassFailure();
    }

    auto resultTypeTensor = isGemm ? matrixTypeTensor : vectorTypeTensor;
    Value dotResultMemref = rewriter.create<bufferization::ToTensorOp>(loc, resultTypeTensor, funcOutput, /*restrict*/ true);
    Value dotResult = rewriter.create<tensor::CastOp>(loc, op->getResult(0).getType(), dotResultMemref);

    op->getResult(0).replaceAllUsesWith(dotResult);
    op->erase();
  });

  getOperation()->walk([&](mhlo::DotGeneralOp op) {
    Location loc = op->getLoc();

    // Check if we can easily fuse this matmul with something before/after it.
    // The pattern matching is currently limited to a single MulOp after matmul.
    mhlo::MulOp mul;
    bool foldTensorConstant = false;
    Value operandToFuse;
    if (enableFusion && op->hasOneUse()) {
      Operation *matmulConsumer = *op->getResult(0).getUsers().begin();
      if (mul = dyn_cast<mhlo::MulOp>(matmulConsumer)) {
        mul->getResult(0).replaceAllUsesWith(op);
        Value operand = mul->getOperand(1);

        if (auto cst = dyn_cast<mhlo::ConstantOp>(operand.getDefiningOp())) {
          // If its coming from a tensor with a constant value, we can
          // simply use a constant instead of the whole tensor. This can
          // be really beneficial for performance as it will reduce the
          // memory loads.
          if (DenseElementsAttr attr = dyn_cast<DenseElementsAttr>(cst.getValue().cast<DenseElementsAttr>())) {
            if (attr.isSplat()) {
              foldTensorConstant = true;
              rewriter.setInsertionPoint(op.getOperation());
              // TODO: Support f64 as well.
              float cstVal = attr.getSplatValue<APFloat>().convertToFloat();
              operandToFuse = rewriter.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(cstVal));
            }
          }
        }

        if (!foldTensorConstant) {
          operandToFuse = operand;
        }
        opsToErase.push_back(mul);
      }
    }

    int batchMatmulType = getBatchMamtulType(op, mul, foldTensorConstant);

    // Not a batch matmul we support, skip it
    if (batchMatmulType == BATCH_SGEMM_INVALID)
      return;

    auto maybeOperandTypes = getOperandsType(op);
    if (failed(maybeOperandTypes))
      signalPassFailure();

    std::vector<RankedTensorType> operandTypes = *maybeOperandTypes;
    if ((operandTypes[0].getRank() != 3) || (operandTypes[1].getRank() != 3))
      return;

    Type elemTy = operandTypes[0].getElementType();
    auto matrixTypeMemref =
        MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic}, elemTy);
    RankedTensorType matrixTypeTensor = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic}, elemTy);

    // 1. Get the batch matmul callee
    FailureOr<func::FuncOp> maybeCallee = this->getOrInsertMatmulFunction(op, &rewriter, builder, batchMatmulType, matrixTypeMemref);
    if (failed(maybeCallee)) {
      emitError(op.getLoc(), "unable to find the right function call to replace the op");
      signalPassFailure();
    }

    func::FuncOp callee = *maybeCallee;

    rewriter.setInsertionPoint(op.getOperation());

    // 2. Cast A and B (from static tensors to dynamic shape memrefs)
    Value matA;
    Value matB;
    if (failed(castMatmulInputs(op, &rewriter, matA, matB, matrixTypeMemref, matrixTypeTensor)))
      signalPassFailure();

    // Cast ops to fuse, if any
    Value operandToFuseMemref;
    if (isFusedMatmulType(batchMatmulType)) {
      assert(dyn_cast<RankedTensorType>(operandToFuse.getType()) != nullptr);
      operandToFuseMemref = castRankedTensorToUnrankedMemref(loc, &rewriter, operandToFuse, matrixTypeMemref, matrixTypeTensor);
    }

    // 3. Create the buffer for C matrix (the shape must be the same as the output from the mhlo op)
    FailureOr<SmallVector<Value>> maybeDynSizes = getOutputDynamicShape(op, &rewriter, builder);
    if (failed(maybeDynSizes))
      signalPassFailure();
    SmallVector<Value> dynamicSizes = *maybeDynSizes;
    Value matC = rewriter.create<memref::AllocOp>(loc, matrixTypeMemref, dynamicSizes); // , builder.getI32IntegerAttr(64));

    // 4. Call the GEMM function
    std::vector<mlir::Value> funcOperandsVec({matA, matB, matC});
    if (isFusedMatmulType(batchMatmulType))
      funcOperandsVec.push_back(operandToFuseMemref);
    else if (batchMatmulType == BATCH_SGEMM_NT_FUSED_MUL_C_CST)
      funcOperandsVec.push_back(operandToFuse);

    ValueRange funcOperands(funcOperandsVec);
    Operation *funcCall = rewriter.create<func::CallOp>(loc, callee, funcOperands);
    assert(funcCall->getNumResults() == 1);
    Value funcOutput = funcCall->getResult(0);

    // 5. Replace the dot operation and its result with the result from the library
    // call, which needs to be casted (from dynamic memref to static tensor)
    if (op->getNumResults() != 1) {
      emitError(op.getLoc(), "expected to find an op with exactly one result");
      signalPassFailure();
    }

    Value dotResultMemref = rewriter.create<bufferization::ToTensorOp>(loc, matrixTypeTensor, funcOutput, /*restrict*/ true);
    Value dotResult = rewriter.create<tensor::CastOp>(loc, op->getResult(0).getType(), dotResultMemref);

    op->getResult(0).replaceAllUsesWith(dotResult);
    op->erase();
  });

  for (auto op : opsToErase)
    op->erase();
}

std::unique_ptr<Pass> hlo::createDotToFunctionCallPass() {
  return std::make_unique<DotToFunctionCallPass>();
}

} // namespace mlir
