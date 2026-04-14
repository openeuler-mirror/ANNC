#include "Conversion/AtirToLinalg/OpLowering.h"
#include "Conversion/Common/AtirLowering.h"
#include "Conversion/AtirToLinalg/AtirTypeConverter.h"
#include "Kernel/KernelSymbolResolver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/Debug.h"

namespace {
    std::vector<AffineMap> getTransposeAttr(MLIRContext *ctx, ArrayRef<bool> transpose)
    {
        std::vector<AffineMap> res;
        if (transpose[0] || transpose[1] || transpose[2])
        {
            auto d0 = mlir::getAffineDimExpr(0, ctx);
            auto d1 = mlir::getAffineDimExpr(1, ctx);
            auto d2 = mlir::getAffineDimExpr(2, ctx);
            AffineMap left, right, output;
            if (transpose[0])
            {
                left = AffineMap::get(3, 0, {d2, d0}, ctx);
            } else {
                right = AffineMap::get(3, 0, {d0, d2}, ctx);
            }
            if (transpose[1])
            {
                right = AffineMap::get(3, 0, {d1, d2}, ctx);
            } else {
                right = AffineMap::get(3, 0, {d2, d1}, ctx);
            }
            if (transpose[2])
            {
                output = AffineMap::get(3, 0, {d1, d0}, ctx);
            } else {
                output = AffineMap::get(3, 0, {d0, d1}, ctx);
            }
            res.push_back(left);
            res.push_back(right);
            res.push_back(output);
        }

        return res;
    }

    std::string getLoweringBackend(ModuleOp module) {
        if (auto backend = module->getAttrOfType<StringAttr>("annc.backend")) {
            return backend.str();
        }
        return "aarch64";
    }

    std::optional<std::string> resolveKernelCallee(atir::CustomizeOp op, ModuleOp module) {
        annc::kernels::KernelSymbolResolverRequest request;
        request.op_type = op.getOpType().str();
        request.backend = getLoweringBackend(module);

        return annc::kernels::ResolveKernelSymbol(request);
    }
}
namespace atir
{
void populateAtirToLinalgConversionPatterns(TypeConverter& inputTypeConverter, TypeConverter& atirTypeConverter, RewritePatternSet& patterns)
{
    // patterns.add<NoneLoweringToLinalg>(typeConverter, patterns.getContext());
    // patterns.add<ConstantLoweringToLinalg>(typeConverter, patterns.getContext());
    // patterns.add<ReluLoweringToLinalg>(typeConverter, patterns.getContext());
    // patterns.add<LoadLoweringToLinalg>(typeConverter, patterns.getContext());
    // patterns.add<AddLoweringToLinalg>(typeConverter, patterns.getContext());
    // patterns.add<ConcatLoweringToLinalg>(typeConverter, patterns.getContext());
    patterns.add<MatMulLoweringToLinalg>(atirTypeConverter, patterns.getContext());
    // patterns.add<ReturnLoweringToLinalg>(inputTypeConverter, patterns.getContext());
    patterns.add<CustomizeLoweringToLinalg>(inputTypeConverter, patterns.getContext());
    patterns.add<FuncReturnOpLowering>(inputTypeConverter, patterns.getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, inputTypeConverter);
}

void NoneLoweringToLinalg::Lowering(PatternRewriter& rewriter, NoneOpAdaptor adaptor, NoneOp op) const
{
    rewriter.eraseOp(op);
}

void ConstantLoweringToLinalg::Lowering(PatternRewriter& rewriter, ConstantOpAdaptor adaptor, ConstantOp op) const
{

}

void ReluLoweringToLinalg::Lowering(PatternRewriter &rewriter, ReluOpAdaptor adaptor, ReluOp op) const {

}

void LoadLoweringToLinalg::Lowering(PatternRewriter &rewriter, LoadOpAdaptor adaptor, LoadOp op) const {

}

void AddLoweringToLinalg::Lowering(PatternRewriter& rewriter, AddOpAdaptor adaptor, AddOp op) const
{

}

void ConcatLoweringToLinalg::Lowering(PatternRewriter& rewriter, ConcatOpAdaptor adaptor, ConcatOp op) const
{

}

void MatMulLoweringToLinalg::Lowering(PatternRewriter& rewriter, MatMulOpAdaptor adaptor, MatMulOp op) const
{
    MLIRContext *ctx = getContext();
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto lhs_type = mlir::cast<RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<RankedTensorType>(rhs.getType());
    auto lhs_rank = lhs_type.getRank();
    auto rhs_rank = rhs_type.getRank();
    Value c = adaptor.getC();
    auto attrs = op->getAttrs();
    if (lhs_rank == 2 && rhs_rank == 2)
    {
        auto linalgMatmul = rewriter.create<linalg::MatmulOp>(loc, c.getType(),ValueRange{lhs, rhs}, c, attrs);
        std::vector<AffineMap> transposeMap = getTransposeAttr(ctx,
            {op.getLeftTranspose(), op.getRightTranspose(), op.getOutputTranspose()});
        if (!transposeMap.empty())
        {
            linalgMatmul->setAttr("indexing_maps", rewriter.getAffineMapArrayAttr(transposeMap));
        }
        Value matmul = linalgMatmul.getResult(0);
        rewriter.replaceOp(op, matmul);
    }
}

void CustomizeLoweringToLinalg::Lowering(mlir::PatternRewriter &rewriter, atir::CustomizeOpAdaptor adaptor,
                                         atir::CustomizeOp op) const {

  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto calleeName = resolveKernelCallee(op, module);
  if (!calleeName.has_value()) {
    op.emitError() << "failed to resolve kernel symbol for CustomizeOp '"
                   << op.getOpType() << "'";
    return;
  }

  SmallVector<mlir::Value> newOperands;
  SmallVector<mlir::Type> inputMemRefTypes;
  for (mlir::Value input : adaptor.getOperands()) {
    newOperands.push_back(input);
    inputMemRefTypes.push_back(input.getType());
  }

  SmallVector<mlir::Type> resultMemrefTypes;
  for (mlir::Type resultType : op->getResultTypes()) {
    if (auto tensorType = llvm::dyn_cast<TensorType>(resultType)) {
      auto outMemrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      resultMemrefTypes.push_back(outMemrefType);
    }else {
      resultMemrefTypes.push_back(resultType);
    }
  }

  auto funcType = rewriter.getFunctionType(inputMemRefTypes,resultMemrefTypes);

  PatternRewriter::InsertionGuard guard(rewriter);

  auto funcOp = module.lookupSymbol<func::FuncOp>(*calleeName);
  if (!funcOp) {
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<func::FuncOp>(op.getLoc(), *calleeName, funcType);
      funcOp.setPrivate();
      auto emitCAttr = UnitAttr::get(rewriter.getContext());
      funcOp->setAttr("llvm.emit_c_interface", emitCAttr);
  } else if (funcOp.getFunctionType() != funcType) {
      op.emitError() << "resolved kernel callee '" << *calleeName
                     << "' already exists with mismatched function type";
      return;
  }

  rewriter.setInsertionPoint(op);

  auto callOp = rewriter.create<func::CallOp>(
      op.getLoc(),
      *calleeName,
      resultMemrefTypes,
      newOperands);

  rewriter.replaceOp(op, callOp);
}

void ReturnLoweringToLinalg::Lowering(PatternRewriter &rewriter, ReturnOpAdaptor adaptor, ReturnOp op) const {
    auto ret = adaptor.getResults();
    auto funcReturn = rewriter.create<func::ReturnOp>(op.getLoc(), ret);
    rewriter.replaceOp(op, funcReturn);
}
}
