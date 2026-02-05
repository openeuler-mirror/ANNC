#include "Conversion/PimpToAffine/OpLowering.h"
#include "Conversion/Common/PimpLowering.h"
#include "Conversion/PimpToAffine/PimpTypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/IntegerSet.h"

namespace {
    int getBiasAxis(Value bias, ArrayRef<int64_t> shape) {
        auto biasShape = mlir::cast<MemRefType>(bias.getType()).getShape();
        return biasShape[0] == shape[1] ? 1 : 0;
    }

    struct Bias {
        Value bias;
        int axis;

        Bias(Value bias, ArrayRef<int64_t> shape) : bias(bias), axis(getBiasAxis(bias, shape)) {}
    };

    arith::SelectOp doRelu(PatternRewriter& rewriter, Location loc, FloatAttr reluLimitAttr, Value originValue) {
        auto reluLimit = rewriter.create<arith::ConstantOp>(loc, reluLimitAttr);
        auto underLimit = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, originValue, reluLimit);
        auto reluVal = rewriter.create<arith::SelectOp>(loc, underLimit, reluLimit, originValue);

        return reluVal;
    }

    // Value doBroadcastAdd(PatternRewriter& rewriter, Location loc, Value preSum, ValueRange pos, Bias bias) {
    //     //todo: get tiling of bias from higher level
    //     auto biasVal = rewriter.create<affine::AffineLoadOp>(loc, bias.bias, ValueRange{pos[bias.axis]});
    //     return rewriter.create<arith::AddFOp>(loc, biasVal, preSum);
    // }
}

namespace pimp
{
void populatePimpToAffineConversionPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns)
{
    patterns.add<NoneLoweringToAffine>(typeConverter, patterns.getContext());
    patterns.add<ConstantLoweringToAffine>(typeConverter, patterns.getContext());
    patterns.add<ReluLoweringToAffine>(typeConverter, patterns.getContext());
    patterns.add<LoadLoweringToAffine>(typeConverter, patterns.getContext());
    patterns.add<AddLoweringToAffine>(typeConverter, patterns.getContext());
    patterns.add<ConcatLoweringToAffine>(typeConverter, patterns.getContext());
    patterns.add<MatMulLoweringToAffine>(typeConverter, patterns.getContext());
    patterns.add<ReturnLoweringToAffine>(typeConverter, patterns.getContext());
    patterns.add<FuncReturnOpLowering>(typeConverter, patterns.getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
}

void NoneLoweringToAffine::Lowering(PatternRewriter& rewriter, NoneOpAdaptor adaptor, NoneOp op) const
{
    rewriter.eraseOp(op);
}

void ConstantLoweringToAffine::Lowering(PatternRewriter& rewriter, ConstantOpAdaptor adaptor, ConstantOp op) const
{
    Location loc = op.getLoc();
    auto elementType = op.getType().getElementType();
    auto tensorType = RankedTensorType::get(op.getType().getShape(), elementType);
    DenseElementsAttr data = op.getResult().getType().getCacheData();
    auto tensor = rewriter.create<arith::ConstantOp>(loc, tensorType, data);

    auto memrefType = MemRefType::get(op.getType().getShape(), elementType);
    auto memref = rewriter.create<bufferization::ToBufferOp>(loc, memrefType, tensor);
    rewriter.replaceOp(op, memref);
}

void ReluLoweringToAffine::Lowering(PatternRewriter &rewriter, ReluOpAdaptor adaptor, ReluOp op) const {
    Value input = adaptor.getInput();
    auto memRefTy = mlir::cast<MemRefType>(input.getType());
    auto shape = memRefTy.getShape();
    Location loc = op.getLoc();

    auto dim0For = rewriter.create<affine::AffineForOp>(loc, 0, shape[0]);
    {
        OpBuilder::InsertionGuard guardDim0(rewriter);
        rewriter.setInsertionPointToStart(dim0For.getBody());

        auto dim1For = rewriter.create<affine::AffineForOp>(loc, 0, shape[1]);
        {
            OpBuilder::InsertionGuard guardDim1(rewriter);
            rewriter.setInsertionPointToStart(dim1For.getBody());

            Value i = dim0For.getInductionVar();
            Value j = dim1For.getInductionVar();
            auto originVal = rewriter.create<affine::AffineLoadOp>(loc, input, ValueRange{i, j});
            Value reluVal = doRelu(rewriter, loc, op.getReluLimitAttr(), originVal);
            rewriter.create<affine::AffineStoreOp>(loc, reluVal, input, ValueRange{i, j});
        }
    }

    rewriter.replaceOp(op, input);
}

void LoadLoweringToAffine::Lowering(PatternRewriter &rewriter, LoadOpAdaptor adaptor, LoadOp op) const {
    auto axes = op.getAxes().getValue();
    auto subStart = op.getStart().getValue();
    auto subSize = op.getSize().getValue();
    auto shape = op.getInput().getType().getShape();
    SmallVector<int64_t> offsets(shape.size(), 0);
    SmallVector<int64_t> sizes(shape.begin(), shape.end());
    SmallVector<int64_t> strides(shape.size(), 1);
    for (int i = 0; i < axes.size(); ++i) {
        int axis = mlir::cast<IntegerAttr>(axes[i]).getInt();
        offsets[axis] = mlir::cast<IntegerAttr>(subStart[i]).getInt();
        sizes[axis] = mlir::cast<IntegerAttr>(subSize[i]).getInt();
    }

    Location loc = op.getLoc();
    auto src = adaptor.getInput();
    auto subView = rewriter.create<memref::SubViewOp>(loc, src, offsets, sizes, strides);
    rewriter.replaceOp(op, subView);
}

void AddLoweringToAffine::Lowering(PatternRewriter& rewriter, AddOpAdaptor adaptor, AddOp op) const
{
    Location loc = op.getLoc();
    auto shape = op.getOutput().getType().getShape();
    MemRefType memRefTy = MemRefType::get(shape, op.getType().getElementType());

    Value out = rewriter.create<memref::AllocOp>(loc, memRefTy);

    auto inputs = llvm::to_vector<>(adaptor.getInputs());

    SmallVector<Bias> broadcast = {};
    SmallVector<Value> tensors = {};
    for (auto input : inputs) {
        if (mlir::cast<MemRefType>(input.getType()).getShape().size() == 1) {
            broadcast.emplace_back(input, shape);
        } else {
            tensors.push_back(input);
        }
    }

    auto zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0));
    auto dim0For = rewriter.create<affine::AffineForOp>(loc, 0, shape[0]);
    {
        OpBuilder::InsertionGuard guardDim0(rewriter);
        rewriter.setInsertionPointToStart(dim0For.getBody());
        auto dim1For = rewriter.create<affine::AffineForOp>(loc, 0, shape[1]);
        {
            OpBuilder::InsertionGuard guardDim1(rewriter);
            rewriter.setInsertionPointToStart(dim1For.getBody());
            Value i = dim0For.getInductionVar();
            Value j = dim1For.getInductionVar();

            Value sum = zero;
            for (Value tensor : tensors) {
                Value inputVal = rewriter.create<affine::AffineLoadOp>(loc, tensor, ValueRange{i, j});
                sum = rewriter.create<arith::AddFOp>(loc, sum, inputVal);
            }

            for (Bias bias : broadcast) {
                // sum = doBroadcastAdd(rewriter, loc, sum, ValueRange{i, j}, bias);
            }

            auto offsetAttr = op.getScalarAttr();

            if (offsetAttr) {
                auto offset = rewriter.create<arith::ConstantOp>(loc, offsetAttr);
                sum = rewriter.create<arith::AddFOp>(loc, sum, offset);
            }

            if (op.getDoRelu())
            {
                sum = doRelu(rewriter, loc, op.getReluLimitAttr(), sum);
            }
            rewriter.create<affine::AffineStoreOp>(loc, sum, out, ValueRange{i, j});
        }
    }

    rewriter.replaceOp(op, out);
}

void ConcatLoweringToAffine::Lowering(PatternRewriter& rewriter, ConcatOpAdaptor adaptor, ConcatOp op) const
{
    auto inputs = llvm::to_vector(adaptor.getInputs());
    if (op.getIsView()) {
        rewriter.replaceOp(op, inputs[0]);
        return;
    }
    uint32_t axis = op.getAxis();
    SmallVector<ArrayRef<int64_t>> inputShapes;
    int64_t outAxisShape = 0;
    for (auto input: inputs) {
        auto inputShape = mlir::cast<MemRefType>(input.getType()).getShape();
        inputShapes.push_back(inputShape);
        outAxisShape += inputShape[axis];
    }
    SmallVector<int64_t> shapeBase(inputShapes[0].begin(), inputShapes[0].end());
    shapeBase[axis] = outAxisShape;
    ArrayRef<int64_t> shape(shapeBase);
    auto elementType = op.getResult().getType().getElementType();
    auto outMemRefTy = MemRefType::get(shape, elementType);
    auto loc = op.getLoc();
    MLIRContext *cxt = rewriter.getContext();

    Value out = rewriter.create<memref::AllocOp>(loc, outMemRefTy);

    int axisStart = 0;
    AffineExpr dAxis = rewriter.getAffineDimExpr(axis);
    for (int i = 0; i < inputs.size(); ++i) {
        Value input = inputs[i];
        auto inputShape = inputShapes[i];
        AffineExpr axisOffset = dAxis + rewriter.getAffineConstantExpr(axisStart);
        AffineMap axisOffsetMap = AffineMap::get(2, 0, axisOffset, cxt);

        auto dim0For = rewriter.create<affine::AffineForOp>(loc, 0, inputShape[0]);
        {
            OpBuilder::InsertionGuard guardDim0(rewriter);
            rewriter.setInsertionPointToStart(dim0For.getBody());
            auto dim1For = rewriter.create<affine::AffineForOp>(loc, 0, inputShape[1]);
            {
                OpBuilder::InsertionGuard guardDim1(rewriter);
                rewriter.setInsertionPointToStart(dim1For.getBody());

                Value m = dim0For.getInductionVar();
                Value n = dim1For.getInductionVar();
                Value val = rewriter.create<affine::AffineLoadOp>(loc, input, ValueRange{m, n});
                SmallVector<Value> outIndex = {m, n};
                outIndex[axis] = rewriter.create<affine::AffineApplyOp>(loc, axisOffsetMap, ValueRange{outIndex});

                if (op.getDoRelu()) {
                    val = doRelu(rewriter, loc, op.getReluLimitAttr(), val);
                }
                rewriter.create<affine::AffineStoreOp>(loc, val, out, ValueRange(outIndex));

            }
        }

        axisStart += inputShape[axis];
    }
    rewriter.replaceOp(op, out);

}

void MatMulLoweringToAffine::Lowering(PatternRewriter& rewriter, MatMulOpAdaptor adaptor, MatMulOp op) const
{
    MLIRContext *cxt = getContext();
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value c = adaptor.getC();
    Value bias = adaptor.getBias();

    auto lShape = mlir::cast<MemRefType>(lhs.getType()).getShape();
    auto rShape = mlir::cast<MemRefType>(rhs.getType()).getShape();
    auto cShape = mlir::cast<MemRefType>(c.getType()).getShape();
    int64_t m = lShape[0], k = lShape[1], n = rShape[1];
    SmallVector<int64_t> strides(2, 1);

    int64_t mOffset = 0, kOffset = 0, nOffset = 0;
    if (op.getMStart()) {
        mOffset = op.getMStart().value();
        m = op.getMSize().value();
    }
    if (op.getKStart()) {
        kOffset = op.getKStart().value();
        k = op.getKSize().value();
    }
    if (op.getNStart()) {
        nOffset = op.getNStart().value();
        n = op.getNSize().value();
    }
    SmallVector lOffsets{mOffset, kOffset};
    SmallVector rOffsets{kOffset, nOffset};
    SmallVector cOffsets{mOffset, nOffset};
    SmallVector lSize{m, k};
    SmallVector rSize{k, n};
    SmallVector cSize{m, n};

    int axis = -1;
    if (op.getWithBias()) {
        axis = getBiasAxis(bias, cShape);
        SmallVector biasOffsets{cOffsets[axis]};
        SmallVector biasSize{cSize[axis]};
        SmallVector biasStrides{strides[axis]};
        bias = rewriter.create<memref::SubViewOp>(loc, bias, biasOffsets, biasSize, biasStrides);
    }

    lhs = rewriter.create<memref::SubViewOp>(loc, lhs, lOffsets, lSize, strides);
    rhs = rewriter.create<memref::SubViewOp>(loc, rhs, rOffsets, rSize, strides);
    Value cView = rewriter.create<memref::SubViewOp>(loc, c, cOffsets, cSize, strides);

    Value biasVal;
    auto dim0For = rewriter.create<affine::AffineForOp>(loc, 0, m);
    {
        OpBuilder::InsertionGuard guardDim0(rewriter);
        rewriter.setInsertionPointToStart(dim0For.getBody());
        auto i = dim0For.getInductionVar();
        if (op.getWithBias() && axis == 0) {
            biasVal = rewriter.create<affine::AffineLoadOp>(loc, bias, ValueRange{i});
        }
        auto dim1For = rewriter.create<affine::AffineForOp>(loc, 0, n);
        {
            OpBuilder::InsertionGuard guardDim1(rewriter);
            rewriter.setInsertionPointToStart(dim1For.getBody());
            auto j = dim1For.getInductionVar();
            if (op.getWithBias() && axis == 1) {
                biasVal = rewriter.create<affine::AffineLoadOp>(loc, bias, ValueRange{j});
            }
            auto dim2For = rewriter.create<affine::AffineForOp>(loc, 0, k);
            {
                OpBuilder::InsertionGuard guardDim2(rewriter);
                rewriter.setInsertionPointToStart(dim2For.getBody());
                auto p = dim2For.getInductionVar();
                auto a = rewriter.create<affine::AffineLoadOp>(loc, lhs, ValueRange{i, p});
                auto b = rewriter.create<affine::AffineLoadOp>(loc, rhs, ValueRange{p, j});
                auto mul = rewriter.create<arith::MulFOp>(loc, a, b);
                auto pre = rewriter.create<affine::AffineLoadOp>(loc, cView, ValueRange{i, j});
                auto sum = rewriter.create<arith::AddFOp>(loc, pre, mul);
                rewriter.create<affine::AffineStoreOp>(loc, sum, cView, ValueRange{i, j});
            }

            if (op.getWithBias()) {
                auto old = rewriter.create<affine::AffineLoadOp>(loc, cView, ValueRange{i, j});
                auto biasAdd = rewriter.create<arith::AddFOp>(loc, old, biasVal);
                rewriter.create<affine::AffineStoreOp>(loc, biasAdd, cView, ValueRange{i, j});
            }
            if (op.getDoRelu()) {
                Value originVal = rewriter.create<affine::AffineLoadOp>(loc, cView, ValueRange{i, j});
                Value reluVal = doRelu(rewriter, loc, op.getReluLimitAttr(), originVal);
                rewriter.create<affine::AffineStoreOp>(loc, reluVal, cView, ValueRange{i, j});
            }
        }
    }

    rewriter.replaceOp(op, c);
}

void ReturnLoweringToAffine::Lowering(PatternRewriter &rewriter, ReturnOpAdaptor adaptor, ReturnOp op) const {
    auto ret = adaptor.getResults();
    auto funcReturn = rewriter.create<func::ReturnOp>(op.getLoc(), ret);
    rewriter.replaceOp(op, funcReturn);
}
}
