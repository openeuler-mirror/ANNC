#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Pimp/PimpOps.h"
#include "Dialect/Pimp/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "iostream"

using namespace llvm;
using namespace mlir;

namespace pimp {

    struct BlockInfo {
        std::vector<int64_t> axes;
        std::vector<std::vector<int64_t>> starts;
        std::vector<std::vector<int64_t>> sizes;
    };

    BlockInfo parseMNTilingAttr(TilingAttr tilingl, TilingAttr tilingr) {
        BlockInfo info;
        std::vector<std::vector<int64_t>> starts;
        std::vector<std::vector<int64_t>> sizes;
        info.axes = tilingl.getTilingAxes();
        starts.push_back(tilingl.getTilingStart()[0]);
        starts.push_back(tilingr.getTilingStart()[1]);
        info.starts = starts;
        sizes.push_back(tilingl.getTilingSize()[0]);
        sizes.push_back(tilingr.getTilingSize()[1]);
        info.sizes = sizes;
        return info;
    }

    // K
    int getKBlockSize(pimp::TensorType tensorType, int kidx) {
        if (auto tiling = tensorType.getOnchipParallel()) {
            // K1
            return tiling.getTilingSize()[kidx].size();
        }
        // 1
        return 1;
    }

    // K
    int64_t getKStart(pimp::TensorType tensorType, int kidx, int blockIndex) {
        if (auto tiling = tensorType.getOnchipParallel()) {
            return tiling.getTilingStart()[kidx][blockIndex];
        }
        return 0;
    }

    // K
    int64_t getKSize(pimp::TensorType tensorType, int kidx, int blockIndex) {
        if (auto tiling = tensorType.getOnchipParallel()) {
            return tiling.getTilingSize()[kidx][blockIndex];
        }
        return tensorType.getShape()[kidx]; // 
    }

    MatMulOp createBlockMatMulWithAttrs(OpBuilder& builder, Location loc, Value lhs, Value rhs, Value C, Value bias, bool withBias,
                                        bool right_transpose, bool left_transpose, bool output_transpose, bool do_relu, float relu_limit,
                                        int64_t MStart, int64_t NStart, int64_t KStart,
                                        int64_t MSize, int64_t NSize, int64_t KSize) {
        std::cout << "this is createBlockMatMulWithAttrs1" << std::endl;
        mlir::Attribute encoding;
        auto outputType = pimp::TensorType::get(ArrayRef<int64_t>{MSize, NSize}, builder.getF32Type(),
                              builder.getStringAttr("matmul_out"), encoding);

        std::vector<NamedAttribute> attrs;

        attrs.push_back(NamedAttribute(builder.getStringAttr("withBias"),
                                       builder.getBoolAttr(withBias)));

        attrs.push_back(NamedAttribute(builder.getStringAttr("right_transpose"),
                                       builder.getBoolAttr(right_transpose)));

        attrs.push_back(NamedAttribute(builder.getStringAttr("left_transpose"),
                                       builder.getBoolAttr(left_transpose)));

        attrs.push_back(NamedAttribute(builder.getStringAttr("output_transpose"),
                                       builder.getBoolAttr(output_transpose)));

        attrs.push_back(NamedAttribute(builder.getStringAttr("do_relu"),
                                       builder.getBoolAttr(do_relu)));

        attrs.push_back(NamedAttribute(builder.getStringAttr("relu_limit"),
                                       builder.getF32FloatAttr(relu_limit)));

        attrs.push_back(NamedAttribute(builder.getStringAttr("m_start"),
                                       builder.getI32IntegerAttr(MStart)));

        attrs.push_back(NamedAttribute(builder.getStringAttr("n_start"),
                                       builder.getI32IntegerAttr(NStart)));

        attrs.push_back(NamedAttribute(builder.getStringAttr("k_start"),
                                       builder.getI32IntegerAttr(KStart)));

        attrs.push_back(NamedAttribute(builder.getStringAttr("m_size"),
                                       builder.getI32IntegerAttr(MSize)));

        attrs.push_back(NamedAttribute(builder.getStringAttr("n_size"),
                                       builder.getI32IntegerAttr(NSize)));

        attrs.push_back(NamedAttribute(builder.getStringAttr("k_size"),
                                       builder.getI32IntegerAttr(KSize)));

        std::vector<Value> ins;
        ins.push_back(lhs);
        ins.push_back(rhs);
        ins.push_back(C);
        if (bias != nullptr) {
            ins.push_back(bias);
        }
        std::vector<Type> outs;
        outs.push_back(outputType);
        auto matmulOp = builder.create<pimp::MatMulOp>(loc,outs,ins,attrs);
        return matmulOp;

    }

    // MultipleTensorAdd
    Value createMultipleTensorAdd(OpBuilder& builder, Location loc, ArrayRef<Value> inputs, bool do_relu, float relu_limit) {
        if (inputs.size() == 1) {
            return inputs[0]; // 
        }
        // 
        auto outputType = inputs[0].getType();
        // 
        auto addOp = builder.create<pimp::AddOp>(loc,outputType,inputs);
        addOp.setDoRelu(do_relu);
        addOp.setReluLimitAttr(builder.getF32FloatAttr(relu_limit));
        return addOp.getResult();
    }

    Value createLoad(OpBuilder& builder, Location loc, Value input, ArrayAttr axesAtt, ArrayAttr startAtt, ArrayAttr sizeAtt , bool copy) {
        //todo vectorop
        std::vector<int64_t> axes;
        int64_t max = 0;
        for (auto& attr :axesAtt) {
            int64_t ax = llvm::dyn_cast<IntegerAttr>(attr).getInt();
            max = std::max(max,ax);
            axes.push_back(ax);
        }
        std::vector<int64_t> starts;
        for (auto& attr :startAtt) {
            starts.push_back(llvm::dyn_cast<IntegerAttr>(attr).getInt());
        }
        std::vector<int64_t> sizes;
        for (auto& attr :sizeAtt) {
            sizes.push_back(llvm::dyn_cast<IntegerAttr>(attr).getInt());
        }
        std::vector<int64_t > shape;
        for (int i = 0; i <= max; i++) {
            auto it = std::find(axes.begin(),axes.end(),i);
            if (it != axes.end()) {
                size_t idx = std::distance(axes.begin(), it);
                shape.push_back(sizes[idx]);
            } else {
                shape.push_back(1);
            }
        }
        auto inputType = llvm::dyn_cast<TensorType>(input.getType());
        auto outputType = TensorType::get(shape, inputType.getElementType(), inputType.getName(), inputType.getEncoding(),
                        inputType.getStride(), inputType.getLayout(), inputType.getMemType(),
                        inputType.getAddress(), inputType.getDeviceParallel(), inputType.getOnchipParallel(), inputType.getCacheData());

        auto loadOp = builder.create<pimp::LoadOp>(loc,outputType, input, axesAtt,startAtt,sizeAtt,BoolAttr::get(builder.getContext(),copy));
        return loadOp.getResult();
    }

    Value create2DConcat(OpBuilder& builder, Location loc,
                         ArrayRef<SmallVector<Value>> results2D, BlockInfo blockInfo) {
        int blockMCount = blockInfo.sizes[0].size();
        int blockNCount = blockInfo.sizes[1].size();

        // n axes concat
        SmallVector<Value> rowResults;
        rowResults.reserve(blockMCount);

        for (int i = 0; i < blockMCount; i++) {
            SmallVector<Value> rowBlocks;
            rowBlocks.reserve(blockNCount);

            for (int j = 0; j < blockNCount; j++) {
                if (i < results2D.size() && j < results2D[i].size()) {
                    rowBlocks.push_back(results2D[i][j]);
                } else {
                    std::cerr << "Error: Index out of bounds in create2DConcat" << std::endl;
                    return nullptr;
                }
            }
            if (rowBlocks.size() > 1) {
                // outtype cal
                int rowSize = 0;
                for (int k = 0; k < rowBlocks.size(); k++) {
                    auto tensorType = llvm::dyn_cast<pimp::TensorType>(rowBlocks[k].getType());
                    rowSize += tensorType.getValueOfShape()[1];
                }

                mlir::Attribute encoding;
                auto outputType = pimp::TensorType::get(ArrayRef<int64_t>{llvm::dyn_cast<pimp::TensorType>(rowBlocks[0].getType()).getValueOfShape()[0], rowSize}, builder.getF32Type(),
                                                        builder.getStringAttr("concat_out_n"), encoding);

                auto rowConcat = builder.create<pimp::ConcatOp>(loc, outputType, rowBlocks,builder.getI32IntegerAttr(1),builder.getBoolAttr(
                        false),builder.getF32FloatAttr(-1),builder.getI32IntegerAttr(0),builder.getBoolAttr(false));  // falseaxis=1
                rowResults.push_back(rowConcat);
            } else {
                rowResults.push_back(rowBlocks[0]);
            }
        }

        // m axes concat
        if (rowResults.size() > 1) {
            int finalSize = 0;
            for (int k = 0; k < rowResults.size(); k++) {
                auto tensorType = llvm::dyn_cast<pimp::TensorType>(rowResults[k].getType());
                finalSize += tensorType.getValueOfShape()[0];
            }

            mlir::Attribute encoding;
            auto outputType = pimp::TensorType::get(ArrayRef<int64_t>{finalSize, llvm::dyn_cast<pimp::TensorType>(rowResults[0].getType()).getValueOfShape()[1]}, builder.getF32Type(),
                                                    builder.getStringAttr("out"), encoding);

            auto rowConcat = builder.create<pimp::ConcatOp>(loc, outputType, rowResults,builder.getI32IntegerAttr(0),builder.getBoolAttr(
                    false),builder.getF32FloatAttr(-1),builder.getI32IntegerAttr(0), builder.getBoolAttr(false));  // falseaxis=1

            return rowConcat;
        } else {
            return rowResults[0];
        }
    }

    void unrollMatMul(MatMulOp matmul, BlockInfo blockInfo){
        OpBuilder builder(matmul.getContext());
        builder.setInsertionPoint(matmul);

        SmallVector<Value> blockResults;

        auto lhsType = llvm::dyn_cast<pimp::TensorType>(matmul.getLhs().getType());
        auto rhsType = llvm::dyn_cast<pimp::TensorType>(matmul.getRhs().getType());

        std::vector<int64_t> lhsShape = lhsType.getValueOfShape();
        std::vector<int64_t> rhsShape = rhsType.getValueOfShape();

        int aBlockMCount = blockInfo.sizes[0].size();
        int aBlockKCount = getKBlockSize(lhsType, 1);

        int bBlockKCount = getKBlockSize(rhsType,0);
        int bBlockNCount = blockInfo.sizes[1].size();

        assert(aBlockKCount == bBlockKCount);

        Value final_value = nullptr;
        for (int i = 0; i < aBlockMCount; i++) {
            for (int k = 0; k < aBlockKCount; k++) {
                for (int j = 0; j < bBlockNCount; j++) {
                    //block param
                    int64_t aMStart = blockInfo.starts[0][i];
                    int64_t aKStart = getKStart(lhsType,1,k);
                    int64_t aMSize = blockInfo.sizes[0][i];
                    int64_t aKSize = getKSize(lhsType,1,k);

                    int64_t bKStart = getKStart(rhsType, 0,k);
                    int64_t bNStart = blockInfo.starts[1][j];
                    int64_t bKSize = getKSize(rhsType,0,k);
                    int64_t bNSize = blockInfo.sizes[1][j];

                    assert(aKSize == bKSize);
                    assert(aKStart == bKStart);
                    if (k == aBlockKCount - 1) {
                        //post process
                        final_value = createBlockMatMulWithAttrs(builder, matmul.getLoc(), matmul.getLhs(), matmul.getRhs(), matmul.getC(),
                                                   matmul.getBias(), matmul.getWithBias(),matmul.getRightTranspose(),
                                                   matmul.getLeftTranspose(),matmul.getOutputTranspose(),
                                                   matmul.getDoRelu(),matmul.getReluLimit().convertToFloat(),
                                                    aMStart, bNStart, aKStart, aMSize, bNSize, aKSize).getResult();
                    } else {
                        //no post process
                        final_value = createBlockMatMulWithAttrs(builder, matmul.getLoc(), matmul.getLhs(), matmul.getRhs(), matmul.getC(),
                                                                             nullptr, false,matmul.getRightTranspose(),matmul.getLeftTranspose(),
                                                                             matmul.getOutputTranspose(),false,-1.0,
                                                                             aMStart, bNStart, aKStart, aMSize, bNSize, aKSize).getResult();
                    }
                }
            }
        }

        auto concat_value = builder.create<pimp::ConcatOp>(matmul.getLoc(), matmul.getResult().getType(),matmul.getC(),builder.getI32IntegerAttr(-1),builder.getBoolAttr(
                false),builder.getF32FloatAttr(-1),builder.getI32IntegerAttr(0),builder.getBoolAttr(true)).getResult();

        assert(concat_value != nullptr);
        matmul.getResult().replaceAllUsesWith(concat_value);
        matmul.erase();
    }

    class PimpUnrollPass : public PimpUnrollBase<PimpUnrollPass> {
    public:
        PimpUnrollPass() = default;

        void runOnOperation() override {
            std::cout << "this is PimpUnrollPass" << std::endl;

            auto module = getOperation();

            for (auto func : module.getOps<mlir::func::FuncOp>()) {
                func.walk([&](Operation* op) {
                    // onchipParallel
                    if (auto matmul = llvm::dyn_cast<pimp::MatMulOp>(op)) {

                        auto lhsType = llvm::dyn_cast<pimp::TensorType>(op->getOperand(0).getType());
                        auto rhsType = llvm::dyn_cast<pimp::TensorType>(op->getOperand(1).getType());

                        if (lhsType.getOnchipParallel() && rhsType.getOnchipParallel()) {
                            auto tilingAttrl = lhsType.getOnchipParallel();
                            auto tilingAttrr = rhsType.getOnchipParallel();
                            BlockInfo blockInfo = parseMNTilingAttr(tilingAttrl, tilingAttrr);
                            unrollMatMul(matmul, blockInfo);
                            lhsType.setOnchipParallel(nullptr);
                            rhsType.setOnchipParallel(nullptr);
                        }
                    }
                });
            }
        }
    };

    std::unique_ptr<OperationPass<ModuleOp>> createPimpUnrollPass() {
        std::cout << "this is createPimpUnrollPass" << std::endl;
        return std::make_unique<PimpUnrollPass>();
    }
}  // namespace pimp