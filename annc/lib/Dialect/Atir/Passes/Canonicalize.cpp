#include "Dialect/Atir/Passes/Passes.h"
#include "Dialect/Atir/AtirOps.h"
#include "Helper.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;

namespace atir {
static const std::set<std::string> g_support_weight_layouts = {"KN4", "N2K4N4"};

struct MatMulRewrite : public OpRewritePattern<MatMulOp> {
  MatMulRewrite(MLIRContext* context, PatternBenefit benefit = 9)
      : OpRewritePattern<MatMulOp>(context, benefit) {}

 public:
  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter& rewriter) const override {
    // only support right-hand side value is constant
    if (!llvm::isa<ConstantOp>(op.getRhs().getDefiningOp())) return failure();

    TensorType type = op.getRhs().getType();
    TilingAttr tiling = type.getOnchipParallel();  // tiling block
    StringAttr layout = type.getLayout();          // layout inside block
    CHECK_LOGICAL_SUCCESS(tiling != nullptr)
    CHECK_LOGICAL_SUCCESS(layout != nullptr)
    CHECK_LOGICAL_SUCCESS(g_support_weight_layouts.count(layout.str()))

    llvm::ArrayRef<int64_t> shape = type.getShape();
    CHECK_LOGICAL_SUCCESS(shape.size() == 2)
    auto elemType = type.getElementType();
    DenseElementsAttr elements = type.getCacheData();
    CHECK_LOGICAL_SUCCESS(elemType != nullptr)
    CHECK_LOGICAL_SUCCESS(elements != nullptr)

    int64_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                   std::multiplies<int64_t>());
    CHECK_LOGICAL_SUCCESS(elements.size() == size)

    if (elemType.isF32()) {
      const std::vector<int64_t> axes = tiling.getTilingAxes();
      const std::vector<std::vector<int64_t>> starts = tiling.getTilingStart();
      const std::vector<std::vector<int64_t>> sizes = tiling.getTilingSize();
      CHECK_LOGICAL_SUCCESS(axes.size() <= shape.size() &&
                            starts.size() == sizes.size())
      std::vector<float_t> reformatData;
      for (size_t i = 0; i < starts.size(); ++i) {
        const std::vector<int64_t>& start = starts[i];
        const std::vector<int64_t>& size = sizes[i];
        CHECK_LOGICAL_SUCCESS(start.size() == size.size() &&
                              start.size() == axes.size())
        sliceData(elements, axes, start, size, layout.str(), reformatData);
      }
      type.setCacheData(DenseElementsAttr::get(
          mlir::RankedTensorType::get(shape, rewriter.getF32Type()),
          llvm::ArrayRef<float_t>(reformatData)));
      return success();
    }
    return failure();
  };

 private:
  void sliceData(DenseElementsAttr elements, const std::vector<int64_t>& axes,
                 const std::vector<int64_t>& start,
                 const std::vector<int64_t>& size, std::string layout,
                 std::vector<float_t>& data) const {
    std::vector<int64_t> shape = elements.getType().getShape().vec();
    auto elems = elements.getValues<float_t>();
    std::vector<int64_t> sliceShape(shape.begin(), shape.end());
    if (axes.size() == 1 && axes[0] == 0) {
      // tiling k
      sliceShape[0] = size[0];
      std::vector<float_t> block(
          elems.begin() + start[0] * shape[1],
          elems.begin() + start[0] * shape[1] + size[0] * shape[1]);
      std::vector<float_t> refBlock = formatData(block, sliceShape, layout);
      data.insert(data.end(), refBlock.begin(), refBlock.end());
    } else if (axes.size() == 1 && axes[0] == 1) {
      // tiling n
      sliceShape[1] = size[0];
      for (int64_t i = 0; i < shape[0]; ++i) {
        std::vector<float_t> block(elems.begin() + i * shape[1],
                                   elems.begin() + i * shape[1] + size[0]);
        std::vector<float_t> refBlock = formatData(block, sliceShape, layout);
        data.insert(data.end(), refBlock.begin(), refBlock.end());
      }
    } else if (axes.size() == 2 && axes[0] == 0 && axes[1] == 1) {
      // tiling k, n
      sliceShape[0] = size[0];
      sliceShape[1] = size[1];
      for (int64_t i = start[0]; i < start[0] * size[0]; ++i) {
        std::vector<float_t> block(
            elems.begin() + i * shape[1] + start[1],
            elems.begin() + i * shape[1] + start[1] + size[1]);
        std::vector<float_t> refBlock = formatData(block, sliceShape, layout);
        data.insert(data.end(), refBlock.begin(), refBlock.end());
      }
    } else {
      llvm::report_fatal_error("Invalid tiling message");
    }
  }
  std::vector<float_t> formatData(std::vector<float_t> block,
                                  const std::vector<int64_t>& shape,
                                  const std::string& layout) const {
    std::vector<float_t> refBlock;
    int64_t k = shape[0], n = shape[1];
    if (layout == "KN4") {
      if (n % 4 != 0)
        llvm::report_fatal_error(llvm::StringRef(
            "Invalid shape for KN4 format: n_size = " + std::to_string(n)));
      for (int64_t kk = 0; kk < k; ++kk) {
        for (int64_t nn = 0; nn < n; nn += 4) {
          refBlock.insert(refBlock.end(), block.begin() + kk * n + nn,
                          block.begin() + kk * n + nn + 4);
        }
      }
    } else if (layout == "N2K4N4") {
      if (k % 4 != 0 || n % 4 != 0)
        llvm::report_fatal_error(llvm::StringRef(
            "Invalid shape for KN4 format: n_size = " + std::to_string(n) +
            ", k_size = " + std::to_string(k)));
      for (int64_t nn = 0; nn < n; nn += 8) {
        for (int64_t kk = 0; kk < k; ++kk) {
          for (int64_t nn = 0; nn < n; nn += 2) {
            for (int64_t nn = 0; nn < n; nn += 2) {
              refBlock.insert(refBlock.end(), block.begin() + kk * n + nn,
                              block.begin() + kk * n + nn + 4);
            }
          }
        }
      }
    } else {
      llvm::report_fatal_error(llvm::StringRef("Invalid layout: " + layout));
    }
    return refBlock;
  }
};

class AtirCanonicalizePass : public AtirCanonicalizeBase<AtirCanonicalizePass> {
 public:
  AtirCanonicalizePass() = default;

  void runOnOperation() override {
    auto m = getOperation();
    auto ctx = m.getContext();

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);

    RewritePatternSet patterns(ctx);
    patterns.add<MatMulRewrite>(ctx);
    (void)applyPatternsGreedily(m, std::move(patterns), config);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAtirCanonicalizePass() {
  return std::make_unique<AtirCanonicalizePass>();
}
}  // namespace atir