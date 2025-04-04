#include <cstdint>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "transforms/passes.h"

namespace mlir {

#define GEN_PASS_DEF_TILELINALGCOPYPASS
#include "transforms/passes.h.inc"

namespace {

class TileLinalgCopyPass
    : public impl::TileLinalgCopyPassBase<TileLinalgCopyPass> {
public:
  using TileLinalgCopyPassBase<TileLinalgCopyPass>::TileLinalgCopyPassBase;
  TileLinalgCopyPass(int tileSize_) {
    tileSize = tileSize_;
  }

private:
  int tileSize;
  void runOnOperation() override;
};

} // namespace

void TileLinalgCopyPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  getOperation()->walk([&](linalg::CopyOp op) {
    // Tiling must happen after bufferization.
    if (!op.hasBufferSemantics()) {
      emitError(op.getLoc(), "expected buffer semantics");
      signalPassFailure();
    }

    mlir::linalg::LinalgTilingOptions options;
    std::vector<int64_t> tile_sizes_vec({tileSize});
    ArrayRef<int64_t> ts(tile_sizes_vec);
    options.setTileSizes(ts);

    FailureOr<linalg::TiledLinalgOp> maybeTiled =
        linalg::tileLinalgOp(rewriter, op, options);
    if (failed(maybeTiled)) {
      emitError(op.getLoc(), "failed to tile");
      signalPassFailure();
    }

    if (maybeTiled->loops.size() == 1) {
      // Replace the old linalg.copy with the tiled loop
      rewriter.replaceOp(op, maybeTiled->loops[0]);
    } else {
      emitError(op.getLoc(), "expected to find exactly one loop");
      signalPassFailure();
    }
  });
}

std::unique_ptr<Pass> hlo::createTileLinalgCopyPass(int tileSize) {
  return std::make_unique<TileLinalgCopyPass>(tileSize);
}

} // namespace mlir
