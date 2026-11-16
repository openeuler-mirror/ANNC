#include "Patterns/LayerNorm.h"

#include "HloGraph.h"
#include "Patterns/Elementwise.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace annc::stablehlo {
namespace {
std::optional<HloElementwise> element(Value value, StringRef kind,
                                      size_t arity) {
  if (!value) return std::nullopt;
  auto fragment = analyzeElementwise(value);
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!fragment || fragment->kind != kind || fragment->inputs.size() != arity ||
      !type || !type.getElementType().isF32())
    return std::nullopt;
  return fragment;
}
bool constant(Value value, int64_t rank, int64_t width = -1) {
  auto op = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
  auto t = dyn_cast<RankedTensorType>(value.getType());
  return op && isa<DenseFPElementsAttr>(op.getValue()) && t &&
         t.getElementType().isF32() && t.getRank() == rank &&
         (rank == 0 || t.getDimSize(0) == width);
}
}  // namespace

void partitionLayerNorm(HloGraph &graph,
                        SmallVectorImpl<Attribute> &rejections) {
  for (const auto &[id, op] : graph.operations) {
    if (!isa<mlir::stablehlo::AddOp>(op)) continue;
    Value output = op->getResult(0);
    auto out = element(output, "Add", 2);
    auto type = dyn_cast<RankedTensorType>(output.getType());
    if (!out || !type || type.getRank() != 2 || type.getDimSize(1) <= 0)
      continue;
    for (unsigned i = 0; i < 2; ++i) {
      auto xm = element(out->inputs[i], "Mul", 2);
      auto bs = element(out->inputs[1 - i], "Sub", 2);
      if (!xm || !bs || !constant(bs->inputs[0], 1, type.getDimSize(1)))
        continue;
      auto mg = element(bs->inputs[1], "Mul", 2);
      if (!mg) continue;
      for (unsigned j = 0; j < 2; ++j) {
        Value x = xm->inputs[j], g = xm->inputs[1 - j];
        if (x.getType() != output.getType()) continue;
        Value mean;
        if (mg->inputs[0] == g)
          mean = mg->inputs[1];
        else if (mg->inputs[1] == g)
          mean = mg->inputs[0];
        else
          continue;
        auto m = element(mean, "ReduceMean", 1);
        auto gm = element(g, "Mul", 2);
        if (!m || m->inputs[0] != x || !gm) continue;
        for (unsigned k = 0; k < 2; ++k) {
          Value gamma = gm->inputs[k];
          if (!constant(gamma, 1, type.getDimSize(1))) continue;
          auto reciprocal = element(gm->inputs[1 - k], "Reciprocal", 1);
          auto root = reciprocal ? element(reciprocal->inputs[0], "Sqrt", 1)
                                 : std::nullopt;
          auto ve = root ? element(root->inputs[0], "Add", 2) : std::nullopt;
          if (!ve) continue;
          for (unsigned e = 0; e < 2; ++e) {
            Value eps = ve->inputs[e];
            if (!constant(eps, 0)) continue;
            auto variance = element(ve->inputs[1 - e], "ReduceMean", 1);
            auto square = variance ? element(variance->inputs[0], "Mul", 2)
                                   : std::nullopt;
            if (!square || square->inputs[0] != square->inputs[1]) continue;
            auto delta = element(square->inputs[0], "Sub", 2);
            if (!delta || delta->inputs[0] != x || delta->inputs[1] != mean)
              continue;
            graph.fuse("layer_norm", "LayerNorm",
                       {x, gamma, bs->inputs[0], eps}, {output}, {},
                       rejections);
          }
        }
      }
    }
  }
}

}  // namespace annc::stablehlo
