#include "Patterns/MaskedMul.h"

#include <cmath>
#include <cstring>
#include <functional>

#include "HloGraph.h"
#include "Patterns/Elementwise.h"
#include "Patterns/ShapeProof.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
namespace annc::stablehlo {
namespace {
struct Mask {
  Value flat, ids;
  int64_t padding;
  int64_t width = 1;
};
struct Site {
  Value output, data;
};
std::optional<Mask> mask(Value value) {
  Mask result;
  auto part = analyzeElementwise(value);
  if (part && part->kind == "Tile") {
    result.width = cast<RankedTensorType>(value.getType()).getDimSize(2);
    value = part->inputs[0];
    part = analyzeElementwise(value);
  }
  if (!part || part->kind != "Unsqueeze") return std::nullopt;
  result.flat = part->inputs[0];
  auto castOp = result.flat.getDefiningOp<mlir::stablehlo::ConvertOp>();
  auto flatType = dyn_cast<RankedTensorType>(result.flat.getType());
  if (!castOp || !flatType || flatType.getRank() != 2 ||
      !flatType.getElementType().isF32())
    return std::nullopt;
  auto invert = castOp.getOperand().getDefiningOp<mlir::stablehlo::NotOp>();
  auto eq = invert ? analyzeElementwise(invert.getOperand()) : std::nullopt;
  if (!eq || eq->kind != "Equal") return std::nullopt;
  for (unsigned i = 0; i < 2; ++i) {
    auto constant = eq->inputs[i].getDefiningOp<mlir::stablehlo::ConstantOp>();
    auto data = constant ? dyn_cast<DenseIntElementsAttr>(constant.getValue())
                         : DenseIntElementsAttr{};
    auto ids = dyn_cast<RankedTensorType>(eq->inputs[1 - i].getType());
    if (!data || !data.isSplat() || !data.getElementType().isInteger(64) ||
        !ids || ids.getRank() != 2 || !ids.getElementType().isInteger(64) ||
        ids.getShape() != flatType.getShape())
      continue;
    result.ids = eq->inputs[1 - i];
    result.padding = data.getSplatValue<APInt>().getSExtValue();
    return result;
  }
  return std::nullopt;
}
bool dataShape(Value value, const Mask &mask, RankedTensorType common = {}) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  auto ids = cast<RankedTensorType>(mask.ids.getType());
  if (!type || type.getRank() != 3 || !type.getElementType().isF32() ||
      type.getShape().take_front(2) != ids.getShape() ||
      type.getDimSize(2) == 0 ||
      (mask.width != 1 && type.getDimSize(2) != mask.width))
    return false;
  return !common || common.getShape() == type.getShape();
}
bool scalar(Value value, float &number) {
  auto c = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
  auto data =
      c ? dyn_cast<DenseFPElementsAttr>(c.getValue()) : DenseFPElementsAttr{};
  if (!data || !data.getElementType().isF32() || data.getNumElements() != 1)
    return false;
  number = data.getSplatValue<APFloat>().convertToFloat();
  return std::isfinite(number) && number != 0;
}
int64_t bits(float number) {
  uint32_t result;
  std::memcpy(&result, &number, sizeof(result));
  return result;
}

bool fuseMask(HloGraph &graph, Value maskValue, const Mask &mask,
              SmallVectorImpl<Attribute> &rejections) {
  // Inspect actual HLO operands. Broadcast wrappers are accepted only after
  // analyzeElementwise proves their dimensions and runtime shape program.
  SmallVector<Site> sites;
  for (const auto &[id, op] : graph.operations) {
    if (!isa<mlir::stablehlo::MulOp>(op)) continue;
    auto fragment = analyzeElementwise(op->getResult(0));
    if (!fragment || fragment->kind != "Mul") continue;
    Value source;
    if (fragment->inputs[0] == maskValue)
      source = fragment->inputs[1];
    else if (fragment->inputs[1] == maskValue)
      source = fragment->inputs[0];
    else
      continue;
    if (!dataShape(source, mask) ||
        source.getType() != op->getResult(0).getType())
      return false;
    sites.push_back({op->getResult(0), source});
  }
  if (sites.empty()) return false;
  llvm::sort(sites, [](const Site &a, const Site &b) {
    return a.output.getDefiningOp()->isBeforeInBlock(b.output.getDefiningOp());
  });
  auto common = cast<RankedTensorType>(sites.front().output.getType());
  for (const auto &site : sites)
    if (!dataShape(site.data, mask, common)) return false;
  auto findSite = [&](Value v) -> const Site * {
    auto it =
        llvm::find_if(sites, [&](const Site &s) { return s.output == v; });
    return it == sites.end() ? nullptr : &*it;
  };
  std::function<bool(Value, unsigned)> depends = [&](Value v, unsigned depth) {
    if (findSite(v)) return true;
    auto *op = v.getDefiningOp();
    if (!op || isa<mlir::stablehlo::ConstantOp>(op)) return false;
    if (depth > 64) return true;
    return llvm::any_of(op->getOperands(),
                        [&](Value x) { return depends(x, depth + 1); });
  };
  SmallVector<Site> raw, fed;
  SmallVector<Value> terms;
  Value interiorValue;
  int64_t interiorKind = 0;
  float divisor = 1;
  for (const auto &site : sites) {
    Value source = site.data;
    // Repeated application of the same 0/1 mask shares a raw source.
    unsigned guard = 0;
    while (const Site *upstream = findSite(source)) {
      if (++guard > sites.size()) return false;
      source = upstream->data;
    }
    if (!depends(source, 0)) {
      raw.push_back({site.output, source});
      continue;
    }
    if (interiorValue && interiorValue != source) return false;
    if (!interiorValue) {
      interiorValue = source;
      auto expr = analyzeElementwise(source);
      if (expr && expr->kind == "Div") {
        if (!scalar(expr->inputs[1], divisor)) return false;
        expr = analyzeElementwise(expr->inputs[0]);
      }
      if (!expr ||
          (expr->kind != "Add" && expr->kind != "Sum" && expr->kind != "Mul"))
        return false;
      interiorKind = expr->kind == "Mul";
      terms = expr->inputs;
    }
    fed.push_back({site.output, source});
  }
  if (raw.empty() || raw.size() > 16) return false;
  // Use the existing MaskedMul accumulation convention: masked results first,
  // followed by extra operands. No per-site HLO term-order attributes.
  SmallVector<Site> ordered;
  SmallVector<Value> extras;
  for (Value term : terms) {
    auto it =
        llvm::find_if(raw, [&](const Site &s) { return s.output == term; });
    if (it != raw.end()) {
      if (llvm::any_of(ordered,
                       [&](const Site &s) { return s.output == term; }))
        return false;
      ordered.push_back(*it);
    } else {
      if (depends(term, 0) || !dataShape(term, mask, common) ||
          llvm::is_contained(extras, term))
        return false;
      extras.push_back(term);
    }
  }
  int64_t sumMask = (int64_t{1} << ordered.size()) - 1;
  for (const Site &site : raw)
    if (!llvm::any_of(ordered,
                      [&](const Site &s) { return s.output == site.output; }))
      ordered.push_back(site);
  SmallVector<Value> args, outputs;
  int64_t argMap = 0;
  for (const auto &site : ordered) {
    auto it = llvm::find(args, site.data);
    auto slot = it - args.begin();
    if (it == args.end()) args.push_back(site.data);
    argMap |= static_cast<uint64_t>(slot) << (4 * outputs.size());
    outputs.push_back(site.output);
  }
  int64_t rawCount = args.size();
  for (Value extra : extras) {
    if (llvm::is_contained(args, extra)) return false;
    args.push_back(extra);
  }
  for (const auto &site : fed) outputs.push_back(site.output);
  // Cast output may also feed model-level consumers (including Shape/Expand).
  // Let the common region verifier decide whether it needs a flat output.
  llvm::DenseSet<Value> boundary(args.begin(), args.end());
  boundary.insert(mask.ids);
  llvm::DenseSet<Operation *> region;
  std::function<void(Value)> collect = [&](Value value) {
    if (boundary.count(value)) return;
    auto *op = value.getDefiningOp();
    if (!op || !region.insert(op).second) return;
    for (Value input : op->getOperands()) collect(input);
  };
  for (Value output : outputs) collect(output);
  bool flat = llvm::any_of(mask.flat.getUsers(), [&](Operation *user) {
    return !region.count(user) && !HloGraph::shapeUse(user);
  });
  if (flat) outputs.push_back(mask.flat);
  std::string builtin =
      "MaskedMul" + std::to_string(args.size()) + (flat ? "Flat" : "");
  SmallVector<int64_t> attrs{rawCount,
                             static_cast<int64_t>(raw.size()),
                             static_cast<int64_t>(fed.size()),
                             argMap,
                             sumMask,
                             interiorKind,
                             bits(divisor),
                             mask.padding};
  args.push_back(mask.ids);
  return graph.fuse("masked_mul", builtin, args, outputs, attrs, rejections);
}
}  // namespace

void partitionMaskedMul(HloGraph &graph,
                        SmallVectorImpl<Attribute> &rejections) {
  for (const auto &[id, op] : graph.operations) {
    if (!isa<mlir::stablehlo::ReshapeOp, mlir::stablehlo::DynamicReshapeOp>(op))
      continue;
    Value value = op->getResult(0);
    auto chain = mask(value);
    if (!chain) continue;
    // Try Tile first so a shared unsqueezed chain forms one complete region.
    bool fused = false;
    for (const auto &[otherId, other] : graph.operations) {
      if (!isa<mlir::stablehlo::DynamicReshapeOp>(other)) continue;
      auto fragment = analyzeElementwise(other->getResult(0));
      if (fragment && fragment->kind == "Tile" &&
          fragment->inputs[0] == value) {
        auto tiled = mask(other->getResult(0));
        if (tiled)
          fused |= fuseMask(graph, other->getResult(0), *tiled, rejections);
      }
    }
    if (!fused) fuseMask(graph, value, *chain, rejections);
  }
}
}  // namespace annc::stablehlo
