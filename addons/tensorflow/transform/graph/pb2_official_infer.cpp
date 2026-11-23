#include "common/pb2_common.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"

int EnrichGraphWithOfficialInference(GraphDef& gp, const Options& opt,
                                     const BuildCtx& ctx) {
  // Diagnostic switch: A/B for measuring what the official layer adds
  // (default runs it; ANNC_NO_OFFICIAL_INFER=1 keeps the graph as-is).
  if (std::getenv("ANNC_NO_OFFICIAL_INFER") != nullptr) {
    std::cerr << "[official-infer] disabled by env (kept as-is)\n";
    return 0;
  }
  tensorflow::grappler::GrapplerItem item;
  item.id = "pb2stablehlo";
  item.graph = gp;
  // Feeds = user entries with explicit shapes: assume_valid_feeds lets the
  // fanout be derived from them (Tensor default shape is scalar; entries
  // without dims contribute dtype-only facts through their node attr).
  for (auto& [name, e] : ctx.entries) {
    if (!e.has_shape) continue;
    tensorflow::TensorShape shape;
    for (int64_t d : e.dims) shape.AddDim(d < 0 ? 1 : d);
    item.feed.emplace_back(name, tensorflow::Tensor(e.dt, shape));
  }
  tensorflow::grappler::GraphProperties props(item);
  auto st = props.InferStatically(/*assume_valid_feeds=*/true,
                                  /*aggressive_shape_inference=*/false,
                                  /*include_tensor_values=*/false);
  if (!st.ok()) {
    std::cerr << "[official-infer] InferStatically failed (kept as-is): "
              << st.ToString() << "\n";
    return 0;
  }
  tensorflow::GraphDef enriched;
  st = props.AnnotateOutputShapes(&enriched);
  if (!st.ok()) {
    std::cerr << "[official-infer] AnnotateOutputShapes failed (kept as-is): "
              << st.ToString() << "\n";
    return 0;
  }
  int annotated = 0;
  for (auto& nd : enriched.node())
    if (FindAttr(nd, "_output_shapes") != nullptr) ++annotated;
  std::cerr << "[official-infer] static inference annotated " << annotated
            << "/" << enriched.node_size() << " nodes\n";
  gp = std::move(enriched);
  return 0;
}
