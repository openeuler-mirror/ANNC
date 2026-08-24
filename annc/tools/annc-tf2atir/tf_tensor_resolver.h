#ifndef ANNC_TF2ATIR_TF_TENSOR_RESOLVER_H
#define ANNC_TF2ATIR_TF_TENSOR_RESOLVER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tf_frontend.h"

namespace annc::tf2atir {

struct TensorDescriptor {
  std::string dtype;
  std::vector<int64_t> shape;
  bool rank_known = true;
};

struct ShapeOverridePolicy {
  int64_t batch_size = -1;
};

struct ResolvedTfGraph {
  // This graph owns the complete output-slot list for every node.
  TfGraph graph;
  std::unordered_map<TensorRef, TensorDescriptor, TensorRefHash> tensors;

  const TensorDescriptor* find(const TensorRef& ref) const;
};

class TfTensorResolver {
 public:
  bool resolve(const TfGraph& graph, ResolvedTfGraph& result,
               std::string& error, ShapeOverridePolicy shape_policy = {}) const;
};

}  // namespace annc::tf2atir

#endif  // ANNC_TF2ATIR_TF_TENSOR_RESOLVER_H
