#ifndef ANNC_TF2ATIR_NODE_INFO_ADAPTER_H
#define ANNC_TF2ATIR_NODE_INFO_ADAPTER_H

#include <string>
#include <vector>

#include "Builder/MLIROpBuilder.h"
#include "tf_tensor_resolver.h"

namespace annc::tf2atir {

class NodeInfoAdapter {
 public:
  bool adapt(const ResolvedTfGraph& graph, std::vector<annc::NodeInfo>& result,
             std::string& error) const;

 private:
  static std::string builderTensorName(const TensorRef& ref);
};

}  // namespace annc::tf2atir

#endif  // ANNC_TF2ATIR_NODE_INFO_ADAPTER_H
