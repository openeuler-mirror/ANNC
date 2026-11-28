#include "annc_optimizer.h"

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"

namespace tensorflow {
namespace grappler {

// 注册 ANNCOptimizer 到 Grappler 优化器注册表
REGISTER_GRAPH_OPTIMIZER(ANNCOptimizer);

}  // namespace grappler
}  // namespace tensorflow

// 符号导出（用于动态加载）
extern "C" {
// 返回优化器名称
const char* GetOptimizerName() {
  return "ANNCOptimizer";
}

void* CreateOptimizer() {
  return new tensorflow::grappler::ANNCOptimizer();
}

void DestroyOptimizer(void* optimizer) {
  delete static_cast<tensorflow::grappler::ANNCOptimizer*>(optimizer);
}
}
