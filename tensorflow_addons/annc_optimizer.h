#ifndef TENSORFLOW_ADDON_ANNC_OPTIMIZER_H_
#define TENSORFLOW_ADDON_ANNC_OPTIMIZER_H_

#include <sys/types.h>

#include <string>
#include <vector>

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

// ANNCOptimizer 是一个 Grappler 优化器插件
// 通过 fork 子进程调用外部编译器来优化 TensorFlow 计算图
class ANNCOptimizer : public CustomGraphOptimizer {
 public:
  ANNCOptimizer() = default;
  ~ANNCOptimizer() override = default;

  // 初始化优化器，从配置中读取参数
  Status Init(const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override;

  // 优化计算图的主入口
  // 输入：grappler_item 包含原始 GraphDef
  // 输出：优化后的 GraphDef
  Status Optimize(Cluster* cluster, const GrapplerItem& grappler_item,
                  GraphDef* output) override;

  // 返回优化器名称
  std::string name() const override { return "ANNCOptimizer"; }

  // 返回是否需要函数库
  bool UsesFunctionLibrary() const override { return false; }

  // 设置是否启用优化
  void SetEnabled(bool enabled) { enabled_ = enabled; }
  bool Enabled() const { return enabled_; }

 private:
  // 将 GraphDef 序列化到临时文件
  Status WriteGraphDefToFile(const GraphDef& graph_def, std::string* filepath);

  // 从临时文件读取优化后的 GraphDef
  Status ReadGraphDefFromFile(const std::string& filepath, GraphDef* graph_def);

  // 调用 annc-tf-pipeline：执行完整 GraphDef rewrite pipeline.
  Status InvokePipeline(const std::string& input_file,
                        const std::string& output_file);

  // 等待子进程完成
  Status WaitForProcess(pid_t pid, const std::string& process_name);

  // 清理临时文件
  void CleanupTempFile(const std::string& filepath);
  void CleanupTempFiles(const std::vector<std::string>& filepaths);

  // annc-tf-pipeline executable path.
  std::string pipeline_path_;

  // Optional work directory passed to annc-tf-pipeline. It must survive until the
  // rewritten graph executes because ANNCFused dlopens the generated .so.
  std::string annc_work_dir_;

  // 临时文件目录
  std::string temp_dir_;

  // 子进程超时时间（秒）
  int timeout_seconds_;

  // 是否启用优化
  bool enabled_;

  // 是否保留临时文件（用于调试）
  bool keep_temp_files_;

  // Whether to pass --verbose to annc-tf-pipeline.
  bool annc_verbose_;

  // 生成唯一临时文件名
  std::string GenerateTempFilename(const std::string& prefix);
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_ADDON_ANNC_OPTIMIZER_H_
