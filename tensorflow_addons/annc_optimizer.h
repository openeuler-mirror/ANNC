#ifndef TENSORFLOW_ADDON_ANNC_OPTIMIZER_H_
#define TENSORFLOW_ADDON_ANNC_OPTIMIZER_H_

#include <sys/types.h>

#include <string>
#include <vector>

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

class ANNCOptimizer : public CustomGraphOptimizer {
 public:
  ANNCOptimizer() = default;
  ~ANNCOptimizer() override = default;

  Status Init(const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override;

  Status Optimize(Cluster* cluster, const GrapplerItem& grappler_item,
                  GraphDef* output) override;

  std::string name() const override { return "ANNCOptimizer"; }

  bool UsesFunctionLibrary() const override { return false; }

  void SetEnabled(bool enabled) { enabled_ = enabled; }
  bool Enabled() const { return enabled_; }

 private:
  Status WriteGraphDefToFile(const GraphDef& graph_def, std::string* filepath);
  Status ReadGraphDefFromFile(const std::string& filepath, GraphDef* graph_def);
  Status InvokePipeline(const std::string& input_file,
                        const std::string& output_file);
  Status WaitForProcess(pid_t pid, const std::string& process_name);
  void CleanupTempFile(const std::string& filepath);
  void CleanupTempFiles(const std::vector<std::string>& filepaths);
  void LoadFusedOpLibrary();

  std::string pipeline_path_;
  std::string annc_work_dir_;
  std::string temp_dir_;
  std::string savedmodel_path_;
  std::string backend_;
  int timeout_seconds_;
  bool enabled_;
  bool keep_temp_files_;
  bool annc_verbose_;
  int64_t batch_size_ = 2;

  std::string GenerateTempFilename(const std::string& prefix);
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_ADDON_ANNC_OPTIMIZER_H_