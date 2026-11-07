#ifndef ANNC_PARTITION_TASK_H
#define ANNC_PARTITION_TASK_H

// annc-partition 的任务公共契约：一次离线切分任务的选项、退出分类、
// 失败信息与任务编排对象。入口（annc-partition.cpp）完成查询模式与
// 参数解释后，把任务交给 PartitionTask 按设计文档
// （docs/design/annc-partition/cli-design.md §7）的阶段顺序推进。

#include <memory>
#include <optional>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

namespace annc_partition {

// 退出码分类依据设计文档 §10.2；8（产物校验）、9（发布）、70（内部
// 错误）与 130/143（取消）保留，随对应阶段实现补充。
enum class ExitCode : int {
  Success = 0,
  Argument = 2,   // 参数或配置契约错误
  Capability = 3, // 能力不可用
  Io = 4,         // 提交前输入输出错误
  Analysis = 5,   // 模型接入或分析失败
  Plan = 6,       // 规划失败
  Export = 7,     // 模型导出失败
};

// 任务选项：参数解释阶段产出，贯穿全部后续阶段的公共关键信息。
struct TaskOptions {
  std::string framework;
  std::string model;
  std::string strategy = "json";
  std::string partitionConfig;
  std::string commConfig;
  std::string outDir;
  std::string tfSignature = "serving_default";
  std::string workDir;
  bool keepTemps = false;
  bool verbose = false;
};

// 一次任务失败的公共信息：阶段、稳定错误标识、原因与退出分类。
// 首次失败决定主退出分类，后续清理告警不覆盖根因（设计 §10.1）。
struct TaskFailure {
  std::string stage;
  std::string code;
  std::string message;
  ExitCode exitCode = ExitCode::Success;
};

// 按诊断契约输出一条阶段进度日志到 stderr（设计 §10.1 的阶段诊断
// 格式），入口与任务共用。
void logStage(llvm::StringRef stage, llvm::StringRef message);

// 按诊断契约输出一次失败并返回对应进程退出码；入口的参数阶段与
// 任务的各阶段共用同一失败格式（设计 §10.1）。
int reportFailure(const TaskFailure &failure);

// 一次离线切分任务（设计 §7）。
//
// 按阶段顺序编排：能力检查 → 配置读取 → 模型接入 → 候选生成 →
// 切分规划 → 分区导出 → 校验汇总 → 目录发布；当前模型接入起的阶段
// 为其他模块的最小类占位，统一返回 not implemented，流程在首个占位
// 调用（模型接入）处终止。任务持有选项与已加载的配置原文（设计
// §5.3，待策略/规划/导出模块消费）；失败保留首次根因。scratch/
// staging 的清理责任随目录事务实现（设计 §9）落在本对象的析构路径。
class PartitionTask {
public:
  explicit PartitionTask(TaskOptions options);

  // 推进各阶段并返回进程退出码；占位阶段在模型接入处以退出码 5 终止。
  int run();

private:
  // 阶段 2：能力检查。骨架期静态注册 tensorflow+json+in_process。
  std::optional<TaskFailure> checkCapability();

  // 阶段 3：读取分区与通信配置，原文保留待后续模块消费。
  std::optional<TaskFailure> loadConfigs();

  TaskOptions options;
  std::unique_ptr<llvm::MemoryBuffer> partitionConfig;
  std::unique_ptr<llvm::MemoryBuffer> commConfig;
};

}  // namespace annc_partition

#endif  // ANNC_PARTITION_TASK_H
