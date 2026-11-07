// annc-partition 任务编排实现。
//
// run() 按设计文档（docs/design/annc-partition/cli-design.md §7）的
// 阶段顺序推进；模型接入起的阶段为「其他模块主流程占位」一节中的
// 最小类，落地后该节整体迁出替换，阶段梯子本身保持稳定。

#include "PartitionTask.h"

#include <utility>

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace annc_partition {

namespace {

// ---- 其他模块主流程占位（最小类） ----
// 以下类型与类仅呈现阶段交接关系，不含模块内部设计；真实实现按总体
// 设计（docs/design/annc-partition/overview.md §3）由各模块提供（TF
// Adapter/Exporter 落于 frontends/tensorflow/planner/，JSON Pattern
// 策略与公共规划落于公共 Planner 库），落地后本节整体迁出并替换
// run() 中的调用点。

struct ModelHandle {};   // 阶段 4：Adapter 产出的不可变源模型句柄
struct CandidateSet {};  // 阶段 5：策略产出的候选区域集合
struct PartitionPlan {}; // 阶段 6：公共规划产出的切分计划
struct ExportResult {};  // 阶段 7：Exporter 产出的分区导出结果

// 阶段 4 占位：接入源模型。
// TODO: TF Adapter 读取 SavedModel 并检查初版 profile（总体设计 §3.2）。
class TensorFlowAdapter {
public:
  llvm::Expected<ModelHandle> load(const TaskOptions &) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "TensorFlowAdapter::load is not"
                                   " implemented");
  }
};

// 阶段 5 占位：生成候选区域。
// TODO: JSON Pattern 策略复用提取后的 matcher（总体设计 §3.3）。
class JsonPatternStrategy {
public:
  llvm::Expected<CandidateSet> generateCandidates(const ModelHandle &) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "candidate generation is not"
                                   " implemented");
  }
};

// 阶段 6 占位：形成切分计划。
// TODO: 公共切分实现完成归属、合法性与边界校验（总体设计 §3.4）。
class PartitionPlanner {
public:
  llvm::Expected<PartitionPlan> createPlan(const ModelHandle &,
                                           const CandidateSet &) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "partition planning is not implemented");
  }
};

// 阶段 7 占位：导出分区模型。
// TODO: TF Exporter 写出 A/B 与 Manifest 到 staging（总体设计 §3.5，
// 包布局见设计 §8.1）。
class TensorFlowExporter {
public:
  llvm::Expected<ExportResult> exportModel(const ModelHandle &,
                                           const PartitionPlan &) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "TensorFlowExporter::exportModel is"
                                   " not implemented");
  }
};

}  // namespace

PartitionTask::PartitionTask(TaskOptions taskOptions)
    : options(std::move(taskOptions)) {}

void logStage(llvm::StringRef stage, llvm::StringRef message) {
  llvm::errs() << "[annc-partition][" << stage << "] " << message << "\n";
}

int reportFailure(const TaskFailure &failure) {
  llvm::errs() << "[annc-partition][" << failure.stage << "]["
               << failure.code << "] " << failure.message << "\n";
  return static_cast<int>(failure.exitCode);
}

std::optional<TaskFailure> PartitionTask::checkCapability() {
  if (options.framework != "tensorflow" || options.strategy != "json")
    return TaskFailure{
        "capability", "E_CAPABILITY_UNAVAILABLE",
        "unsupported framework/strategy combination; skeleton registers"
        " tensorflow+json+in_process only",
        ExitCode::Capability};
  return std::nullopt;
}

std::optional<TaskFailure> PartitionTask::loadConfigs() {
  // TODO: 严格 schema 解析（设计 §5.3/§5.4），失败映射退出码 2。
  auto partitionFile = llvm::MemoryBuffer::getFile(options.partitionConfig);
  if (!partitionFile)
    return TaskFailure{"config", "E_CONFIG_UNREADABLE",
                       partitionFile.getError().message() + ": " +
                           options.partitionConfig,
                       ExitCode::Io};
  partitionConfig = std::move(partitionFile.get());
  auto commFile = llvm::MemoryBuffer::getFile(options.commConfig);
  if (!commFile)
    return TaskFailure{"config", "E_CONFIG_UNREADABLE",
                       commFile.getError().message() + ": " +
                           options.commConfig,
                       ExitCode::Io};
  commConfig = std::move(commFile.get());
  return std::nullopt;
}

int PartitionTask::run() {
  // 阶段 2：能力检查。
  if (auto failure = checkCapability())
    return reportFailure(*failure);
  logStage("capability", "tensorflow+json+in_process selected");

  // 阶段 3：读取两份配置文件，原文保留待后续模块消费。
  if (auto failure = loadConfigs())
    return reportFailure(*failure);
  logStage("config",
           "loaded partition and comm config files (schema parsing TODO)");

  // 阶段 4：接入源模型（其他模块占位）。
  TensorFlowAdapter adapter;
  auto model = adapter.load(options);
  if (!model)
    return reportFailure({"adapter", "E_ANALYSIS",
                          llvm::toString(model.takeError()),
                          ExitCode::Analysis});
  logStage("adapter", "source model loaded");

  // 阶段 5：候选生成（其他模块占位）。
  JsonPatternStrategy strategy;
  auto candidates = strategy.generateCandidates(*model);
  if (!candidates)
    return reportFailure({"strategy", "E_ANALYSIS",
                          llvm::toString(candidates.takeError()),
                          ExitCode::Analysis});
  logStage("strategy", "candidate regions generated");

  // 阶段 6：形成切分计划（其他模块占位）。
  PartitionPlanner planner;
  auto plan = planner.createPlan(*model, *candidates);
  if (!plan)
    return reportFailure(
        {"plan", "E_PLAN", llvm::toString(plan.takeError()), ExitCode::Plan});
  logStage("plan", "partition plan created");

  // 阶段 7：导出分区模型（其他模块占位）。
  TensorFlowExporter exporter;
  auto exportResult = exporter.exportModel(*model, *plan);
  if (!exportResult)
    return reportFailure({"export", "E_EXPORT",
                          llvm::toString(exportResult.takeError()),
                          ExitCode::Export});
  logStage("export", "partition models exported");

  // 阶段 8-10：产物校验、Report 汇总与目录事务发布（设计 §8/§9）。
  // TODO: 校验汇总、原子发布；当前流程不会执行到此处。
  return static_cast<int>(ExitCode::Success);
}

}  // namespace annc_partition
