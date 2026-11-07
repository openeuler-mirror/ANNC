// ANNC 异构图切分优化器 CLI 入口。
//
// 入口承担查询模式（--help/--version，无任务副作用）与参数解释
// （设计文档 docs/design/annc-partition/cli-design.md §5.1），随后把
// 任务交给 PartitionTask 按阶段顺序编排（§7）：能力检查 → 配置读取 →
// 模型接入 → 候选生成 → 切分规划 → 分区导出 → 校验汇总 → 目录发布。
// 模型接入起的阶段为其他模块的最小类占位，统一返回 not implemented，
// 流程在首个占位调用（模型接入）处终止。

#include <string>
#include <vector>

#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include "PartitionTask.h"

namespace {

using annc_partition::ExitCode;
using annc_partition::TaskOptions;

// 极简参数解释：单值选项按 "--flag value" 匹配，布尔选项无值。
// TODO: 按设计 §5.1 补齐重复参数、别名重复与空值拒绝。
llvm::Expected<TaskOptions>
parseTaskOptions(const std::vector<std::string> &args) {
  TaskOptions options;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string &arg = args[i];
    auto value = [&](const char *name, std::string &out) -> llvm::Error {
      if (i + 1 >= args.size())
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       std::string("missing value for ") +
                                           name);
      out = args[++i];
      return llvm::Error::success();
    };
    if (arg == "--framework") {
      if (llvm::Error e = value("--framework", options.framework))
        return e;
    } else if (arg == "--model") {
      if (llvm::Error e = value("--model", options.model))
        return e;
    } else if (arg == "--strategy") {
      if (llvm::Error e = value("--strategy", options.strategy))
        return e;
    } else if (arg == "--partition-config") {
      if (llvm::Error e = value("--partition-config", options.partitionConfig))
        return e;
    } else if (arg == "--comm-config") {
      if (llvm::Error e = value("--comm-config", options.commConfig))
        return e;
    } else if (arg == "--out-dir") {
      if (llvm::Error e = value("--out-dir", options.outDir))
        return e;
    } else if (arg == "--tf-signature") {
      if (llvm::Error e = value("--tf-signature", options.tfSignature))
        return e;
    } else if (arg == "--work-dir") {
      if (llvm::Error e = value("--work-dir", options.workDir))
        return e;
    } else if (arg == "--keep-temps") {
      options.keepTemps = true;
    } else if (arg == "--verbose" || arg == "-v") {
      options.verbose = true;
    } else {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "unknown argument: " + arg);
    }
  }
  auto missing = [](const char *option) -> llvm::Error {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   std::string("missing required option: ") +
                                       option);
  };
  if (options.framework.empty())
    return missing("--framework");
  if (options.model.empty())
    return missing("--model");
  if (options.partitionConfig.empty())
    return missing("--partition-config");
  if (options.commConfig.empty())
    return missing("--comm-config");
  if (options.outDir.empty())
    return missing("--out-dir");
  return std::move(options);
}

// TODO: 按设计 §5.3/§5.4 实现严格 schema 解析后，帮助文本同步默认值与能力表。
void printHelp() {
  llvm::outs()
      << "Usage: annc-partition --framework tensorflow --model DIR"
         " --partition-config FILE --comm-config FILE --out-dir DIR"
         " [options]\n"
         "\n"
         "Stages: argument -> capability -> config -> adapter -> strategy ->"
         " plan -> export -> validate -> publish\n"
         "The pipeline is a skeleton: stages after config are module"
         " placeholders returning \"not implemented\".\n"
         "\n"
         "Options:\n"
         "  --strategy json\n"
         "  --tf-signature serving_default\n"
         "  --work-dir DIR\n"
         "  --keep-temps\n"
         "  --verbose, -v\n"
         "  --help, -h\n"
         "  --version\n"
         "\n"
         "Registered capabilities:\n"
         "  tensorflow+json+in_process: skeleton only\n"
         "\n"
         "Exit classes reserved by the design: 0, 2, 3, 4, 5, 6, 7, 8, 9, 70,"
         " 130, 143.\n";
}

void printVersion() {
  llvm::outs() << "annc-partition " << ANNC_PARTITION_VERSION
               << "\nsource revision: " << ANNC_SOURCE_REVISION
               << "\ntensorflow+json+in_process: skeleton only\n";
}

}  // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  std::vector<std::string> args(argv + 1, argv + argc);

  // 查询模式：无任务副作用。
  for (const std::string &arg : args) {
    if (arg == "--help" || arg == "-h") {
      printHelp();
      return static_cast<int>(ExitCode::Success);
    }
    if (arg == "--version") {
      printVersion();
      return static_cast<int>(ExitCode::Success);
    }
  }

  // 阶段 1：参数解释。
  auto options = parseTaskOptions(args);
  if (!options)
    return annc_partition::reportFailure(
        {"argument", "E_ARGUMENT", llvm::toString(options.takeError()),
         ExitCode::Argument});
  annc_partition::logStage("argument", "task options parsed");

  // 阶段 2-10：任务编排。
  annc_partition::PartitionTask task(std::move(*options));
  return task.run();
}
