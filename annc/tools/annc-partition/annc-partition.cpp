// CLI entry of the ANNC heterogeneous graph partitioner.
//
// The entry owns the query modes and argument interpretation
// (docs/design/annc-partition/cli-design.md §5.1) and the final failure
// output with exit-code mapping; the stage ladder of §7 belongs to
// PartitionTask, which owns everything from config loading onward and
// returns its first failure as a structured record.

#include <string>
#include <vector>

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include "PartitionTask.h"

namespace {

using annc_partition::ExitCode;
using annc_partition::TaskOptions;

// Intentionally minimal: single-value and boolean options only.
// TODO: enforce the rest of the argument contract of §5.1 (reject repeated
// options, duplicate aliases, and empty values).
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

// The help text must mirror the capabilities this build actually registered
// (§4.3) and keep the shown defaults in sync once strict schema parsing
// (§5.3/§5.4) lands.
void printHelp() {
  llvm::outs()
      << "Usage: annc-partition --framework FRAMEWORK --model DIR"
         " --partition-config FILE --comm-config FILE --out-dir DIR"
         " [options]\n"
         "\n"
         "Stages: argument -> config -> adapter -> strategy -> plan ->"
         " export -> validate -> publish\n"
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
         "Registered capabilities:\n";
  for (const auto &cap : annc_partition::registeredCapabilities())
    llvm::outs() << "  " << cap.kind << " " << cap.name
                 << (cap.skeleton ? ": skeleton only" : ": available") << "\n";
  llvm::outs()
      << "\n"
         "Exit classes reserved by the design: 0, 2, 3, 4, 5, 6, 7, 8, 9, 70,"
         " 130, 143.\n";
}

void printVersion() {
  llvm::outs() << "annc-partition " << ANNC_PARTITION_VERSION
               << "\nsource revision: " << ANNC_SOURCE_REVISION << "\n";
  for (const auto &cap : annc_partition::registeredCapabilities())
    llvm::outs() << cap.kind << " " << cap.name
                 << (cap.skeleton ? ": skeleton only" : ": available") << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  std::vector<std::string> args(argv + 1, argv + argc);

  // Query modes are honored regardless of the validity of the remaining
  // arguments and never touch task state (§10.2 exit class 0).
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

  auto options = parseTaskOptions(args);
  if (!options)
    return annc_partition::reportFailure(
        {"argument", "E_ARGUMENT", llvm::toString(options.takeError()),
         ExitCode::Argument});
  annc_partition::logStage(
      "argument",
      llvm::formatv("task options parsed (framework {0}, strategy {1})",
                    options->framework, options->strategy));

  annc_partition::PartitionTask task(std::move(*options));
  if (auto failure = task.run())
    return annc_partition::reportFailure(*failure);
  return static_cast<int>(ExitCode::Success);
}
