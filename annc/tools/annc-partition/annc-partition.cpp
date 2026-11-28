// CLI entry of the ANNC heterogeneous graph partitioner.
//
// The entry stays thin by design (docs/design/annc-partition/cli-design.md
// §5.1): the CliOptions module owns the query modes and the argument
// interpretation behind the §5.1 option table, and the stage ladder of §7
// belongs to PartitionTask, which owns everything from config loading
// onward and returns its first failure as a structured record. The entry
// only maps that record to the final failure output and exit code.

#include "CliOptions.h"
#include "PartitionTask.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);

  // Query modes are honored regardless of the validity of the remaining
  // arguments and never touch task state (§10.2 exit class 0).
  if (auto exitCode = annc_partition::handleQueryModes(argc, argv))
    return *exitCode;

  auto options = annc_partition::parseTaskOptions(argc, argv);
  if (!options)
    return annc_partition::reportFailure({"argument", "E_ARGUMENT",
                                          llvm::toString(options.takeError()),
                                          annc_partition::ExitCode::Argument});
  annc_partition::logStage(
      "argument",
      llvm::formatv("task options parsed (framework {0}, strategy {1})",
                    options->framework, options->strategy));

  annc_partition::PartitionTask task(std::move(*options));
  if (auto failure = task.run()) return annc_partition::reportFailure(*failure);
  return static_cast<int>(annc_partition::ExitCode::Success);
}
