#ifndef ANNC_PARTITION_CLI_OPTIONS_H
#define ANNC_PARTITION_CLI_OPTIONS_H

// CLI option management of annc-partition: the declarative option table of
// docs/design/annc-partition/cli-design.md §5.1, the query modes, and the
// interpretation of argv into TaskOptions. Keeping them in this module
// leaves the entry a thin stage driver and gives option names, value
// syntax, defaults, requiredness, and help text a single source.

#include <optional>

#include "PartitionTask.h"
#include "llvm/Support/Error.h"

namespace annc_partition {

// Serves --help/-h and --version regardless of the validity of the
// remaining arguments and without touching task state (§10.2 exit class
// 0). Returns the exit code when a query mode was served.
std::optional<int> handleQueryModes(int argc, char *const *argv);

// Interprets argv against the option table and returns the task options,
// or the first argument failure as an error, which the entry maps to exit
// class 2 (§10.2). Query modes must be served before calling this.
llvm::Expected<TaskOptions> parseTaskOptions(int argc, char *const *argv);

}  // namespace annc_partition

#endif  // ANNC_PARTITION_CLI_OPTIONS_H
