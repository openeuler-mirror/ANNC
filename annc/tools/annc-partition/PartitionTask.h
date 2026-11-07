#ifndef ANNC_PARTITION_TASK_H
#define ANNC_PARTITION_TASK_H

// Shared task contract of annc-partition: task-wide options, the exit-code
// classification, the failure record, the capability registry, and the task
// object walking the stage ladder of docs/design/annc-partition/cli-design.md
// §7. The CliOptions module owns the query modes and the §5.1 option table
// behind argument interpretation, and the CLI entry keeps only the final
// failure output with exit-code mapping; everything from config loading
// onward belongs to the task.

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"

namespace annc_partition {

// Exit-code classification of §10.2. Codes 8 (artifact validation), 9
// (publish), 70 (internal error), and 130/143 (cancellation) stay reserved
// until the corresponding stages exist.
enum class ExitCode : int {
  Success = 0,
  Argument = 2,
  Capability = 3,
  Io = 4,
  Analysis = 5,
  Plan = 6,
  Export = 7,
};

// Task-wide options produced by argument interpretation and consumed by
// every later stage.
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

// One task failure. The first failure fixes the root cause: later cleanup
// warnings must not overwrite the recorded stage, identifier, or exit
// class (§10.1).
struct TaskFailure {
  std::string stage;
  std::string code;
  std::string message;
  ExitCode exitCode = ExitCode::Success;
};

// One entry of the fixed capability registry of this build: the kind
// ("framework" or "strategy"), the registered name, and whether the linked
// implementation is still a skeleton placeholder. The registry is the
// single source for capability selection and for the capability display of
// --help/--version (§4.3).
struct CapabilityDesc {
  llvm::StringRef kind;
  llvm::StringRef name;
  bool skeleton;
};

// Returns the capabilities registered in this build, framework entries
// first. Entries flagged skeleton are registered placeholders, not yet
// usable for real partitioning.
std::vector<CapabilityDesc> registeredCapabilities();

// Emits one stage progress line to stderr under the diagnostics contract of
// §10.1; shared by the entry's argument stage and the task stages.
void logStage(llvm::StringRef stage, const llvm::Twine &message);

// Reports a failure in the shared §10.1 format and returns the mapped
// process exit code. The entry owns this call for both argument-stage and
// task failures, so every failure prints identically.
int reportFailure(const TaskFailure &failure);

// One offline partitioning task (§7).
//
// Design commitments:
//  - Frameworks provide Adapter and Exporter as one pair and are selected
//    from the same registry entry, which is what forbids cross-framework
//    mixing (§4.3); only tensorflow is registered so far.
//  - Config files stay raw buffers: the task checks readability only, while
//    schemas and semantics belong to the downstream modules (§5.3).
//  - Stages from model loading onward are placeholders owned by the future
//    modules; each fails with "not implemented", so the pipeline stops at
//    the first one.
//  - Exporters receive only the staging directory of the current run; the
//    final output directory is exclusive to the publish stage (§8/§9).
//  - scratch/staging cleanup is part of the directory-transaction work (§9)
//    and will attach to this object's destruction path.
class PartitionTask {
public:
  explicit PartitionTask(TaskOptions options);

  // Runs the stage ladder and returns the first failure, or nullopt on
  // success; printing the failure and mapping the exit code belong to the
  // entry. Placeholder stages stop the pipeline at model loading, in exit
  // class Analysis. Placeholder errors flatten to strings at this boundary;
  // real modules are expected to return categorized errors so that, e.g.,
  // a disk error during export still classifies as I/O (§10.2).
  std::optional<TaskFailure> run();

private:
  // Loads both config files as raw text; schema parsing belongs to the
  // downstream modules (§5.3).
  std::optional<TaskFailure> loadConfigs();

  // The staging directory of this run: a sibling of the final output
  // directory in its parent (§9). Exporters see this path only; the final
  // output directory stays exclusive to the publish stage.
  // TODO: exclusive creation and the unique isolation suffix (§9).
  std::string stagingDir() const;

  TaskOptions options;
  std::unique_ptr<llvm::MemoryBuffer> partitionConfig;
  std::unique_ptr<llvm::MemoryBuffer> commConfig;
};

}  // namespace annc_partition

#endif  // ANNC_PARTITION_TASK_H
