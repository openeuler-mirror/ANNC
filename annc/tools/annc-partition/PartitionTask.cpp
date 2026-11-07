// Stage orchestration of one annc-partition task (cli-design.md §7).
//
// The placeholder section below belongs to the future modules: when they
// land, it moves out wholesale and only the call sites in run() change,
// while the stage ladder stays the stable contract.

#include "PartitionTask.h"

#include <functional>
#include <utility>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace annc_partition {

namespace {

// ---- Placeholders for the future module implementations ----
// The empty types and stub classes below pin only the stage handoffs and
// framework/strategy selection, not any module-internal design. Real
// implementations live with their modules (overview.md §3: the TF
// Adapter/Exporter under frontends/tensorflow/planner/, the JSON-pattern
// strategy and the shared planner in the common Planner library) and
// replace this section together with the call sites in run(). Stubs log
// the handoff data they receive -- so the real implementations keep the
// log lines in place -- and then fail with "not implemented".

struct ModelHandle {};    // Immutable source model handed on by the adapter.
struct CandidateSet {};   // Candidate regions produced by the strategy.
struct PartitionPlan {};  // Split plan produced by the shared planner.
struct ExportResult {};   // Export artifacts produced by the exporter.

// Framework plug points: orchestration depends only on these interfaces.
// Each framework provides Adapter and Exporter as one pair selected from
// the same registry entry, which is what forbids cross-framework mixing
// (§4.3).
class ModelAdapter {
public:
  virtual ~ModelAdapter() = default;
  virtual llvm::Expected<ModelHandle> load(const TaskOptions &opts) = 0;
};

class ModelExporter {
public:
  virtual ~ModelExporter() = default;
  // Exporters write only into the staging directory of the current run and
  // never see the final output directory, which stays exclusive to the
  // publish stage (§8/§9).
  virtual llvm::Expected<ExportResult>
  exportModel(const ModelHandle &, const PartitionPlan &,
              const llvm::MemoryBuffer &commConfig,
              llvm::StringRef stagingDir) = 0;
};

// TODO: the real TF adapter reads the SavedModel and checks the initial
// profile (overview.md §3.2).
class TensorFlowAdapter : public ModelAdapter {
public:
  llvm::Expected<ModelHandle> load(const TaskOptions &opts) override {
    logStage("adapter", llvm::formatv("loading SavedModel from {0}"
                                      " (signature {1})",
                                      opts.model, opts.tfSignature));
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "TensorFlowAdapter::load is not"
                                   " implemented");
  }
};

// The template fixes the strategy handoff without prescribing analysis
// inputs: JSON patterns, heuristics, and cost-guided strategies prepare
// different inputs but all produce the same source-mapped candidates.
class PartitionStrategy {
public:
  virtual ~PartitionStrategy() = default;

  llvm::Expected<CandidateSet>
  generateCandidates(const ModelHandle &model,
                     const llvm::MemoryBuffer &partitionConfig) {
    if (llvm::Error e = prepareAnalysis(model, partitionConfig))
      return std::move(e);
    auto candidates = collectCandidates(model, partitionConfig);
    if (!candidates)
      return candidates.takeError();
    if (llvm::Error e = validateCandidates(model, *candidates))
      return std::move(e);
    return candidates;
  }

protected:
  virtual llvm::Error
  prepareAnalysis(const ModelHandle &model,
                  const llvm::MemoryBuffer &partitionConfig) = 0;
  virtual llvm::Expected<CandidateSet>
  collectCandidates(const ModelHandle &model,
                    const llvm::MemoryBuffer &partitionConfig) = 0;
  virtual llvm::Error
  validateCandidates(const ModelHandle &model,
                     const CandidateSet &candidates) = 0;
};

// TODO: implement the template steps with the extracted match-only matcher
// and Adapter source mapping (overview.md §3.3).
class JsonPatternStrategy final : public PartitionStrategy {
protected:
  llvm::Error prepareAnalysis(const ModelHandle &,
                              const llvm::MemoryBuffer &) override {
    return llvm::Error::success();
  }

  llvm::Expected<CandidateSet>
  collectCandidates(const ModelHandle &,
                    const llvm::MemoryBuffer &partitionConfig) override {
    logStage("strategy",
             llvm::formatv("matching JSON patterns from partition config"
                           " ({0} bytes)",
                           partitionConfig.getBufferSize()));
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "candidate generation is not"
                                   " implemented");
  }

  llvm::Error validateCandidates(const ModelHandle &,
                                 const CandidateSet &) override {
    return llvm::Error::success();
  }
};

// TODO: the shared planner enforces ownership, legality, and boundary
// checks (overview.md §3.4).
class PartitionPlanner {
public:
  llvm::Expected<PartitionPlan>
  createPlan(const ModelHandle &, const CandidateSet &,
             const llvm::MemoryBuffer &partitionConfig,
             const llvm::MemoryBuffer &commConfig) {
    logStage("plan",
             llvm::formatv("planning A/B split (partition config {0} bytes,"
                           " comm config {1} bytes)",
                           partitionConfig.getBufferSize(),
                           commConfig.getBufferSize()));
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "partition planning is not implemented");
  }
};

// TODO: the real TF exporter writes A/B and the manifest to staging
// (overview.md §3.5; package layout in §8.1).
class TensorFlowExporter : public ModelExporter {
public:
  llvm::Expected<ExportResult>
  exportModel(const ModelHandle &, const PartitionPlan &,
              const llvm::MemoryBuffer &,
              llvm::StringRef stagingDir) override {
    logStage("export",
             llvm::formatv("exporting A/B and manifest to staging {0}",
                           stagingDir));
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "TensorFlowExporter::exportModel is"
                                   " not implemented");
  }
};

// The Adapter and Exporter of one framework, selected as a unit.
struct FrameworkImpl {
  std::unique_ptr<ModelAdapter> adapter;
  std::unique_ptr<ModelExporter> exporter;
};

// The fixed build-time capability registry (§4.3) in its minimal form: the
// single source that drives selection, --help, and --version. Entries
// flagged skeleton are registered placeholders, not yet functional; an
// unregistered name is a capability failure (exit 3, §10.2), not an
// argument error.
struct FrameworkEntry {
  llvm::StringRef name;
  bool skeleton;
  std::function<std::unique_ptr<ModelAdapter>()> makeAdapter;
  std::function<std::unique_ptr<ModelExporter>()> makeExporter;
};

struct StrategyEntry {
  llvm::StringRef name;
  bool skeleton;
  std::function<std::unique_ptr<PartitionStrategy>()> makeStrategy;
};

const FrameworkEntry kFrameworks[] = {
    {"tensorflow", true,
     [] { return std::make_unique<TensorFlowAdapter>(); },
     [] { return std::make_unique<TensorFlowExporter>(); }},
};

const StrategyEntry kStrategies[] = {
    {"json", true, [] { return std::make_unique<JsonPatternStrategy>(); }},
};

template <typename Entries> std::string joinedNames(const Entries &entries) {
  std::string out;
  for (const auto &entry : entries) {
    if (!out.empty())
      out += ", ";
    out += entry.name.str();
  }
  return out;
}

llvm::Expected<std::unique_ptr<PartitionStrategy>>
createStrategy(const TaskOptions &opts) {
  for (const auto &entry : kStrategies)
    if (entry.name == opts.strategy)
      return entry.makeStrategy();
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "no registered partition strategy for '" + opts.strategy +
          "' (registered: " + joinedNames(kStrategies) + ")");
}

llvm::Expected<FrameworkImpl> createFrameworkImpl(const TaskOptions &opts) {
  for (const auto &entry : kFrameworks)
    if (entry.name == opts.framework)
      return FrameworkImpl{entry.makeAdapter(), entry.makeExporter()};
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "no registered framework implementation for '" + opts.framework +
          "' (registered: " + joinedNames(kFrameworks) + ")");
}

}  // namespace

std::vector<CapabilityDesc> registeredCapabilities() {
  std::vector<CapabilityDesc> out;
  for (const auto &entry : kFrameworks)
    out.push_back({"framework", entry.name, entry.skeleton});
  for (const auto &entry : kStrategies)
    out.push_back({"strategy", entry.name, entry.skeleton});
  return out;
}

PartitionTask::PartitionTask(TaskOptions taskOptions)
    : options(std::move(taskOptions)) {}

void logStage(llvm::StringRef stage, const llvm::Twine &message) {
  llvm::errs() << "[annc-partition][" << stage << "] " << message << "\n";
}

int reportFailure(const TaskFailure &failure) {
  llvm::errs() << "[annc-partition][" << failure.stage << "]["
               << failure.code << "] " << failure.message << "\n";
  return static_cast<int>(failure.exitCode);
}

std::optional<TaskFailure> PartitionTask::loadConfigs() {
  // TODO: strict schema parsing per §5.3/§5.4; schema violations then map
  // to exit class 2, while unreadable files stay in class 4 (§10.2).
  auto partitionFile = llvm::MemoryBuffer::getFile(options.partitionConfig);
  if (!partitionFile)
    return TaskFailure{"config", "E_CONFIG_UNREADABLE",
                       partitionFile.getError().message() + ": " +
                           options.partitionConfig,
                       ExitCode::Io};
  partitionConfig = std::move(partitionFile.get());
  logStage("config", llvm::formatv("loaded partition config {0} ({1} bytes)",
                                   options.partitionConfig,
                                   partitionConfig->getBufferSize()));
  auto commFile = llvm::MemoryBuffer::getFile(options.commConfig);
  if (!commFile)
    return TaskFailure{"config", "E_CONFIG_UNREADABLE",
                       commFile.getError().message() + ": " +
                           options.commConfig,
                       ExitCode::Io};
  commConfig = std::move(commFile.get());
  logStage("config", llvm::formatv("loaded comm config {0} ({1} bytes)",
                                   options.commConfig,
                                   commConfig->getBufferSize()));
  return std::nullopt;
}

std::string PartitionTask::stagingDir() const {
  // Staging is a sibling of the final output directory in its parent (§9);
  // exclusive creation and the unique isolation suffix belong to the
  // directory transaction.
  llvm::SmallString<128> parent(options.outDir);
  std::string name(llvm::sys::path::filename(parent));
  llvm::sys::path::remove_filename(parent);
  llvm::sys::path::append(parent, "." + name + ".annc-staging");
  return std::string(parent);
}

std::optional<TaskFailure> PartitionTask::run() {
  if (auto failure = loadConfigs())
    return failure;

  auto frameworkImpl = createFrameworkImpl(options);
  if (!frameworkImpl)
    return TaskFailure{"adapter", "E_CAPABILITY_UNAVAILABLE",
                       llvm::toString(frameworkImpl.takeError()),
                       ExitCode::Capability};
  auto strategy = createStrategy(options);
  if (!strategy)
    return TaskFailure{"strategy", "E_CAPABILITY_UNAVAILABLE",
                       llvm::toString(strategy.takeError()),
                       ExitCode::Capability};

  auto model = frameworkImpl->adapter->load(options);
  if (!model)
    return TaskFailure{"adapter", "E_ANALYSIS",
                       llvm::toString(model.takeError()),
                       ExitCode::Analysis};

  auto candidates = (*strategy)->generateCandidates(*model, *partitionConfig);
  if (!candidates)
    return TaskFailure{"strategy", "E_ANALYSIS",
                       llvm::toString(candidates.takeError()),
                       ExitCode::Analysis};

  PartitionPlanner planner;
  auto plan = planner.createPlan(*model, *candidates, *partitionConfig,
                                 *commConfig);
  if (!plan)
    return TaskFailure{"plan", "E_PLAN", llvm::toString(plan.takeError()),
                       ExitCode::Plan};

  auto exportResult = frameworkImpl->exporter->exportModel(
      *model, *plan, *commConfig, stagingDir());
  if (!exportResult)
    return TaskFailure{"export", "E_EXPORT",
                       llvm::toString(exportResult.takeError()),
                       ExitCode::Export};

  // The stages after export -- artifact validation, report assembly, and
  // atomic publish (§8/§9) -- have no code yet and are unreachable until
  // the placeholders above become real.
  // TODO: validation summary and atomic publish.
  return std::nullopt;
}

}  // namespace annc_partition
