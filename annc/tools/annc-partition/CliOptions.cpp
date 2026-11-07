// Option management of the annc-partition CLI entry.
//
// The cl::opt table below is the single declarative source of the §5.1
// argument contract: names, value syntax, defaults, and help text are
// declared once here, and parsing, strictness (unknown arguments,
// positional values, single-dash long options), and the option section of
// --help all derive from it. Requiredness is enforced after parsing
// instead of via cl::Required, and a trailing value-taking option is
// rejected up front, because both cl diagnostics bypass the buffered Errs
// stream and would break the shared failure format of §10.1/§10.2.
// Repeated options take the last value (LLVM cl semantics, an accepted
// deviation from the strict duplicate rejection of §5.1). Query modes and
// the assembly of TaskOptions stay in this module so that the entry
// remains a thin stage driver.

#include "CliOptions.h"

#include <algorithm>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace annc_partition {
namespace {

using llvm::cl::alias;
using llvm::cl::opt;

// ---- Option table (cli-design.md §5.1); --help renders in this order ----
// The five required options are listed in kRequiredOptions below.

opt<std::string> Framework(
    "framework",
    llvm::cl::desc("Framework implementation to select (required)"),
    llvm::cl::value_desc("FRAMEWORK"));

opt<std::string> Model("model",
                       llvm::cl::desc("Source model directory (required)"),
                       llvm::cl::value_desc("DIR"));

opt<std::string> Strategy(
    "strategy", llvm::cl::desc("Registered partition strategy (default json)"),
    llvm::cl::value_desc("NAME"), llvm::cl::init("json"));

opt<std::string> PartitionConfig(
    "partition-config",
    llvm::cl::desc("Partition config file, passed through raw (required)"),
    llvm::cl::value_desc("FILE"));

opt<std::string> CommConfig(
    "comm-config",
    llvm::cl::desc("Communication config file, passed through raw (required)"),
    llvm::cl::value_desc("FILE"));

opt<std::string> OutDir(
    "out-dir",
    llvm::cl::desc("Final output directory; must not exist yet (required)"),
    llvm::cl::value_desc("DIR"));

opt<std::string> TfSignature(
    "tf-signature",
    llvm::cl::desc("SavedModel signature for the TensorFlow adapter"
                   " (default serving_default)"),
    llvm::cl::value_desc("NAME"), llvm::cl::init("serving_default"));

opt<std::string> WorkDir(
    "work-dir",
    llvm::cl::desc("Parent directory of the per-run scratch directory"),
    llvm::cl::value_desc("DIR"), llvm::cl::init(""));

opt<bool> KeepTemps(
    "keep-temps",
    llvm::cl::desc("Keep scratch and staging directories after the run"));

opt<bool> Verbose("verbose", llvm::cl::desc("Emit stage progress summaries"));

// Single-letter aliases keep the single-dash spelling under the
// double-dash policy of parseTaskOptions.
alias VerboseShort("v", llvm::cl::desc("Alias for --verbose"),
                   llvm::cl::NotHidden, llvm::cl::aliasopt(Verbose));

// The option list drives --help: adding an option means one declaration
// above and one pointer here.
llvm::cl::Option *const kOptionTable[] = {
    &Framework,  &Model,   &Strategy,     &PartitionConfig,
    &CommConfig, &OutDir,  &TfSignature,  &WorkDir,
    &KeepTemps,  &Verbose, &VerboseShort,
};

void printOptionLines() {
  size_t width = 0;
  for (const auto *option : kOptionTable)
    width = std::max(width, option->getOptionWidth());
  for (const auto *option : kOptionTable) option->printOptionInfo(width);
  // Query modes are served by handleQueryModes rather than the table; keep
  // them listed and aligned with the generated lines. printOptionInfo
  // budgets the two-space lead and the " - " separator inside width, so
  // the name column of a manual line is width - 5.
  const size_t nameWidth = width - 5;
  llvm::outs() << "  " << llvm::left_justify("--help, -h", nameWidth)
               << " - Print this help and exit\n"
               << "  " << llvm::left_justify("--version", nameWidth)
               << " - Print version and registered capabilities\n";
}

void printHelp() {
  llvm::outs() << "Usage: annc-partition --framework FRAMEWORK --model DIR"
                  " --partition-config FILE --comm-config FILE --out-dir DIR"
                  " [options]\n"
                  "\n"
                  "Stages: argument -> config -> adapter -> strategy -> plan ->"
                  " export -> validate -> publish\n"
                  "The pipeline is a skeleton: stages after config are module"
                  " placeholders returning \"not implemented\".\n"
                  "\n"
                  "Options:\n";
  printOptionLines();
  llvm::outs() << "\n"
                  "Registered capabilities:\n";
  for (const auto &cap : registeredCapabilities())
    llvm::outs() << "  " << cap.kind << " " << cap.name
                 << (cap.skeleton ? ": skeleton only" : ": available") << "\n";
}

void printVersion() {
  llvm::outs() << "annc-partition " << ANNC_PARTITION_VERSION
               << "\nsource revision: " << ANNC_SOURCE_REVISION << "\n";
  for (const auto &cap : registeredCapabilities())
    llvm::outs() << cap.kind << " " << cap.name
                 << (cap.skeleton ? ": skeleton only" : ": available") << "\n";
}

// Collapses the parser's per-line diagnostics into one message for the
// shared failure format; each cl line carries a "progname: " prefix that
// the stage and code tags of that format already convey.
std::string summarizeDiagnostics(const std::string &diagnostics,
                                 const char *argv0) {
  const std::string prefix = llvm::sys::path::filename(argv0).str() + ": ";
  llvm::SmallVector<llvm::StringRef, 4> lines;
  llvm::StringRef(diagnostics).split(lines, '\n');
  std::string message;
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (line.empty()) continue;
    line.consume_front(prefix);
    if (!message.empty()) message += "; ";
    message += line.str();
  }
  if (message.empty()) message = "invalid arguments";
  return message;
}

}  // namespace

std::optional<int> handleQueryModes(int argc, char *const *argv) {
  for (int i = 1; i < argc; ++i) {
    const llvm::StringRef arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printHelp();
      return static_cast<int>(ExitCode::Success);
    }
    if (arg == "--version") {
      printVersion();
      return static_cast<int>(ExitCode::Success);
    }
  }
  return std::nullopt;
}

llvm::Expected<TaskOptions> parseTaskOptions(int argc, char *const *argv) {
  // A value-taking option as the last argument makes the parser print its
  // "requires a value" diagnostic straight to stderr, bypassing the
  // buffered Errs stream; reject that spelling here so the failure keeps
  // the shared format. Tokens after a bare "--" are positional, not
  // options, and are left to the parser.
  const llvm::StringRef valueOptions[] = {
      "--framework",   "--model",   "--strategy",     "--partition-config",
      "--comm-config", "--out-dir", "--tf-signature", "--work-dir"};
  bool separatorSeen = false;
  for (int i = 1; i < argc; ++i) {
    const llvm::StringRef arg = argv[i];
    if (arg == "--") separatorSeen = true;
    if (i == argc - 1 && !separatorSeen &&
        std::find(std::begin(valueOptions), std::end(valueOptions), arg) !=
            std::end(valueOptions))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "missing value for " + arg.str());
  }

  // Long options must use the double-dash spelling; single-dash long names
  // and positional values are unknown arguments (§5.1). Diagnostics are
  // buffered so that a query run exiting inside the parser stays silent on
  // stderr, and argument failures reach the shared failure format with exit
  // class 2 (§10.2) instead of cl's own exit code.
  std::string diagnostics;
  llvm::raw_string_ostream errs(diagnostics);
  const bool parsed = llvm::cl::ParseCommandLineOptions(
      argc, argv, "ANNC heterogeneous graph partitioner", &errs, nullptr,
      /*LongOptionsUseDoubleDash=*/true);
  errs.flush();
  if (!parsed)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   summarizeDiagnostics(diagnostics, argv[0]));

  TaskOptions options;
  options.framework = Framework;
  options.model = Model;
  options.strategy = Strategy;
  options.partitionConfig = PartitionConfig;
  options.commConfig = CommConfig;
  options.outDir = OutDir;
  options.tfSignature = TfSignature;
  options.workDir = WorkDir;
  options.keepTemps = KeepTemps;
  options.verbose = Verbose;

  // Required options of §5.1, in contract-table order. An explicitly empty
  // value is reported separately from a missing option; values of optional
  // options are left to later stages.
  const std::pair<const char *, const opt<std::string> *> kRequiredOptions[] = {
      {"--framework", &Framework},
      {"--model", &Model},
      {"--partition-config", &PartitionConfig},
      {"--comm-config", &CommConfig},
      {"--out-dir", &OutDir}};
  for (const auto &[name, option] : kRequiredOptions) {
    if (option->getNumOccurrences() == 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          std::string("missing required option: ") + name);
    if (option->getValue().empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     std::string("empty value for ") + name);
  }

  return options;
}

}  // namespace annc_partition
