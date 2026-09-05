#include <unistd.h>

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Dialect/Atir/AtirOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

namespace fs = std::filesystem;
using namespace mlir;

namespace {

struct PipelineOptions {
  std::string inputGraphDef;
  std::string outputGraphDef;
  std::string kernelName;
  std::string sharedLibPath;
  std::string workDir;
  std::vector<std::string> outputTensors;
  int64_t batchSize = -1;
  int64_t intraThreadCount = -1;
  bool keepTemps = false;
  bool dumpFusionMetadata = false;
  bool verbose = false;
};

static void printUsage() {
  std::cout
      << "Usage: annc-tf-pipeline --input_graphdef <path> "
         "--output_graphdef <path> [options]\n"
         "Options:\n"
         "  --work_dir <dir>          Intermediate artifact directory\n"
         "  --shared_lib_path <path>  AOT library path written to GraphDef\n"
         "  --batch_size <n>          Override dynamic batch dimensions\n"
         "  --intra_thread_count <n>  Set module-level GEMM thread count\n"
         "  --output_tensor <name>    Preserve a named graph output\n"
         "  --keep_temps              Keep intermediate artifacts\n"
         "  --dump-fusion-metadata    Write fusion_metadata.json\n"
         "  --verbose                 Print invoked commands\n";
}

static bool hasArg(int argc, char **argv, const std::string &name) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == name) return true;
  }
  return false;
}

static std::string takeValue(int argc, char **argv, const std::string &name,
                             const std::string &defaultValue = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) return argv[i + 1];
  }
  return defaultValue;
}

static std::string shellQuote(const std::string &value) {
  std::string out = "'";
  for (char c : value) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  out += "'";
  return out;
}

static bool runCommand(const std::vector<std::string> &args, bool verbose) {
  std::ostringstream os;
  for (size_t i = 0; i < args.size(); ++i) {
    if (i) os << ' ';
    os << shellQuote(args[i]);
  }
  std::string command = os.str();
  if (verbose) std::cerr << "[annc-tf-pipeline] " << command << "\n";
  int ret = std::system(command.c_str());
  if (ret != 0) {
    std::cerr << "[annc-tf-pipeline] command failed: " << command
              << " exit=" << ret << "\n";
    return false;
  }
  return true;
}

static bool containsAotKernel(const fs::path &path, bool &hasAotKernel) {
  DialectRegistry registry;
  registry.insert<func::FuncDialect, LLVM::LLVMDialect, atir::AtirDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = parseSourceFile<ModuleOp>(path.string(), &context);
  if (!module) {
    std::cerr << "[annc-tf-pipeline] failed to parse AOT ATIR: " << path
              << "\n";
    return false;
  }

  hasAotKernel = false;
  module->walk([&](func::FuncOp function) {
    if (!function->hasAttr("annc.kernel")) return;
    auto mode = function->getAttrOfType<StringAttr>("annc.execution_mode");
    if (mode && mode.getValue() == "aot") hasAotKernel = true;
  });
  return true;
}

static std::string executableSibling(const char *argv0, const std::string &name) {
  fs::path self(argv0);
  if (self.has_parent_path()) return (self.parent_path() / name).string();
  return name;
}

static std::string defaultWorkDir() {
  fs::path base = fs::temp_directory_path();
  return (base / ("annc_tf_rewrite_" + std::to_string(std::time(nullptr)) +
                  "_" + std::to_string(getpid())))
      .string();
}

static bool parsePipelineOptions(int argc, char **argv, PipelineOptions *opts) {
  opts->inputGraphDef = takeValue(argc, argv, "--input_graphdef");
  opts->outputGraphDef = takeValue(argc, argv, "--output_graphdef");
  opts->kernelName = takeValue(argc, argv, "--kernel_name", opts->kernelName);
  opts->sharedLibPath = takeValue(argc, argv, "--shared_lib_path");
  opts->workDir = takeValue(argc, argv, "--work_dir", defaultWorkDir());
  opts->batchSize = std::stoll(takeValue(argc, argv, "--batch_size", "-1"));
  opts->intraThreadCount =
      std::stoll(takeValue(argc, argv, "--intra_thread_count", "-1"));
  opts->keepTemps = hasArg(argc, argv, "--keep_temps") ||
                    hasArg(argc, argv, "--keep_temp_files");
  opts->dumpFusionMetadata = hasArg(argc, argv, "--dump-fusion-metadata");
  opts->verbose = hasArg(argc, argv, "--verbose") || hasArg(argc, argv, "-v");

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) != "--output_tensor") continue;
    if (i + 1 >= argc || std::string(argv[i + 1]).empty() ||
        std::string(argv[i + 1]).front() == '-') {
      std::cerr << "[annc-tf-pipeline] --output_tensor requires a non-empty "
                   "value\n";
      return false;
    }
    opts->outputTensors.emplace_back(argv[i + 1]);
    ++i;
  }

  if (opts->inputGraphDef.empty() || opts->outputGraphDef.empty()) {
    std::cerr << "[annc-tf-pipeline] --input_graphdef and --output_graphdef "
              << "are required\n";
    return false;
  }
  if (opts->intraThreadCount == 0 || opts->intraThreadCount < -1) {
    std::cerr << "[annc-tf-pipeline] --intra_thread_count must be positive\n";
    return false;
  }
  return true;
}

static bool runGraphDefRewrite(int argc, char **argv) {
  PipelineOptions opts;
  if (!parsePipelineOptions(argc, argv, &opts)) return false;

  fs::create_directories(opts.workDir);
  fs::path work(opts.workDir);
  fs::path rawAtir = work / "model_raw_atir.mlir";
  fs::path fusedAtir = work / "model_fused_atir.mlir";
  fs::path aotAtir = work / "model_aot_atir.mlir";
  fs::path fusionMetadata = work / "fusion_metadata.json";
  fs::path loweredMlir = work / "model_lowered.mlir";
  fs::path packedC = work / "model_packed_rhs.c";
  fs::path generatedSo = fs::absolute(work / "annc_generated_kernel.so");
  std::string runtimeSharedLibPath =
      opts.sharedLibPath.empty() ? generatedSo.string() : opts.sharedLibPath;

  std::string anncTf2Atir = executableSibling(argv[0], "annc-tf2atir");
  std::string anncOpt = executableSibling(argv[0], "annc-opt");
  std::string anncFusionMetadata =
      executableSibling(argv[0], "annc-fusion-metadata");
  std::string anncAsm = executableSibling(argv[0], "annc-asm");
  std::string annc = executableSibling(argv[0], "annc");
  std::string anncConverter = executableSibling(argv[0], "annc-converter");

  std::vector<std::string> converterArgs = {
      anncConverter, fusedAtir.string(), "--tf-graphdef-rewrite",
      "--input_graphdef", opts.inputGraphDef, "--output_graphdef",
      opts.outputGraphDef,
  };
  converterArgs.push_back("--shared_lib_path");
  converterArgs.push_back(runtimeSharedLibPath);
  converterArgs.push_back("--atir_module_path");
  converterArgs.push_back(fs::absolute(fusedAtir).string());
  if (!opts.kernelName.empty()) {
    converterArgs.push_back("--kernel_name");
    converterArgs.push_back(opts.kernelName);
  }

  std::string foldBatchNormPass = "--atir-fold-batch-norm";
  std::string rankInferencePass = "--atir-rank-inference";
  std::string identityCanonicalizePass = "--atir-identity-canonicalize";
  std::string fusionPass = "--atir-op-fusion";
  std::string fastCodegenPass = "--atir-fast-codegen";
#ifdef ANNC_ENABLE_KDNN_ADAPTOR
  fastCodegenPass = "--atir-fast-codegen=enable-kdnn=true";
#endif

  std::vector<std::string> tf2atirArgs = {
      anncTf2Atir, opts.inputGraphDef, "-o", rawAtir.string()};
  if (opts.batchSize > 0) {
    tf2atirArgs.insert(tf2atirArgs.begin() + 2, "--batch_size");
    tf2atirArgs.insert(tf2atirArgs.begin() + 3,
                       std::to_string(opts.batchSize));
  }
  if (opts.intraThreadCount > 0) {
    tf2atirArgs.push_back("--intra_thread_count");
    tf2atirArgs.push_back(std::to_string(opts.intraThreadCount));
  }
  for (const std::string &tensor : opts.outputTensors) {
    tf2atirArgs.push_back("--output_tensor");
    tf2atirArgs.push_back(tensor);
  }

  bool ok =
      runCommand(tf2atirArgs, opts.verbose) &&
      runCommand({anncOpt, rawAtir.string(), rankInferencePass,
                  identityCanonicalizePass, fusionPass, "-o",
                  fusedAtir.string()},
                 opts.verbose) &&
      runCommand({anncOpt, fusedAtir.string(),
                  "--atir-prune-func=execution-mode=aot", "-o",
                  aotAtir.string()},
                 opts.verbose);
  if (ok && opts.dumpFusionMetadata) {
    ok = runCommand({anncFusionMetadata, fusedAtir.string(), "-o",
                     fusionMetadata.string()},
                    opts.verbose);
  }
  bool hasAot = false;
  if (ok) ok = containsAotKernel(aotAtir, hasAot);
  if (hasAot) {
    std::vector<std::string> asmArgs = {
        anncAsm, aotAtir.string(), fastCodegenPass,
        "--annc-aarch64-gemm-pipeline", "-o", loweredMlir.string()};
    std::vector<std::string> linkArgs = {
        annc, loweredMlir.string(), "--shared", "-o", generatedSo.string()};
#ifdef ANNC_ENABLE_CONSTANT_FOLDING
    asmArgs[3] = "--annc-aarch64-gemm-pipeline=packed-c=" + packedC.string();
    linkArgs.push_back("--packed-rhs-c");
    linkArgs.push_back(packedC.string());
#endif
    ok = ok && runCommand(asmArgs, opts.verbose) &&
         runCommand(linkArgs, opts.verbose);
  }
  ok = ok && runCommand(converterArgs, opts.verbose);

  if (!ok) return false;

  if (opts.verbose) {
    if (hasAot) {
      std::cerr << "[annc-tf-pipeline] generated compiler kernel artifact: "
                << generatedSo << "\n";
    } else {
      std::cerr << "[annc-tf-pipeline] no AOT fusion functions; shared library "
                   "generation skipped\n";
    }
    std::cerr
        << "[annc-tf-pipeline] runtime fusion template written to GraphDef: "
        << fs::absolute(fusedAtir) << "\n";
    if (opts.dumpFusionMetadata) {
      std::cerr << "[annc-tf-pipeline] fusion metadata dumped to: "
                << fusionMetadata << "\n";
    }
  }

  if (!opts.keepTemps) {
    std::error_code ec;
    // The generated shared library and full fusion template are referenced by
    // the rewritten GraphDef, so retain them for runtime execution.
    for (const fs::path &temp : {rawAtir, aotAtir, loweredMlir, packedC}) {
      fs::remove(temp, ec);
      ec.clear();
    }
    if (!opts.dumpFusionMetadata) {
      fs::remove(fusionMetadata, ec);
      ec.clear();
    }
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (hasArg(argc, argv, "--help") || hasArg(argc, argv, "-h")) {
    printUsage();
    return 0;
  }
  if (hasArg(argc, argv, "--tf-graphdef-rewrite")) {
    std::cerr << "[annc-tf-pipeline] warning: --tf-graphdef-rewrite is "
              << "deprecated on pipeline entry and will be ignored\n";
  }
  return runGraphDefRewrite(argc, argv) ? 0 : 1;
}
