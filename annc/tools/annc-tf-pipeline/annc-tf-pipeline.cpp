#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct PipelineOptions {
  std::string inputGraphDef;
  std::string outputGraphDef;
  std::string kernelName;
  std::string sharedLibPath;
  std::string workDir;
  int64_t batchSize = 2;
  bool keepTemps = false;
  bool verbose = false;
};

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

static bool envFlagEnabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value) return false;
  std::string flag(value);
  return flag == "1" || flag == "true" || flag == "TRUE" ||
         flag == "on" || flag == "ON" || flag == "yes" || flag == "YES";
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

static std::string executableSibling(const char *argv0, const std::string &name) {
  fs::path self(argv0);
  if (self.has_parent_path()) return (self.parent_path() / name).string();
  return name;
}

static std::string defaultWorkDir() {
  fs::path base = fs::temp_directory_path();
  return (base / ("annc_tf_rewrite_" + std::to_string(std::time(nullptr)))).string();
}

static bool parsePipelineOptions(int argc, char **argv, PipelineOptions *opts) {
  opts->inputGraphDef = takeValue(argc, argv, "--input_graphdef");
  opts->outputGraphDef = takeValue(argc, argv, "--output_graphdef");
  opts->kernelName = takeValue(argc, argv, "--kernel_name", opts->kernelName);
  opts->sharedLibPath = takeValue(argc, argv, "--shared_lib_path");
  opts->workDir = takeValue(argc, argv, "--work_dir", defaultWorkDir());
  opts->batchSize = std::stoll(takeValue(argc, argv, "--batch_size", "2"));
  opts->keepTemps = hasArg(argc, argv, "--keep_temps") ||
                    hasArg(argc, argv, "--keep_temp_files");
  opts->verbose = hasArg(argc, argv, "--verbose") || hasArg(argc, argv, "-v");

  if (opts->inputGraphDef.empty() || opts->outputGraphDef.empty()) {
    std::cerr << "[annc-tf-pipeline] --input_graphdef and --output_graphdef "
              << "are required\n";
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
  fs::path fusionMetadata = work / "fusion_metadata.json";
  fs::path loweredMlir = work / "model_lowered.mlir";
  fs::path generatedSo = fs::absolute(work / "annc_generated_kernel.so");
  std::string runtimeSharedLibPath =
      opts.sharedLibPath.empty() ? generatedSo.string() : opts.sharedLibPath;

  std::string anncTf2Atir = executableSibling(argv[0], "annc-tf2atir");
  std::string anncOpt = executableSibling(argv[0], "annc-opt");
  std::string anncFusionMetadata = executableSibling(argv[0], "annc-fusion-metadata");
  std::string anncAsm = executableSibling(argv[0], "annc-asm");
  std::string annc = executableSibling(argv[0], "annc");
  std::string anncConverter = executableSibling(argv[0], "annc-converter");

  std::vector<std::string> converterArgs = {
      anncConverter, fusedAtir.string(), "--tf-graphdef-rewrite",
      "--input_graphdef", opts.inputGraphDef, "--output_graphdef",
      opts.outputGraphDef, "--shared_lib_path", runtimeSharedLibPath,
      "--metadata_json", fusionMetadata.string()};
  if (!opts.kernelName.empty()) {
    converterArgs.push_back("--kernel_name");
    converterArgs.push_back(opts.kernelName);
  }

  std::string fusionPass = "--atir-op-fusion";

  std::vector<std::string> asmArgs = {
      anncAsm, fusedAtir.string(), "--atir-prune-func",
      "--atir-fast-codegen",
      "--convert-atir-to-affine", "-o", loweredMlir.string()};
#ifdef ANNC_ENABLE_KDNN_ADAPTOR
  asmArgs[3] = "--atir-fast-codegen=enable-kdnn=true";
#endif

  bool ok =
      runCommand({anncTf2Atir, opts.inputGraphDef, "--batch_size",
                  std::to_string(opts.batchSize), "-o", rawAtir.string()},
                 opts.verbose) &&
      runCommand({anncOpt, rawAtir.string(), fusionPass, "-o",
                  fusedAtir.string()},
                 opts.verbose) &&
      runCommand({anncFusionMetadata, fusedAtir.string(), "-o",
                  fusionMetadata.string()},
                 opts.verbose) &&
      runCommand(asmArgs, opts.verbose) &&
      runCommand({annc, loweredMlir.string(), "--shared", "-o",
                  generatedSo.string()},
                 opts.verbose) &&
      runCommand(converterArgs, opts.verbose);

  if (!ok) return false;

  if (opts.verbose) {
    std::cerr << "[annc-tf-pipeline] generated compiler kernel artifact: "
              << generatedSo << "\n";
    std::cerr << "[annc-tf-pipeline] runtime shared_lib_path written to GraphDef: "
              << runtimeSharedLibPath << "\n";
  }

  if (!opts.keepTemps) {
    std::error_code ec;
    fs::remove_all(work, ec);
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (hasArg(argc, argv, "--tf-graphdef-rewrite")) {
    std::cerr << "[annc-tf-pipeline] warning: --tf-graphdef-rewrite is "
              << "deprecated on pipeline entry and will be ignored\n";
  }
  return runGraphDefRewrite(argc, argv) ? 0 : 1;
}
