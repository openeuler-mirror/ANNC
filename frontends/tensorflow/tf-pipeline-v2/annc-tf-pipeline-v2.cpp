// annc-tf-pipeline-v2: offline orchestrator for the TF stablehlo V2 route
// (entered with ANNC_ENABLE_TF2ATIR_V2). Pure process orchestration - no
// MLIR library is linked, so the tool stays out of the LLVM/MLIR link
// closure (mirrors frontends/tensorflow/tf-pipeline).
//
// Chain (the end-to-end reviewed wd_dcn pipeline):
//   pb2stablehlo -> annc-stablehlo-sink -> annc-asm(-atir-op-fusion-v2)
//     -> annc-asm(--annc-fast-codegen) -> annc-fusion-metadata
//     -> annc-asm(prune+codegen+convert-atir-to-linalg
//                 +convert-atir-v2-to-linalg)
//     -> [AOT] annc -shared -> annc-converter-v2
//
// Step 8 is optional: --skip-converter leaves the GraphDef rewrite to the
// in-process plugin (predictor deployments; the rewritten.pb artifact is
// for manual inspection only and never feeds the runtime).
//
// Logging goes through the header-only Support/Log.h (no LLVM symbol in
// the link closure; only the annc/include header path is added). Errors
// log at ANNC_LOG_ERROR and stay visible at the default level; the
// --verbose switch maps onto ANNC_LOG_LEVEL=info unless the variable is
// already set, so the orchestrator obeys the one-knob logging rule.
// Step 7 sets ANNC_ENABLE_LOWERING_CONTEXT_ABI for the child: the V2
// route's context kernels lower through that path, which the default
// build guards off (see Aarch64CodegenPipelineBuild).
// Step 6 deliberately lowers the non-kernel V2 carriers ahead of time;
// annc -shared appends its own ConvertContextKernelABI +
// ConvertAtirV2ToLinalg for the kernel funcs themselves.
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "Support/Log.h"
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
  std::string kernelLibPath;
  std::string workDir;
  std::vector<std::string> inputs;
  std::vector<std::string> outputTensors;
  bool skipPreprocess = true;
  bool assumeOpaqueI8 = true;
  // Plugin-only deployments: the in-process rewriter does the GraphDef
  // rewrite inside the predictor, so the offline converter step is optional.
  bool skipConverter = false;
  bool keepTemps = false;
  bool verbose = false;
};

// The customize artifact carries the kernel funcs; an AOT site shows up as
// annc.kernel together with execution_mode = "aot". Text-based on purpose:
// the orchestrator deliberately links no MLIR library. Residual risk: the
// two markers can sit on different functions, and the converter writes the
// declared paths into GraphDef without checking file existence.
bool containsAotKernel(const fs::path &path, bool &hasAotKernel) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    ANNC_LOG_ERROR("tf-pipeline-v2")
        << "cannot read customize artifact: " << path << "\n";
    return false;
  }
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  hasAotKernel = text.find("annc.kernel") != std::string::npos &&
                 text.find("execution_mode = \"aot\"") != std::string::npos;
  return true;
}

// Usage goes to stdout by CLI convention (--help must stay capturable and
// visible regardless of ANNC_LOG_LEVEL); Log.h deliberately never writes
// stdout since that channel carries MLIR output.
void printUsage() {
  std::cout
      << "Usage: annc-tf-pipeline-v2 --input_graphdef <path> "
         "--output_graphdef <path> [options]\n"
         "Options:\n"
         "  --work_dir <dir>          Intermediate artifact directory\n"
         "  --input <name>            Graph entry tensor name (repeatable)\n"
         "  --output_tensor <name>    Graph fetch tensor name (repeatable)\n"
         "  --shared_lib_path <path>  AOT library path written to GraphDef\n"
         "  --kernel_name <name>      Restrict the rewrite to one kernel\n"
         "  --kernel_lib_path <dir>   GEMM runtime libs for annc -shared\n"
         "                            (default: env ANNC_KERNEL_LIB_PATH)\n"
         "  --no-skip-preprocess      Run the official preprocess step\n"
         "  --skip-converter          Skip the offline GraphDef rewrite; the\n"
         "                            in-process plugin rewrites inside the\n"
         "                            predictor (no rewritten.pb output)\n"
         "  --no-assume-opaque-i8     Stop marking dtype-less slots DT_INT8\n"
         "  --keep_temps              Keep intermediate artifacts\n"
         "  --verbose                 Print invoked commands\n"
         "                            (maps onto ANNC_LOG_LEVEL=info)\n"
         "  GEMM fusion is gated by ANNC_GEMM (default 0 = off: no dot\n"
         "  kernel is produced); export it in the calling environment.\n";
}

bool hasArg(int argc, char **argv, const std::string &name) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == name) return true;
  }
  return false;
}

std::string takeValue(int argc, char **argv, const std::string &name,
                      const std::string &defaultValue = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) return argv[i + 1];
  }
  return defaultValue;
}

bool collectRepeated(int argc, char **argv, const std::string &name,
                     std::vector<std::string> *out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] != name) continue;
    std::string value = argv[i + 1];
    if (value.empty() || value.front() == '-') {
      ANNC_LOG_ERROR("tf-pipeline-v2")
          << name << " requires a non-empty value\n";
      return false;
    }
    out->push_back(std::move(value));
  }
  return true;
}

std::string shellQuote(const std::string &value) {
  std::string out = "'";
  for (char c : value) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  out += "'";
  return out;
}

bool runCommand(const std::vector<std::string> &args) {
  std::ostringstream os;
  for (size_t i = 0; i < args.size(); ++i) {
    if (i) os << ' ';
    os << shellQuote(args[i]);
  }
  std::string command = os.str();
  ANNC_LOG_INFO("tf-pipeline-v2") << command << "\n";
  int ret = std::system(command.c_str());
  if (ret != 0) {
    if (WIFSIGNALED(ret)) {
      ANNC_LOG_ERROR("tf-pipeline-v2")
          << "command killed by signal " << WTERMSIG(ret) << ": " << command
          << "\n";
    } else {
      ANNC_LOG_ERROR("tf-pipeline-v2")
          << "command failed: " << command << " exit=" << WEXITSTATUS(ret)
          << "\n";
    }
    return false;
  }
  return true;
}

std::string executableSibling(const char *argv0, const std::string &name) {
  fs::path self(argv0);
  if (self.has_parent_path()) return (self.parent_path() / name).string();
  return name;
}

std::string defaultWorkDir() {
  fs::path base = fs::temp_directory_path();
  return (base / ("annc_tf_pipeline_v2_" + std::to_string(std::time(nullptr)) +
                  "_" + std::to_string(getpid())))
      .string();
}

bool parsePipelineOptions(int argc, char **argv, PipelineOptions *opts) {
  opts->inputGraphDef = takeValue(argc, argv, "--input_graphdef");
  opts->outputGraphDef = takeValue(argc, argv, "--output_graphdef");
  opts->kernelName = takeValue(argc, argv, "--kernel_name", opts->kernelName);
  opts->sharedLibPath = takeValue(argc, argv, "--shared_lib_path");
  opts->kernelLibPath = takeValue(argc, argv, "--kernel_lib_path",
                                  std::getenv("ANNC_KERNEL_LIB_PATH")
                                      ? std::getenv("ANNC_KERNEL_LIB_PATH")
                                      : "");
  opts->workDir = takeValue(argc, argv, "--work_dir", defaultWorkDir());
  opts->skipPreprocess = !hasArg(argc, argv, "--no-skip-preprocess");
  opts->assumeOpaqueI8 = !hasArg(argc, argv, "--no-assume-opaque-i8");
  opts->skipConverter = hasArg(argc, argv, "--skip-converter");
  opts->keepTemps = hasArg(argc, argv, "--keep_temps");
  opts->verbose = hasArg(argc, argv, "--verbose") || hasArg(argc, argv, "-v");
  // --verbose is the CLI face of ANNC_LOG_LEVEL=info; an explicit variable
  // wins. Must run before the first logging call (the level is cached on
  // first use).
  if (opts->verbose && !std::getenv("ANNC_LOG_LEVEL"))
    ::setenv("ANNC_LOG_LEVEL", "info", /*overwrite=*/0);
  if (!collectRepeated(argc, argv, "--input", &opts->inputs) ||
      !collectRepeated(argc, argv, "--output_tensor", &opts->outputTensors))
    return false;

  if (opts->inputGraphDef.empty()) {
    ANNC_LOG_ERROR("tf-pipeline-v2")
        << "--input_graphdef is required\n";
    return false;
  }
  if (opts->outputGraphDef.empty() && !opts->skipConverter) {
    ANNC_LOG_ERROR("tf-pipeline-v2")
        << "--output_graphdef is required unless --skip-converter is given\n";
    return false;
  }
  if (opts->outputTensors.empty()) {
    ANNC_LOG_ERROR("tf-pipeline-v2")
        << "at least one --output_tensor is required\n";
    return false;
  }
  return true;
}

bool runV2Chain(int argc, char **argv) {
  PipelineOptions opts;
  if (!parsePipelineOptions(argc, argv, &opts)) return false;

  std::error_code fsEc;
  fs::create_directories(opts.workDir, fsEc);
  fs::permissions(opts.workDir, fs::perms::owner_all, fsEc);
  if (fsEc) {
    ANNC_LOG_ERROR("tf-pipeline-v2")
        << "cannot create work dir " << opts.workDir << ": "
        << fsEc.message() << "\n";
    return false;
  }
  fs::path work(opts.workDir);
  fs::path stablehlo = work / "model_stablehlo.mlir";
  fs::path sunk = work / "model_sunk.mlir";
  fs::path fusedAtir = work / "model_fused_atir.mlir";
  fs::path customizeAtir = work / "model_customize.mlir";
  fs::path fusionMetadata = work / "fusion_metadata.json";
  fs::path loweredMlir = work / "model_lowered.mlir";
  fs::path generatedSo = fs::absolute(work / "annc_generated_kernel.so");
  std::string runtimeSharedLibPath =
      opts.sharedLibPath.empty() ? generatedSo.string() : opts.sharedLibPath;

  std::string pb2Stablehlo = executableSibling(argv[0], "pb2stablehlo");
  std::string anncTf2AtirV2 = executableSibling(argv[0], "annc-stablehlo-sink");
  std::string anncAsm = executableSibling(argv[0], "annc-asm");
  std::string anncFusionMetadata =
      executableSibling(argv[0], "annc-fusion-metadata");
  std::string annc = executableSibling(argv[0], "annc");
  std::string anncConverterV2 =
      executableSibling(argv[0], "annc-converter-v2");

  // Step 1: GraphDef -> stablehlo (the V2 frontend).
  std::vector<std::string> pb2Args = {pb2Stablehlo, opts.inputGraphDef};
  for (const std::string &input : opts.inputs) {
    pb2Args.push_back("--input");
    pb2Args.push_back(input);
  }
  for (const std::string &tensor : opts.outputTensors) {
    pb2Args.push_back("--output");
    pb2Args.push_back(tensor);
  }
  if (opts.skipPreprocess) pb2Args.push_back("--skip-preprocess");
  if (opts.assumeOpaqueI8) pb2Args.push_back("--assume-opaque-i8");
  pb2Args.push_back("-o");
  pb2Args.push_back(stablehlo.string());

  // Step 2: sink (stablehlo -> atir with residue carriers).
  // Step 3: the V2 fusion pass.
  // Step 4: fast codegen over the fused artifact.
  // Step 5: the converter requires this metadata; always produced.
  // Step 6: prune to the AOT kernel funcs (the prune drops @main and the
  // model-level calls), then turn each Customize body into a context entry
  // and lower through the V2 conversion. The legacy ConvertAtirToLinalg/
  // CustomizeCallLowering path rejects the annc_context_v3 ABI and must
  // not run here; annc -shared appends the ABI conversion for the entries.
  // The context-ABI switch is set before both steps so the asm children
  // and the linker see the same route.
  ::setenv("ANNC_ENABLE_LOWERING_CONTEXT_ABI", "1", /*overwrite=*/1);
  bool ok =
      runCommand(pb2Args) &&
      runCommand({anncTf2AtirV2, stablehlo.string(), "-o", sunk.string()}) &&
      runCommand({anncAsm, sunk.string(), "-atir-op-fusion-v2", "-o",
                  fusedAtir.string()}) &&
      runCommand({anncAsm, fusedAtir.string(), "--annc-fast-codegen", "-o",
                  customizeAtir.string()}) &&
      runCommand({anncFusionMetadata, customizeAtir.string(), "-o",
                  fusionMetadata.string()}) &&
      runCommand({anncAsm, customizeAtir.string(),
                  "--atir-prune-func=execution-mode=aot", "--annc-fast-codegen",
                  "--lower-context-kernel-entries", "--convert-atir-v2-to-linalg",
                  "-o", loweredMlir.string()});
  if (!ok) return false;

  // Step 7: the AOT shared library, only when a site materialized. JIT-only
  // graphs skip the library; the converter still needs the paths for its
  // GraphDef attributes.
  bool hasAot = false;
  if (!containsAotKernel(customizeAtir, hasAot)) return false;
  {
    // Zero fusion sites is usually a routing miss (ANNC_GEMM off by
    // default), not a valid empty graph; surface it instead of a silent
    // no-op rewrite pass.
    std::ifstream probe(customizeAtir, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(probe)),
                     std::istreambuf_iterator<char>());
    if (text.find("annc.kernel") == std::string::npos)
      ANNC_LOG_WARN("tf-pipeline-v2")
          << "no annc.kernel sites materialized; check ANNC_GEMM (dot "
             "fusion is off by default) and the fusion coverage\n";
  }
  if (hasAot) {
    // The context-kernel ABI lowering behind annc -shared is guarded off in
    // V2 builds unless this switch is on; the pipeline owns the switch so
    // its AOT path works out of the box.
    ::setenv("ANNC_ENABLE_LOWERING_CONTEXT_ABI", "1", /*overwrite=*/1);
    std::vector<std::string> linkArgs = {
        annc, loweredMlir.string(), "--shared", "-o", generatedSo.string()};
    if (!opts.kernelLibPath.empty()) {
      linkArgs.push_back("--kernel-lib-path");
      linkArgs.push_back(opts.kernelLibPath);
    }
    ok = runCommand(linkArgs);
    if (!ok) return false;
  }

  // Step 8: rewrite the GraphDef. Both module/library paths are passed so
  // aot and jit metadata are covered by one invocation. Skipped entirely on
  // plugin-only deployments: the predictor's in-process rewriter consumes
  // the metadata/module/library through ANNC_V2_* env instead.
  bool skipNote = opts.skipConverter;
  if (skipNote) {
    ANNC_LOG_INFO("tf-pipeline-v2")
        << "offline converter step skipped (--skip-converter); the in-process "
           "plugin rewrites inside the predictor\n";
  }
  std::vector<std::string> converterArgs = {
      anncConverterV2,
      "--metadata_json",
      fusionMetadata.string(),
      "--input_graphdef",
      opts.inputGraphDef,
      "--output_graphdef",
      opts.outputGraphDef,
      "--atir_module_path",
      fs::absolute(fusedAtir).string(),
      "--shared_lib_path",
      fs::absolute(runtimeSharedLibPath).string(),
  };
  if (!opts.kernelName.empty()) {
    converterArgs.push_back("--kernel_name");
    converterArgs.push_back(opts.kernelName);
  }
  ok = skipNote || runCommand(converterArgs);
  if (!ok) return false;

  {
    ANNC_LOG_INFO("tf-pipeline-v2")
        << "fused artifact (jit template): " << fs::absolute(fusedAtir)
        << "\n";
    std::string siteNote =
        (hasAot || !opts.sharedLibPath.empty())
            ? ""
            : " (skipped: no AOT site)";
    ANNC_LOG_INFO("tf-pipeline-v2")
        << "kernel shared library: " << fs::absolute(runtimeSharedLibPath)
        << siteNote << "\n";
    ANNC_LOG_INFO("tf-pipeline-v2")
        << "fusion metadata: " << fusionMetadata << "\n";
    ANNC_LOG_INFO("tf-pipeline-v2")
        << "rewritten GraphDef: " << opts.outputGraphDef << "\n";
  }

  if (!opts.keepTemps) {
    std::error_code ec;
    // The fused artifact, metadata and generated library stay referenced by
    // the rewritten GraphDef; the upstream intermediates are removable.
    for (const fs::path &temp :
         {stablehlo, sunk, customizeAtir, loweredMlir}) {
      fs::remove(temp, ec);
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
  return runV2Chain(argc, argv) ? 0 : 1;
}
