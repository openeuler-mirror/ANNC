// annc-onnx-pipeline — whole-model ONNX -> AOT kernel pipeline for the ANNC
// ONNX Runtime EP (the runtime-compile mode).
//
// The EP exports its graph to ONNX and runs this driver on a cold cache.
// ANNC_ONNX_USE_STABLEHLO=1 selects the optional StableHLO frontend.
//
// Usage:
//   annc-onnx-pipeline --model <model.onnx> --out-dir <dir>
//                      [--work-dir <dir>] [--execution-modes aot] [--verbose]
//   annc-onnx-pipeline --toolchain-info
//
// The ATIR frontend publishes sites.json and StableHLO publishes model.fused.mlir.
// Both produce libannc_kernels.so when at least one site was fused.
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Dialect/Atir/AtirOps.h"
#include "FusionMetadata/FusionMetadataJson.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

namespace fs = std::filesystem;
using namespace mlir;

namespace {

constexpr const char *kReportSchema = "annc-ep-site-report/1";
constexpr const char *kContractVersion = "1.0.0";

// The tools this pipeline invokes, directly or through the annc driver (which
// shells out to the LLVM tools by bare name). The cache key's toolchain
// fingerprint covers all of them: a rebuilt codegen must not reuse kernels
// compiled by the previous one.
constexpr const char *kPipelineTools[] = {
    "annc-onnx-pipeline", "annc-asm", "annc", "mlir-opt", "mlir-translate",
    "opt", "llc", "clang",
};

std::string stablehloToolDir() {
#ifdef ANNC_STABLEHLO_TOOL_DIR
  return ANNC_STABLEHLO_TOOL_DIR;
#else
  return "";
#endif
}

struct Options {
  std::string model;
  std::string outDir;
  std::string workDir;
  std::string executionModes = "aot";
  bool toolchainInfo = false;
  bool verbose = false;
  bool keepTemps = false;
};

void printUsage() {
  std::cout
      << "Usage: annc-onnx-pipeline --model <model.onnx> --out-dir <dir>\n"
         "                           [--work-dir <dir>] [--execution-modes aot]\n"
         "                           [--keep-temps] [--verbose]\n"
         "       annc-onnx-pipeline --toolchain-info\n"
         "  Set ANNC_ONNX_USE_STABLEHLO=1 to enable the StableHLO frontend.\n";
}

bool hasArg(int argc, char **argv, const std::string &name) {
  for (int i = 1; i < argc; ++i) {
    if (name == argv[i]) return true;
  }
  return false;
}

std::string takeValue(int argc, char **argv, const std::string &name,
                      const std::string &fallback = "") {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == name && i + 1 < argc) return argv[i + 1];
    if (arg.rfind(name + "=", 0) == 0) return arg.substr(name.size() + 1);
  }
  return fallback;
}

bool parseOptions(int argc, char **argv, bool stablehlo, Options *opts) {
  if (stablehlo) {
    for (int i = 1; i < argc; ++i) {
      const std::string raw = argv[i];
      const std::string arg = raw.substr(0, raw.find('='));
      if (arg == "--model" || arg == "--out-dir" || arg == "--work-dir" ||
          arg == "--execution-modes") {
        if (raw != arg || (++i < argc && argv[i][0] != '-')) continue;
        std::cerr << "[annc-onnx-pipeline] missing value for " << arg << "\n";
        return false;
      }
      if (raw == "--toolchain-info" || raw == "--verbose" || raw == "-v" ||
          raw == "--keep-temps")
        continue;
      std::cerr << "[annc-onnx-pipeline] unknown option: " << arg << "\n";
      return false;
    }
  }
  opts->model = takeValue(argc, argv, "--model");
  opts->outDir = takeValue(argc, argv, "--out-dir");
  opts->workDir = takeValue(argc, argv, "--work-dir", opts->outDir);
  opts->executionModes = takeValue(argc, argv, "--execution-modes", "aot");
  opts->toolchainInfo = hasArg(argc, argv, "--toolchain-info");
  opts->verbose = hasArg(argc, argv, "--verbose") || hasArg(argc, argv, "-v");
  opts->keepTemps = hasArg(argc, argv, "--keep-temps");
  if (opts->toolchainInfo) return true;
  if (opts->model.empty() || opts->outDir.empty()) {
    std::cerr << "[annc-onnx-pipeline] --model and --out-dir are required\n";
    return false;
  }
  return true;
}

std::string selfDir(const char *argv0) {
  fs::path executable(argv0);
  if (!executable.has_parent_path()) {
    const char *searchPath = std::getenv("PATH");
    std::stringstream path(searchPath ? searchPath : "");
    std::string directory;
    while (std::getline(path, directory, ':')) {
      const fs::path candidate = fs::path(directory) / executable;
      if (fs::is_regular_file(candidate) &&
          ::access(candidate.c_str(), X_OK) == 0) {
        executable = candidate;
        break;
      }
    }
  }
  return fs::canonical(executable).parent_path().string();
}

std::string sibling(const char *argv0, const std::string &name) {
  return (fs::path(selfDir(argv0)) / name).string();
}

// The install layout is <prefix>/bin/<tool> + <prefix>/lib, which is also the
// layout of the build tree this tool is run from during development.
std::string prefixOf(const char *argv0) {
  return fs::path(selfDir(argv0)).parent_path().string();
}

std::vector<std::string> toolSearchDirs(const char *argv0);

struct ToolFingerprint {
  const char *name;
  long long size;
  long long mtimeNs;
  bool found;
};

ToolFingerprint fingerprintOf(const char *argv0, const char *name) {
  ToolFingerprint entry{name, 0, 0, false};
  std::vector<std::string> candidates;
  if (std::strcmp(name, "onnx-mlir-opt") == 0) {
    const std::string upstream = stablehloToolDir();
    if (!upstream.empty()) candidates.push_back((fs::path(upstream) / name).string());
  }
  for (const std::string &dir : toolSearchDirs(argv0)) {
    candidates.push_back((fs::path(dir) / name).string());
  }
  candidates.push_back(sibling(argv0, name));
  if (const char *path = std::getenv("PATH")) {
    std::stringstream stream(path);
    std::string dir;
    while (std::getline(stream, dir, ':')) {
      if (!dir.empty()) candidates.push_back((fs::path(dir) / name).string());
    }
  }
  for (const std::string &candidate : candidates) {
    struct stat info {};
    if (::stat(candidate.c_str(), &info) != 0) continue;
    if (!S_ISREG(info.st_mode)) continue;
    entry.size = static_cast<long long>(info.st_size);
    entry.mtimeNs = static_cast<long long>(info.st_mtim.tv_sec) * 1000000000LL +
                    static_cast<long long>(info.st_mtim.tv_nsec);
    entry.found = true;
    break;
  }
  return entry;
}

void printToolchainInfo(const char *argv0, bool stablehlo) {
  std::ostringstream out;
  out << "{\"schema\":\"annc-toolchain-info/1\",\"frontend\":\""
      << (stablehlo ? "stablehlo" : "atir") << "\",\"tools\":[";
  bool first = true;
  auto emit = [&](const char *name) {
    ToolFingerprint entry = fingerprintOf(argv0, name);
    if (!entry.found) return;
    if (!first) out << ",";
    first = false;
    out << "{\"name\":\"" << entry.name << "\",\"size\":" << entry.size
        << ",\"mtime_ns\":" << entry.mtimeNs << "}";
  };
  for (const char *name : kPipelineTools) emit(name);
  if (stablehlo) {
    for (const char *name : {"annc-onnx2stablehlo", "annc-onnx-framework-opt",
                            "annc-stablehlo-opt", "onnx-mlir-opt"})
      emit(name);
  } else {
    emit("annc-onnx2atir");
    emit("annc-opt");
  }
  out << "]}";
  std::cout << out.str() << "\n";
}

#ifndef ANNC_LLVM_BIN_DEFAULT
#define ANNC_LLVM_BIN_DEFAULT ""
#endif
#ifndef ANNC_LLVM_LIB_DIR_DEFAULT
#define ANNC_LLVM_LIB_DIR_DEFAULT ""
#endif
#ifndef ANNC_KERNEL_LIB_PATH
#define ANNC_KERNEL_LIB_PATH ""
#endif

// LLVM 20 toolchain: ANNC_LLVM_BIN overrides the directory baked in at
// configure time. Every tool is resolved from here, so a system LLVM 17 of the
// same name is never picked up through PATH.
std::string llvmBinDir() {
  if (const char *value = std::getenv("ANNC_LLVM_BIN")) {
    if (value[0] != '\0') return value;
  }
  return ANNC_LLVM_BIN_DEFAULT;
}

// The LLVM static archives live in the toolchain lib directory next to the
// selected bin directory.
std::string llvmLibDir() {
  if (const char *value = std::getenv("ANNC_LLVM_BIN")) {
    if (value[0] != '\0') {
      const fs::path prefix = fs::path(value).parent_path();
      for (const char *sub : {"lib64", "lib"}) {
        const fs::path candidate = prefix / sub;
        if (fs::exists(candidate / "libLLVMSupport.a") &&
            fs::exists(candidate / "libLLVMDemangle.a")) {
          return candidate.string();
        }
      }
      return (prefix / "lib64").string();
    }
  }
  return ANNC_LLVM_LIB_DIR_DEFAULT;
}

// Directories the pipeline tools and the LLVM tools they shell out to live in.
// The pipeline tools are siblings in <prefix>/bin; mlir-opt/clang/etc. come
// from the LLVM 20 toolchain directory selected above.
std::vector<std::string> toolSearchDirs(const char *argv0) {
  std::vector<std::string> dirs;
  dirs.push_back(selfDir(argv0));
  const std::string llvmBin = llvmBinDir();
  if (!llvmBin.empty() && llvmBin != dirs.front()) dirs.push_back(llvmBin);
  return dirs;
}

// The annc driver links the generated kernels against the ANNC kernel archives
// in ANNC_KERNEL_LIB_PATH and the LLVM support archives in the LLVM 20 lib
// directory. The installed layout (<prefix>/<libdir>) satisfies both; a build
// tree has no install libdir, so the caller must set ANNC_KERNEL_LIB_PATH.
std::string kernelLibPathOf(const char *argv0) {
  if (const char *value = std::getenv("ANNC_KERNEL_LIB_PATH")) {
    if (value[0] != '\0') return value;
  }
  // Installed layout: the configured libdir (lib64 on aarch64). A build tree
  // has no such directory, so fall back to <prefix>/lib.
  std::error_code ec;
  const std::string compiled = ANNC_KERNEL_LIB_PATH;
  if (!compiled.empty() && fs::exists(compiled, ec)) return compiled;
  return (fs::path(prefixOf(argv0)) / "lib").string();
}

// Child environment: the annc driver resolves mlir-opt/mlir-translate/opt/llc/
// clang from ANNC_LLVM_BIN and its kernel archives from ANNC_KERNEL_LIB_PATH,
// so the pipeline must not rely on the host's PATH or install layout.
std::vector<std::string> buildChildEnv(const char *argv0) {
  std::vector<std::string> env;
  const std::string prefix = prefixOf(argv0);
  const std::string binDir = selfDir(argv0);
  // clang is launched by the driver and needs the caller's target runtime
  // search paths to find crtbeginS.o, libstdc++, and related files.
  const char *keep[] = {"HOME", "LANG", "LC_ALL", "TERM", "LIBRARY_PATH",
                        "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"};

  std::string pathEnv;
  for (const std::string &dir : toolSearchDirs(argv0)) pathEnv += dir + ":";
  const char *path = std::getenv("PATH");
  pathEnv += path ? path : "/usr/bin:/bin";
  env.push_back("PATH=" + pathEnv);
  for (const char *name : keep) {
    if (const char *value = std::getenv(name)) env.push_back(std::string(name) + "=" + value);
  }

  if (const char *value = std::getenv("ANNC_KERNEL_LIB_PATH")) {
    env.push_back(std::string("ANNC_KERNEL_LIB_PATH=") + value);
  } else {
    env.push_back("ANNC_KERNEL_LIB_PATH=" + kernelLibPathOf(argv0));
  }

  // Pin the driver to the same LLVM 20 toolchain this process resolves.
  env.push_back("ANNC_LLVM_BIN=" + llvmBinDir());
  // GEMM tuning config and the fusion switches travel with the caller: they
  // change what the ANNC matcher accepts, so the EP's graph view and this
  // compile must agree on them. The four ANNC_EP_FUSION_* switches are read by
  // annc-opt's OpFusion pass, which then skips the disabled matcher entirely.
  // The ORT EP defaults these patterns off: an unset switch is passed as "0"
  // so the compiler and the EP's claim-time filter agree.
  const char *fusion_switches[] = {
      "ANNC_EP_FUSION_TENSORDOT_MATMUL", "ANNC_EP_FUSION_LAYER_NORM",
      "ANNC_EP_FUSION_MASKED_MUL", "ANNC_EP_FUSION_BATCHED_MATMUL_ADD"};
  for (const char *name : {"ANNC_GEMM_CONFIG", "ANNC_GEMM",
                           "ANNC_EP_FUSION_TENSORDOT_MATMUL",
                           "ANNC_EP_FUSION_LAYER_NORM",
                           "ANNC_EP_FUSION_MASKED_MUL",
                           "ANNC_EP_FUSION_BATCHED_MATMUL_ADD",
                           "ANNC_EP_ORT_BUILD_TREE"}) {
    if (const char *value = std::getenv(name)) {
      env.push_back(std::string(name) + "=" + value);
      continue;
    }
    bool is_fusion_switch = false;
    for (const char *fusion : fusion_switches) {
      if (std::strcmp(name, fusion) == 0) {
        is_fusion_switch = true;
        break;
      }
    }
    if (is_fusion_switch) {
      env.push_back(std::string(name) + "=0");
    }
  }
  return env;
}

// fork/exec with an explicit environment and working directory. execve (not
// system()) so paths with spaces and a curated env are both safe; the child
// runs in `cwd` because the annc driver creates its temporaries in the current
// directory and concurrent compiles must not collide.
int runTool(const std::vector<std::string> &args, const std::string &cwd,
            const std::vector<std::string> &env, bool verbose) {
  std::ostringstream pretty;
  for (size_t i = 0; i < args.size(); ++i) {
    if (i) pretty << ' ';
    pretty << args[i];
  }
  if (verbose) std::cerr << "[annc-onnx-pipeline] " << pretty.str() << "\n";

  std::vector<char *> argv;
  argv.reserve(args.size() + 1);
  for (const std::string &arg : args) argv.push_back(const_cast<char *>(arg.c_str()));
  argv.push_back(nullptr);
  std::vector<char *> envp;
  envp.reserve(env.size() + 1);
  for (const std::string &entry : env) envp.push_back(const_cast<char *>(entry.c_str()));
  envp.push_back(nullptr);

  pid_t pid = ::fork();
  if (pid < 0) {
    std::cerr << "[annc-onnx-pipeline] fork failed\n";
    return -1;
  }
  if (pid == 0) {
    if (::chdir(cwd.c_str()) != 0) ::_exit(126);
    ::execve(argv[0], argv.data(), envp.data());
    std::fprintf(stderr, "[annc-onnx-pipeline] cannot execute %s: %s\n",
                 argv[0], std::strerror(errno));
    ::_exit(127);
  }
  int status = 0;
  if (::waitpid(pid, &status, 0) < 0) return -1;
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
}

bool writeFile(const std::string &path, const std::string &content) {
  std::ofstream out(path);
  if (!out.is_open()) return false;
  out << content;
  return out.good();
}

// The fused module carries one kernel func per site; the report is the
// serialized FusionInfo list, which already names the kernel, its arguments,
// the ABI and the fused region's member nodes.
bool writeFusionArtifact(const fs::path &fusedModulePath,
                         const fs::path &outPath, bool stablehlo,
                         size_t *siteCount, std::string *error) {
  DialectRegistry registry;
  registry.insert<func::FuncDialect, arith::ArithDialect, LLVM::LLVMDialect>();
  // The EP reads the published StableHLO kernel IR with this same registry.
  if (!stablehlo) registry.insert<atir::AtirDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  if (!stablehlo) context.allowUnregisteredDialects();

  auto module = parseSourceFile<ModuleOp>(fusedModulePath.string(), &context);
  if (!module) {
    *error = "cannot parse the fused module '" + fusedModulePath.string() + "'";
    return false;
  }

  // Walk the kernels directly instead of using extractFusionInfos(): that
  // helper treats "no fused kernel" as an error (the AOT tools rely on it),
  // while for a graph with no fusable region zero sites is the expected,
  // successful outcome -- the EP then claims nothing and the model runs on the
  // CPU EP.
  std::vector<annc::fusion::FusionInfo> infos;
  llvm::Error errors = llvm::Error::success();
  module->walk([&](func::FuncOp funcOp) {
    if (!funcOp->hasAttr("annc.kernel")) return;
    auto info = annc::fusion::readFusionInfo(funcOp);
    if (!info) {
      errors = llvm::joinErrors(std::move(errors), info.takeError());
      return;
    }
    infos.push_back(std::move(*info));
  });
  if (errors) {
    *error = llvm::toString(std::move(errors));
    return false;
  }

  *siteCount = infos.size();
  std::string content;
  if (stablehlo) {
    // Publish the same extracted module that enters the common backend.
    llvm::raw_string_ostream stream(content);
    module->print(stream);
    stream << "\n";
  } else {
    nlohmann::json report;
    report["schema"] = kReportSchema;
    report["contract_version"] = kContractVersion;
    report["sites"] = annc::fusion::fusionInfosToJson(infos)["fusions"];
    content = report.dump(1) + "\n";
  }
  if (writeFile(outPath.string(), content)) return true;
  *error = "cannot write '" + outPath.string() + "'";
  return false;
}

int runPipeline(const char *argv0, const Options &opts, bool stablehlo) {
  if (opts.executionModes != "aot") {
    std::cerr << "[annc-onnx-pipeline] execution mode '" << opts.executionModes
              << "' is not implemented (only 'aot')\n";
    return 2;
  }

  const fs::path model = fs::absolute(opts.model).lexically_normal();
  const fs::path work = fs::absolute(opts.workDir).lexically_normal();
  const fs::path out = fs::absolute(opts.outDir).lexically_normal();
  const fs::path raw = work / "model.mlirbc";
  const fs::path hlo = work / "model.stablehlo.mlir";
  const fs::path hloMetadata = work / "model.stablehlo.json";
  const fs::path partitioned = work / "model.partitioned.mlir";
  const fs::path bound = work / "model.bound.mlir";
  const fs::path fused = work / "model.fused.mlirbc";
  const fs::path aot = stablehlo ? fused : work / "model.aot.mlirbc";
  const fs::path lowered = work / "model.low.mlir";
  const fs::path jsonReport = out / "sites.json";
  const fs::path kernelIr = out / "model.fused.mlir";
  const fs::path kernelSo = out / "libannc_kernels.so";

  const std::vector<fs::path> intermediates =
      stablehlo ? std::vector<fs::path>{hlo, hloMetadata, partitioned, bound,
                                       fused, lowered}
                : std::vector<fs::path>{raw, fused, aot, lowered};
  if (!fs::is_regular_file(model)) {
    std::cerr << "[annc-onnx-pipeline] model not found: " << model << "\n";
    return 2;
  }
  auto outputs = intermediates;
  outputs.insert(outputs.end(), {jsonReport, kernelIr, kernelSo});
  for (const auto &output : outputs) {
    std::error_code ignored;
    if (model == output || fs::equivalent(model, output, ignored)) {
      std::cerr << "[annc-onnx-pipeline] ONNX input conflicts with pipeline output: "
                << output << "\n";
      return 2;
    }
    if (fs::exists(output) && !fs::is_regular_file(output)) {
      std::cerr << "[annc-onnx-pipeline] output must be a regular file: " << output
                << "\n";
      return 2;
    }
  }

  // The annc driver links the kernel and LLVM support archives.
  const std::pair<fs::path, const char *> required[] = {
      {kernelLibPathOf(argv0), "libANNCBuiltinKernels.a"},
      {llvmLibDir(), "libLLVMSupport.a"},
      {llvmLibDir(), "libLLVMDemangle.a"},
  };
  for (const auto &[dir, name] : required) {
    if (!fs::is_regular_file(dir / name)) {
      std::cerr << "[annc-onnx-pipeline] " << (dir / name)
                << " is missing; set ANNC_KERNEL_LIB_PATH to a complete ANNC "
                   "install and select an LLVM toolchain with static archives\n";
      return 2;
    }
  }

  fs::create_directories(work);
  fs::create_directories(out);
  const auto env = buildChildEnv(argv0);
  auto runStage = [&](const char *name, const std::vector<std::string> &args,
                      int allowedStatus = 0) {
    if (opts.verbose)
      std::cerr << "[annc-onnx-pipeline] stage: " << name << "\n";
    const int status = runTool(args, work.string(), env, opts.verbose);
    if (status != 0 && status != allowedStatus)
      std::cerr << "[annc-onnx-pipeline] " << name << " failed (exit " << status
                << ")\n";
    return status == allowedStatus ? 0 : status;
  };

  int rc;
  if (stablehlo) {
    std::vector<std::string> command{
        sibling(argv0, "annc-onnx2stablehlo"), model.string(), "-o", hlo.string(),
        "--metadata", hloMetadata.string()};
    const std::string upstream = stablehloToolDir();
    if (!upstream.empty())
      command.insert(command.end(), {"--tool-dir", upstream});
    rc = runStage("ONNX to StableHLO", command);
    if (rc != 0) return rc;

    // Generic assembly preserves provenance across the matcher and binder.
    const std::string reader = sibling(argv0, "annc-stablehlo-opt");
    rc = runStage("StableHLO fusion",
                  {reader, hlo.string(), "--annc-stablehlo-partition",
                   "--mlir-print-debuginfo", "--mlir-print-op-generic", "-o",
                   partitioned.string()});
    if (rc != 0) return rc;
    rc = runStage("ONNX binding",
                  {reader, partitioned.string(), "--annc-stablehlo-bind-onnx",
                   "--mlir-print-debuginfo", "--mlir-print-op-generic", "-o",
                   bound.string()});
    if (rc != 0) return rc;
    rc = runStage("kernel extraction",
                  {reader, bound.string(), "--annc-stablehlo-kernels",
                   "--emit-bytecode", "-o", fused.string()});
  } else {
    // The importer exits 2 when unsupported nodes were imported as opaque ops.
    rc = runStage("ONNX to ATIR",
                  {sibling(argv0, "annc-onnx2atir"), model.string(), "--quiet",
                   "--emit-bytecode", "-o", raw.string()},
                  2);
    if (rc != 0) return rc;
    rc = runStage("ATIR fusion",
                  {sibling(argv0, "annc-opt"), raw.string(),
                   "--atir-identity-canonicalize", "--atir-op-fusion",
                   "--emit-bytecode", "-o", fused.string()});
  }
  if (rc != 0) return rc;

  size_t siteCount = 0;
  std::string error;
  if (!writeFusionArtifact(fused, stablehlo ? kernelIr : jsonReport, stablehlo,
                           &siteCount, &error)) {
    std::cerr << "[annc-onnx-pipeline] kernel metadata failed: " << error << "\n";
    return 1;
  }

  if (siteCount != 0) {
    if (!stablehlo) {
      rc = runStage("AOT kernel extraction",
                    {sibling(argv0, "annc-opt"), fused.string(),
                     "--atir-prune-func=execution-mode=aot", "--emit-bytecode",
                     "-o", aot.string()});
      if (rc != 0) return rc;
    }
    std::string fastCodegenPass = "--annc-fast-codegen";
#ifdef ANNC_ENABLE_KDNN_ADAPTOR
    fastCodegenPass += "=enable-kdnn=true";
#endif
    rc = runStage("backend lowering",
                  {sibling(argv0, "annc-asm"), aot.string(), fastCodegenPass,
                   "--annc-aarch64-gemm-pipeline", "-o", lowered.string()});
    if (rc != 0) return rc;
    rc = runStage("kernel compilation",
                  {sibling(argv0, "annc"), lowered.string(), "--shared", "-o",
                   kernelSo.string()});
    if (rc != 0) return rc;
  }

  // The EP prefers kernel IR. Remove an earlier run's inactive report.
  fs::remove(stablehlo ? jsonReport : kernelIr);
  if (siteCount == 0) fs::remove(kernelSo);
  if (!opts.keepTemps) {
    for (const auto &path : intermediates) {
      std::error_code ignored;
      fs::remove(path, ignored);
    }
  }
  std::cerr << "[annc-onnx-pipeline] " << siteCount << " site(s)";
  if (siteCount != 0)
    std::cerr << " -> " << kernelSo.string();
  else
    std::cerr << "; skipping codegen";
  std::cerr << "\n";
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (hasArg(argc, argv, "--help") || hasArg(argc, argv, "-h")) {
    printUsage();
    return 0;
  }

  const char *frontend = std::getenv("ANNC_ONNX_USE_STABLEHLO");
  const bool stablehlo = frontend && std::strcmp(frontend, "1") == 0;
  Options opts;
  if (!parseOptions(argc, argv, stablehlo, &opts)) {
    printUsage();
    return 2;
  }
  try {
    if (opts.toolchainInfo) {
      printToolchainInfo(argv[0], stablehlo);
      return 0;
    }
    return runPipeline(argv[0], opts, stablehlo);
  } catch (const std::exception &error) {
    std::cerr << "[annc-onnx-pipeline] " << error.what() << "\n";
    return 1;
  }
}
