// annc-onnx-pipeline — whole-model ONNX -> AOT kernel pipeline for the ANNC
// ONNX Runtime EP (the runtime-compile mode).
//
// The EP exports the OrtGraph it partitions to an ONNX model and runs this tool
// once per (model, toolchain) on a cold cache, from GetCapability. It performs
// the same stages the offline flow documents, with two deliberate differences:
//
//   * intermediates are MLIR bytecode, not text: the target models' textual
//     ATIR reaches 465 MB for a 116 MB model (measured);
//   * the site report is written here, by parsing the fused module in-process.
//     annc-fusion-metadata reads textual MLIR only, and the report must carry
//     the fused region's member nodes so the EP can claim exactly that set.
//
// Usage:
//   annc-onnx-pipeline --model <model.onnx> --out-dir <dir>
//                      [--work-dir <dir>] [--execution-modes aot] [--verbose]
//   annc-onnx-pipeline --toolchain-info
//
// Outputs in --out-dir: sites.json, libannc_kernels.so (the latter only when at
// least one site was fused). Exit status: 0 when the pipeline ran (even with
// zero sites), non-zero when a stage failed -- the caller then falls back to
// the CPU EP rather than failing the session.
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
    "annc-onnx-pipeline", "annc-onnx2atir", "annc-opt",
    "annc-asm",           "annc",           "mlir-opt",
    "mlir-translate",     "opt",            "llc",
    "clang",
};

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
         "       annc-onnx-pipeline --toolchain-info\n";
}

bool hasArg(int argc, char **argv, const std::string &name) {
  for (int i = 1; i < argc; ++i) {
    if (name == argv[i]) return true;
  }
  return false;
}

std::string takeValue(int argc, char **argv, const std::string &name,
                      const std::string &fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (name == argv[i]) return argv[i + 1];
  }
  return fallback;
}

bool parseOptions(int argc, char **argv, Options *opts) {
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
  return fs::absolute(fs::path(argv0)).lexically_normal().parent_path().string();
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

void printToolchainInfo(const char *argv0) {
  std::ostringstream out;
  out << "{\"schema\":\"annc-toolchain-info/1\",\"tools\":[";
  bool first = true;
  for (const char *name : kPipelineTools) {
    ToolFingerprint entry = fingerprintOf(argv0, name);
    if (!entry.found) continue;
    if (!first) out << ",";
    first = false;
    out << "{\"name\":\"" << entry.name << "\",\"size\":" << entry.size
        << ",\"mtime_ns\":" << entry.mtimeNs << "}";
  }
  out << "]}";
  std::cout << out.str() << "\n";
}

// Directories the pipeline tools and the LLVM tools they shell out to live in.
// The installed layout keeps everything in <prefix>/bin; a build tree keeps the
// LLVM tools under _deps/llvm-build/bin instead, so both are searched.
std::vector<std::string> toolSearchDirs(const char *argv0) {
  std::vector<std::string> dirs;
  const std::string binDir = selfDir(argv0);
  dirs.push_back(binDir);
  const fs::path llvmBuildBin =
      fs::path(prefixOf(argv0)) / "_deps" / "llvm-build" / "bin";
  if (fs::exists(llvmBuildBin)) dirs.push_back(llvmBuildBin.string());
  return dirs;
}

// The annc driver links the generated kernels with a single `-L<dir>`: that
// directory must hold the ANNC kernel archives *and* the LLVM archives they
// reference (libLLVMSupport.a, libLLVMDemangle.a). Only the installed layout
// (<prefix>/lib) satisfies this; a build tree keeps the LLVM archives under
// _deps/llvm-build/lib, so the caller must set ANNC_KERNEL_LIB_PATH there.
std::string kernelLibPathOf(const char *argv0) {
  if (const char *value = std::getenv("ANNC_KERNEL_LIB_PATH")) {
    if (value[0] != '\0') return value;
  }
  return (fs::path(prefixOf(argv0)) / "lib").string();
}

// Child environment: the annc driver calls mlir-opt/mlir-translate/opt/llc/clang
// by bare name and reads its kernel archives from ANNC_KERNEL_LIB_PATH, so the
// pipeline must not rely on the host's PATH or install layout.
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
    env.push_back("ANNC_KERNEL_LIB_PATH=" +
                  (fs::path(prefix) / "lib").string());
  }
  if (const char *value = std::getenv("ANNC_CLANG")) {
    env.push_back(std::string("ANNC_CLANG=") + value);
  } else {
    fs::path clang = fs::path(binDir) / "clang";
    if (fs::exists(clang)) env.push_back("ANNC_CLANG=" + clang.string());
  }
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
bool writeSiteReport(const std::string &fusedModulePath, const std::string &outPath,
                     size_t *siteCount, std::string *error) {
  DialectRegistry registry;
  registry.insert<func::FuncDialect, LLVM::LLVMDialect, atir::AtirDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  context.allowUnregisteredDialects();

  auto module = parseSourceFile<ModuleOp>(fusedModulePath, &context);
  if (!module) {
    *error = "cannot parse the fused module '" + fusedModulePath + "'";
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

  nlohmann::json report;
  report["schema"] = kReportSchema;
  report["contract_version"] = kContractVersion;
  report["sites"] = annc::fusion::fusionInfosToJson(infos)["fusions"];
  *siteCount = infos.size();
  return writeFile(outPath, report.dump(1) + "\n");
}

int runPipeline(int argc, char **argv, const Options &opts) {
  if (opts.executionModes != "aot") {
    // The GEMM/MatMul family is jit-mode and needs a different stage sequence
    // (select-kernel + specialize-shapes, shape-driven); refusing here keeps
    // the EP on the CPU path instead of producing an empty kernel library.
    std::cerr << "[annc-onnx-pipeline] execution mode '" << opts.executionModes
              << "' is not implemented (only 'aot')\n";
    return 2;
  }
  std::error_code code;
  fs::create_directories(opts.workDir, code);
  fs::create_directories(opts.outDir, code);
  if (!fs::exists(opts.model)) {
    std::cerr << "[annc-onnx-pipeline] model not found: " << opts.model << "\n";
    return 2;
  }
  // Fail before the (expensive) frontend run when the toolchain cannot link:
  // the annc driver resolves every archive -- the ANNC kernels and the LLVM
  // support libraries -- from this single directory.
  {
    const fs::path kernelLib = kernelLibPathOf(argv[0]);
    // The annc driver resolves every archive -- the ANNC kernels and the LLVM
    // support libraries -- from this single directory.
    for (const char *required : {"libANNCBuiltinKernels.a", "libLLVMSupport.a"}) {
      if (!fs::exists(kernelLib / required)) {
        std::cerr << "[annc-onnx-pipeline] " << (kernelLib / required).string()
                  << " is missing; set ANNC_KERNEL_LIB_PATH to a complete ANNC "
                     "install lib directory (install/lib), or run `ninja install`\n";
        return 2;
      }
    }
  }

  const fs::path work(opts.workDir);
  const fs::path rawModule = work / "model.mlirbc";
  const fs::path fusedModule = work / "model.fused.mlirbc";
  const fs::path aotModule = work / "model.aot.mlirbc";
  const fs::path loweredModule = work / "model.low.mlir";
  const fs::path reportPath = fs::path(opts.outDir) / "sites.json";
  const fs::path kernelSo = fs::path(opts.outDir) / "libannc_kernels.so";

  const std::string anncOnnx2Atir = sibling(argv[0], "annc-onnx2atir");
  const std::string anncOpt = sibling(argv[0], "annc-opt");
  const std::string anncAsm = sibling(argv[0], "annc-asm");
  const std::string annc = sibling(argv[0], "annc");
  const std::vector<std::string> env = buildChildEnv(argv[0]);
  const std::string cwd = work.string();

  // 1. ONNX -> ATIR. The frontend exits 2 when the import succeeded but some
  //    nodes fell back to atir.opaque (annc-onnx2atir.cpp:585); that is normal
  //    for real models and does not affect the sites we fuse.
  int rc = runTool({anncOnnx2Atir, opts.model, "--quiet", "--emit-bytecode",
                    "-o", rawModule.string()},
                   cwd, env, opts.verbose);
  if (rc != 0 && rc != 2) return rc;

  // 2. Canonicalize + fuse. Every pattern runs here, including ones this EP
  //    does not support yet; the report decides what is claimed, not this tool.
  rc = runTool({anncOpt, rawModule.string(), "--atir-identity-canonicalize",
                "--atir-op-fusion", "--emit-bytecode", "-o", fusedModule.string()},
               cwd, env, opts.verbose);
  if (rc != 0) return rc;

  // 3. Site report, in-process (see the file header).
  auto dropIntermediates = [&]() {
    if (opts.keepTemps) return;
    std::error_code ignored;
    fs::remove(rawModule, ignored);
    fs::remove(fusedModule, ignored);
    fs::remove(aotModule, ignored);
    fs::remove(loweredModule, ignored);
  };

  size_t siteCount = 0;
  std::string error;
  if (!writeSiteReport(fusedModule.string(), reportPath.string(), &siteCount, &error)) {
    std::cerr << "[annc-onnx-pipeline] site report failed: " << error << "\n";
    return 1;
  }
  if (siteCount == 0) {
    // Zero sites is a valid outcome (a graph with no fusable region); skipping
    // codegen keeps the cold path short for such models.
    dropIntermediates();
    std::cerr << "[annc-onnx-pipeline] no fusion sites; skipping codegen\n";
    return 0;
  }

  // 4-6. Prune to the AOT kernels, lower, link one library per model.
  rc = runTool({anncOpt, fusedModule.string(),
                "--atir-prune-func=execution-mode=aot", "--emit-bytecode",
                "-o", aotModule.string()},
               cwd, env, opts.verbose);
  if (rc != 0) return rc;
  std::string fastCodegenPass = "--atir-fast-codegen";
#ifdef ANNC_ENABLE_KDNN_ADAPTOR
  fastCodegenPass = "--atir-fast-codegen=enable-kdnn=true";
#endif
  rc = runTool({anncAsm, aotModule.string(), fastCodegenPass,
                "--annc-aarch64-gemm-pipeline", "-o", loweredModule.string()},
               cwd, env, opts.verbose);
  if (rc != 0) return rc;
  rc = runTool({annc, loweredModule.string(), "--shared", "-o", kernelSo.string()},
               cwd, env, opts.verbose);
  if (rc != 0) return rc;

  dropIntermediates();
  std::cerr << "[annc-onnx-pipeline] " << siteCount << " site(s) -> "
            << kernelSo.string() << "\n";
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  Options opts;
  if (!parseOptions(argc, argv, &opts)) {
    printUsage();
    return 2;
  }
  if (opts.toolchainInfo) {
    printToolchainInfo(argv[0]);
    return 0;
  }
  return runPipeline(argc, argv, opts);
}
