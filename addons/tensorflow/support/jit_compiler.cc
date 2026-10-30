#include "jit_compiler.h"

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <vector>

namespace annc {
namespace jit {
namespace {

namespace fs = std::filesystem;

template <typename... Args>
std::string ErrorMessage(const Args&... args) {
  std::ostringstream message;
  (message << ... << args);
  return message.str();
}

void LogJitMessage(AnncJitLogCallback log, void* context, int warning,
                   const std::string& message) {
  if (log) log(context, warning, message.c_str());
}

bool EnvFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
         std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0;
}

bool AnncVerboseEnabled() { return EnvFlagEnabled("ANNC_VERBOSE"); }

std::string AnncToolPath(const char* tool_name) {
  constexpr char kDefaultPipelinePath[] = "/usr/local/bin/annc-tf-pipeline";
  const char* configured_pipeline = std::getenv("ANNC_PIPELINE_PATH");
  fs::path pipeline_path = configured_pipeline && configured_pipeline[0] != '\0'
                               ? configured_pipeline
                               : kDefaultPipelinePath;
  if (!pipeline_path.has_parent_path()) return tool_name;
  return (pipeline_path.parent_path() / tool_name).string();
}

std::string CreateJitWorkDir(std::string* work_dir) {
  std::error_code ec;
  const char* configured_work_dir = std::getenv("ANNC_WORK_DIR");
  fs::path base = configured_work_dir && configured_work_dir[0] != '\0'
                      ? fs::path(configured_work_dir)
                      : fs::temp_directory_path(ec);
  if (ec) {
    return ErrorMessage("Cannot resolve ANNC JIT temporary directory: ",
                        ec.message());
  }
  fs::create_directories(base, ec);
  if (ec) {
    return ErrorMessage("Cannot create ANNC JIT work root ", base.string(),
                        ": ", ec.message());
  }

  std::string path_template = (base / "annc_jit_XXXXXX").string();
  std::vector<char> writable(path_template.begin(), path_template.end());
  writable.push_back('\0');
  char* created = mkdtemp(writable.data());
  if (!created) {
    return ErrorMessage("Cannot create ANNC JIT work directory under ",
                        base.string(), ": ", std::strerror(errno));
  }
  *work_dir = created;
  return "";
}

std::string WriteRuntimeShapeSpec(
    const std::string& path, const std::string& kernel_name,
    const std::vector<std::vector<int64_t>>& argument_shapes) {
  nlohmann::json json;
  json["kernel_name"] = kernel_name;
  json["arguments"] = nlohmann::json::array();
  for (size_t i = 0; i < argument_shapes.size(); ++i) {
    json["arguments"].push_back({{"index", i}, {"shape", argument_shapes[i]}});
  }

  std::ofstream output(path);
  if (!output.is_open()) {
    return ErrorMessage("Cannot create ANNC JIT shape spec ", path, ": ",
                        std::strerror(errno));
  }
  output << json.dump(2) << '\n';
  output.close();
  if (!output) {
    return ErrorMessage("Failed to write ANNC JIT shape spec ", path);
  }
  return "";
}

std::string RunJitTool(const std::vector<std::string>& args,
                       const char* tool_name, AnncJitLogCallback log,
                       void* log_context, AnncJitStatus* result_status) {
  *result_status = ANNC_JIT_INTERNAL;
  if (args.empty() || args.front().empty()) {
    *result_status = ANNC_JIT_INVALID_ARGUMENT;
    return ErrorMessage("Missing executable for ", tool_name);
  }

  if (AnncVerboseEnabled()) {
    LogJitMessage(log, log_context, 0,
                  ErrorMessage("[ANNC-JIT-TRACE] ", tool_name,
                               " begin executable=", args.front(),
                               " argc=", args.size()));
  }

  pid_t pid = fork();
  if (pid < 0) {
    return ErrorMessage("Failed to fork ", tool_name, ": ",
                        std::strerror(errno));
  }
  if (pid == 0) {
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (const std::string& arg : args) {
      argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);
    unsetenv("LD_PRELOAD");
    execvp(argv[0], argv.data());
    _exit(errno == ENOENT ? 127 : 126);
  }

  int status = 0;
  pid_t waited;
  do {
    waited = waitpid(pid, &status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited < 0) {
    return ErrorMessage("Failed to wait for ", tool_name, ": ",
                        std::strerror(errno));
  }
  if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
    *result_status = ANNC_JIT_OK;
    return "";
  }
  if (WIFEXITED(status)) {
    return ErrorMessage(tool_name, " exited with code ", WEXITSTATUS(status));
  }
  if (WIFSIGNALED(status)) {
    return ErrorMessage(tool_name, " terminated by signal ", WTERMSIG(status));
  }
  return ErrorMessage(tool_name, " did not exit normally");
}

}  // namespace

static void CleanupAnncJitWorkDir(std::string& work_dir,
                                  AnncJitLogCallback log, void* log_context) {
  if (work_dir.empty() || EnvFlagEnabled("ANNC_KEEP_TEMPS")) return;
  std::error_code ec;
  fs::remove_all(work_dir, ec);
  if (ec) {
    LogJitMessage(log, log_context, 1,
                  ErrorMessage("[ANNC-JIT] failed to remove work directory ",
                               work_dir, ": ", ec.message()));
  }
  // Failure cleanup runs while the borrowed log callback is valid; the handle
  // destructor must not repeat it after returning to the caller.
  work_dir.clear();
}

AnncJitStatus CompileAnncJitKernel(const JitCompileRequest& request,
                                   AnncJitKernel* kernel) {
  kernel->keep_temps = EnvFlagEnabled("ANNC_KEEP_TEMPS");
  std::string& work_dir = kernel->work_dir;
  std::string& error = kernel->error;
  error = CreateJitWorkDir(&work_dir);
  if (!error.empty()) return ANNC_JIT_INTERNAL;

  if (AnncVerboseEnabled()) {
    LogJitMessage(request.log, request.log_context, 0,
                  ErrorMessage("[ANNC-JIT-TRACE] work directory created path=",
                               work_dir, " kernel=", request.kernel_name));
  }

  fs::path work(work_dir);
  const std::string shape_spec = (work / "runtime_shapes.json").string();
  const std::string lowered_mlir = (work / "kernel_lowered.mlir").string();
  const std::string packed_rhs_c = (work / "kernel_packed_rhs.c").string();
  const std::string so_path = (work / "kernel.so").string();
  error = WriteRuntimeShapeSpec(shape_spec, request.kernel_name,
                                request.argument_shapes);
  if (!error.empty()) {
    CleanupAnncJitWorkDir(work_dir, request.log, request.log_context);
    return ANNC_JIT_INTERNAL;
  }
  if (AnncVerboseEnabled()) {
    LogJitMessage(
        request.log, request.log_context, 0,
        ErrorMessage(
            "[ANNC-JIT-TRACE] runtime shape spec written path=", shape_spec,
            " arguments=", request.argument_shapes.size()));
  }

  std::string gemm_pipeline_arg = "--annc-aarch64-gemm-pipeline";
#ifdef ANNC_ENABLE_KDNN_ADAPTOR
  const std::string fast_codegen_arg = "--atir-fast-codegen=enable-kdnn=true";
#else
  const std::string fast_codegen_arg = "--atir-fast-codegen";
#endif
  std::string gemm_pipeline_options;
#ifdef ANNC_ENABLE_CONSTANT_FOLDING
  gemm_pipeline_options += "packed-c=" + packed_rhs_c;
#endif
  if (request.intra_thread_count > 0) {
    if (!gemm_pipeline_options.empty()) gemm_pipeline_options += " ";
    gemm_pipeline_options +=
        "intra-thread-count=" + std::to_string(request.intra_thread_count);
  }
  if (!gemm_pipeline_options.empty()) {
    gemm_pipeline_arg += "=" + gemm_pipeline_options;
  }
  std::vector<std::string> asm_args = {
      AnncToolPath("annc-asm"),
      request.atir_module_path,
      "--atir-select-kernel=kernel-name=" + request.kernel_name,
      "--atir-specialize-shapes=shape-spec=" + shape_spec,
      fast_codegen_arg,
      gemm_pipeline_arg,
      "-o",
      lowered_mlir,
  };
  AnncJitStatus status;
  error = RunJitTool(asm_args, "annc-asm", request.log, request.log_context,
                     &status);
  if (!error.empty()) {
    CleanupAnncJitWorkDir(work_dir, request.log, request.log_context);
    return status;
  }
  if (AnncVerboseEnabled()) {
    LogJitMessage(request.log, request.log_context, 0,
                  ErrorMessage("[ANNC-JIT-TRACE] annc-asm completed output=",
                               lowered_mlir));
  }

  std::vector<std::string> link_args = {
      AnncToolPath("annc"), lowered_mlir, "--shared", "-o", so_path};
#ifdef ANNC_ENABLE_CONSTANT_FOLDING
  link_args.push_back("--packed-rhs-c");
  link_args.push_back(packed_rhs_c);
#endif
  error = RunJitTool(link_args, "annc", request.log, request.log_context, &status);
  if (!error.empty()) {
    CleanupAnncJitWorkDir(work_dir, request.log, request.log_context);
    return status;
  }
  if (AnncVerboseEnabled()) {
    LogJitMessage(
        request.log, request.log_context, 0,
        ErrorMessage("[ANNC-JIT-TRACE] annc completed output=", so_path));
  }

  if (AnncVerboseEnabled()) {
    LogJitMessage(request.log, request.log_context, 0,
                  ErrorMessage("[ANNC-JIT-TRACE] dlopen begin path=", so_path));
  }
  kernel->shared_library_path = so_path;
  void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  kernel->library_handle = handle;
  if (!handle) {
    const char* detail = dlerror();
    error = ErrorMessage("Cannot load JIT library ", so_path, ": ",
                         detail ? detail : "<null>");
    CleanupAnncJitWorkDir(work_dir, request.log, request.log_context);
    return ANNC_JIT_NOT_FOUND;
  }
  if (AnncVerboseEnabled()) {
    LogJitMessage(request.log, request.log_context, 0,
                  ErrorMessage("[ANNC-JIT-TRACE] dlopen completed path=",
                               so_path, " handle=", handle));
  }
  const std::string symbol_name = "_mlir_ciface_" + request.kernel_name;
  dlerror();
  void* kernel_function = dlsym(handle, symbol_name.c_str());
  const char* symbol_error = dlerror();
  if (symbol_error || !kernel_function) {
    error = ErrorMessage("Cannot find symbol ", symbol_name, " in ", so_path,
                         ": ", symbol_error ? symbol_error : "<null>");
    dlclose(handle);
    kernel->library_handle = nullptr;
    CleanupAnncJitWorkDir(work_dir, request.log, request.log_context);
    return ANNC_JIT_NOT_FOUND;
  }
  if (AnncVerboseEnabled()) {
    LogJitMessage(
        request.log, request.log_context, 0,
        ErrorMessage("[ANNC-JIT-TRACE] kernel symbol resolved symbol=",
                     symbol_name, " address=", kernel_function));
  }
  void* set_thread_pool = dlsym(handle, "annc_set_current_threadpool");
  void* get_thread_pool = dlsym(handle, "annc_get_current_threadpool");
  if (!set_thread_pool || !get_thread_pool) {
    set_thread_pool = nullptr;
    get_thread_pool = nullptr;
  }
  kernel->kernel_function = kernel_function;
  kernel->set_thread_pool_function = set_thread_pool;
  kernel->get_thread_pool_function = get_thread_pool;
  return ANNC_JIT_OK;
}

}  // namespace jit
}  // namespace annc

AnncJitKernel::~AnncJitKernel() {
  if (library_handle) dlclose(library_handle);
  if (!keep_temps && !work_dir.empty()) {
    std::error_code error;
    std::filesystem::remove_all(work_dir, error);
  }
}
