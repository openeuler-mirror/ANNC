#include "annc_fused_op_jit.h"

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

namespace fs = std::filesystem;

bool EnvFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
         std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0;
}

std::string AnncToolPath(const char* tool_name) {
  constexpr char kDefaultPipelinePath[] = "/usr/local/bin/annc-tf-pipeline";
  const char* configured_pipeline = std::getenv("ANNC_PIPELINE_PATH");
  fs::path pipeline_path = configured_pipeline && configured_pipeline[0] != '\0'
                               ? configured_pipeline
                               : kDefaultPipelinePath;
  if (!pipeline_path.has_parent_path()) return tool_name;
  return (pipeline_path.parent_path() / tool_name).string();
}

Status CreateJitWorkDir(std::string* work_dir) {
  std::error_code ec;
  const char* configured_work_dir = std::getenv("ANNC_WORK_DIR");
  fs::path base = configured_work_dir && configured_work_dir[0] != '\0'
                      ? fs::path(configured_work_dir)
                      : fs::temp_directory_path(ec);
  if (ec) {
    return errors::Internal("Cannot resolve ANNC JIT temporary directory: ",
                            ec.message());
  }
  fs::create_directories(base, ec);
  if (ec) {
    return errors::Internal("Cannot create ANNC JIT work root ", base.string(),
                            ": ", ec.message());
  }

  std::string path_template = (base / "annc_jit_XXXXXX").string();
  std::vector<char> writable(path_template.begin(), path_template.end());
  writable.push_back('\0');
  char* created = mkdtemp(writable.data());
  if (!created) {
    return errors::Internal("Cannot create ANNC JIT work directory under ",
                            base.string(), ": ", std::strerror(errno));
  }
  *work_dir = created;
  return OkStatus();
}

Status WriteRuntimeShapeSpec(
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
    return errors::Internal("Cannot create ANNC JIT shape spec ", path, ": ",
                            std::strerror(errno));
  }
  output << json.dump(2) << '\n';
  output.close();
  if (!output) {
    return errors::Internal("Failed to write ANNC JIT shape spec ", path);
  }
  return OkStatus();
}

Status RunJitTool(const std::vector<std::string>& args, const char* tool_name) {
  if (args.empty() || args.front().empty()) {
    return errors::InvalidArgument("Missing executable for ", tool_name);
  }

  pid_t pid = fork();
  if (pid < 0) {
    return errors::Internal("Failed to fork ", tool_name, ": ",
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
    return errors::Internal("Failed to wait for ", tool_name, ": ",
                            std::strerror(errno));
  }
  if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return OkStatus();
  if (WIFEXITED(status)) {
    return errors::Internal(tool_name, " exited with code ",
                            WEXITSTATUS(status));
  }
  if (WIFSIGNALED(status)) {
    return errors::Internal(tool_name, " terminated by signal ",
                            WTERMSIG(status));
  }
  return errors::Internal(tool_name, " did not exit normally");
}

}  // namespace

static void CleanupAnncJitWorkDir(const std::string& work_dir) {
  if (work_dir.empty() || EnvFlagEnabled("ANNC_KEEP_TEMPS")) return;
  std::error_code ec;
  fs::remove_all(work_dir, ec);
  if (ec) {
    LOG(WARNING) << "[ANNC-JIT] failed to remove work directory " << work_dir
                 << ": " << ec.message();
  }
}

Status CompileAnncJitKernel(
    const AnncJitCompileRequest& request,
    std::shared_ptr<annc::jit::JitExecutable>* executable) {
  if (!executable) {
    return errors::InvalidArgument("Missing ANNC JIT executable output");
  }
  executable->reset();
  std::string work_dir;
  TF_RETURN_IF_ERROR(CreateJitWorkDir(&work_dir));

  fs::path work(work_dir);
  const std::string shape_spec = (work / "runtime_shapes.json").string();
  const std::string lowered_mlir = (work / "kernel_lowered.mlir").string();
  const std::string so_path = (work / "kernel.so").string();
  Status status = WriteRuntimeShapeSpec(shape_spec, request.kernel_name,
                                        request.argument_shapes);
  if (!status.ok()) {
    CleanupAnncJitWorkDir(work_dir);
    return status;
  }

  std::vector<std::string> asm_args = {
      AnncToolPath("annc-asm"),
      request.atir_module_path,
      "--atir-select-kernel=kernel-name=" + request.kernel_name,
      "--atir-specialize-shapes=shape-spec=" + shape_spec,
      "--atir-fast-codegen",
      "--annc-aarch64-gemm-pipeline",
      "-o",
      lowered_mlir,
  };
#ifdef ANNC_ENABLE_KDNN_ADAPTOR
  asm_args[4] = "--atir-fast-codegen=enable-kdnn=true";
#endif
  status = RunJitTool(asm_args, "annc-asm");
  if (!status.ok()) {
    CleanupAnncJitWorkDir(work_dir);
    return status;
  }

  status = RunJitTool(
      {AnncToolPath("annc"), lowered_mlir, "--shared", "-o", so_path}, "annc");
  if (!status.ok()) {
    CleanupAnncJitWorkDir(work_dir);
    return status;
  }

  void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    const char* error = dlerror();
    CleanupAnncJitWorkDir(work_dir);
    return errors::NotFound("Cannot load JIT library ", so_path, ": ",
                            error ? error : "<null>");
  }
  const std::string symbol_name = "_mlir_ciface_" + request.kernel_name;
  dlerror();
  void* kernel_function = dlsym(handle, symbol_name.c_str());
  const char* symbol_error = dlerror();
  if (symbol_error || !kernel_function) {
    dlclose(handle);
    CleanupAnncJitWorkDir(work_dir);
    return errors::NotFound("Cannot find symbol ", symbol_name, " in ",
                            so_path, ": ",
                            symbol_error ? symbol_error : "<null>");
  }
  void* set_thread_pool = dlsym(handle, "annc_set_current_threadpool");
  void* get_thread_pool = dlsym(handle, "annc_get_current_threadpool");
  if (!set_thread_pool || !get_thread_pool) {
    set_thread_pool = nullptr;
    get_thread_pool = nullptr;
  }
  *executable = std::make_shared<annc::jit::JitExecutable>(
      work_dir, so_path, handle, kernel_function, set_thread_pool,
      get_thread_pool, EnvFlagEnabled("ANNC_KEEP_TEMPS"));
  return OkStatus();
}

}  // namespace tensorflow
