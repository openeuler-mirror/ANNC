#include "annc_optimizer.h"

#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "absl/strings/str_cat.h"

namespace tensorflow {
namespace grappler {

namespace {

bool ReadSavedModelGraphDef(const std::string& path, GraphDef* graph_def) {
  std::string pb_path;
  if (path.size() > 3 && path.substr(path.size() - 3) == ".pb") {
    pb_path = path;
  } else {
    pb_path = path + "/saved_model.pb";
  }

  std::string data;
  Status s = ReadFileToString(Env::Default(), pb_path, &data);
  if (!s.ok()) {
    LOG(WARNING) << "ANNC: cannot read SavedModel: " << pb_path
                 << ": " << s.message();
    return false;
  }

  SavedModel saved_model;
  if (!saved_model.ParseFromString(data)) {
    LOG(WARNING) << "ANNC: cannot parse SavedModel: " << pb_path;
    return false;
  }

  if (saved_model.meta_graphs_size() == 0) {
    LOG(WARNING) << "ANNC: SavedModel has no MetaGraphDefs: " << pb_path;
    return false;
  }

  *graph_def = saved_model.meta_graphs(0).graph_def();
  LOG(INFO) << "ANNC: extracted GraphDef from SavedModel: " << pb_path
            << " (" << graph_def->node_size() << " nodes)";
  return true;
}

constexpr char kEnvEnable[] = "ANNC_ENABLE";
constexpr char kEnvPipelinePath[] = "ANNC_PIPELINE_PATH";
constexpr char kEnvWorkDir[] = "ANNC_WORK_DIR";
constexpr char kEnvBackend[] = "ANNC_BACKEND";
constexpr char kEnvVerbose[] = "ANNC_VERBOSE";
constexpr char kEnvTimeout[] = "ANNC_TIMEOUT";
constexpr char kEnvKeepTemps[] = "ANNC_KEEP_TEMPS";
constexpr char kEnvFusedOpPath[] = "ANNC_FUSED_OP_PATH";
constexpr char kEnvSavedModelPath[] = "ANNC_SAVEDMODEL_PATH";

constexpr int kDefaultTimeoutSeconds = 300;
constexpr char kDefaultTempDir[] = "/tmp";
constexpr char kDefaultPipelinePath[] = "/usr/local/bin/annc-tf-pipeline";
constexpr char kOutputPrefix[] = "tf_graph_output_";

void AnncOptimizerLibraryAnchor() {}

bool GetEnvFlag(const char* name) {
  const char* val = getenv(name);
  if (!val) return false;
  return (val[0] == '1' || val[0] == 't' || val[0] == 'T' ||
          val[0] == 'y' || val[0] == 'Y');
}

std::string GetEnvStr(const char* name, const std::string& default_val = "") {
  const char* val = getenv(name);
  return val ? std::string(val) : default_val;
}

int GetEnvInt(const char* name, int default_val = 0) {
  const char* val = getenv(name);
  if (!val) return default_val;
  return std::stoi(val);
}

}  // namespace

Status ANNCOptimizer::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  enabled_ = true;
  keep_temp_files_ = false;
  annc_verbose_ = false;
  timeout_seconds_ = kDefaultTimeoutSeconds;
  temp_dir_ = kDefaultTempDir;
  pipeline_path_ = kDefaultPipelinePath;
  annc_work_dir_.clear();

  if (GetEnvFlag(kEnvEnable)) {
    enabled_ = true;
  }

  const std::string env_pipeline = GetEnvStr(kEnvPipelinePath);
  if (!env_pipeline.empty()) {
    pipeline_path_ = env_pipeline;
  }

  const std::string env_work = GetEnvStr(kEnvWorkDir);
  if (!env_work.empty()) {
    annc_work_dir_ = env_work;
  }

  if (GetEnvFlag(kEnvVerbose)) {
    annc_verbose_ = true;
  }

  int env_timeout = GetEnvInt(kEnvTimeout);
  if (env_timeout > 0) {
    timeout_seconds_ = env_timeout;
  }

  if (GetEnvFlag(kEnvKeepTemps)) {
    keep_temp_files_ = true;
  } else {
    const char* env_keep = getenv(kEnvKeepTemps);
    if (env_keep && (env_keep[0] == '0' || env_keep[0] == 'f' || env_keep[0] == 'F')) {
      keep_temp_files_ = false;
    }
  }

  const std::string env_backend = GetEnvStr(kEnvBackend);
  if (!env_backend.empty()) {
    backend_ = env_backend;
  }

  const std::string env_sm = GetEnvStr(kEnvSavedModelPath);
  if (!env_sm.empty()) {
    savedmodel_path_ = env_sm;
  }

  if (config != nullptr) {
    for (const auto& param : config->parameter_map()) {
      const std::string& name = param.first;
      const std::string& value = param.second.s();

      if (name == "pipeline_path" || name == "annc_pipeline_path") {
        pipeline_path_ = value;
      } else if (name == "work_dir" || name == "annc_work_dir") {
        annc_work_dir_ = value;
      } else if (name == "temp_dir") {
        temp_dir_ = value;
      } else if (name == "timeout_seconds") {
        timeout_seconds_ = std::stoi(value);
      } else if (name == "keep_temp_files") {
        keep_temp_files_ = (value == "true" || value == "1");
      } else if (name == "annc_verbose" || name == "verbose") {
        annc_verbose_ = (value == "true" || value == "1");
      } else if (name == "enabled") {
        enabled_ = (value == "true" || value == "1");
      } else if (name == "batch_size") {
        batch_size_ = std::stoll(value);
      } else if (name == "savedmodel_path") {
        savedmodel_path_ = value;
      } else if (name == "backend") {
        backend_ = value;
      }
    }
  }

  LoadFusedOpLibrary();

  LOG(INFO) << "ANNCOptimizer initialized: "
            << "pipeline_path=" << pipeline_path_
            << ", temp_dir=" << temp_dir_
            << ", annc_work_dir=" << annc_work_dir_
            << ", timeout=" << timeout_seconds_ << "s"
            << ", enabled=" << enabled_
            << ", keep_temp_files=" << keep_temp_files_
            << ", annc_verbose=" << annc_verbose_
            << ", batch_size=" << batch_size_
            << ", savedmodel_path=" << savedmodel_path_
            << ", backend=" << (backend_.empty() ? "generic" : backend_);

  return OkStatus();
}

void ANNCOptimizer::LoadFusedOpLibrary() {
  static bool loaded = false;
  if (loaded) return;

  const char* env_path = getenv(kEnvFusedOpPath);
  std::string so_path;
  if (env_path && env_path[0]) {
    so_path = env_path;
  } else {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&AnncOptimizerLibraryAnchor), &info)) {
      std::string self_dir(info.dli_fname);
      auto pos = self_dir.rfind('/');
      if (pos != std::string::npos) {
        so_path = self_dir.substr(0, pos + 1) + "libannc_fused_op.so";
      }
    }
  }

  if (so_path.empty()) {
    so_path = "libannc_fused_op.so";
  }

  void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle) {
    LOG(INFO) << "ANNCOptimizer loaded fused op library: " << so_path;
    loaded = true;
  } else {
    LOG(WARNING) << "ANNCOptimizer could not load fused op library: "
                 << so_path << ": " << dlerror();
  }
}

Status ANNCOptimizer::Optimize(Cluster* cluster,
                                  const GrapplerItem& grappler_item,
                                  GraphDef* output) {
  (void)cluster;

  if (!enabled_) {
    LOG(INFO) << "ANNCOptimizer is disabled, skipping optimization";
    *output = grappler_item.graph;
    return OkStatus();
  }

  LOG(INFO) << "Running ANNCOptimizer on graph: " << grappler_item.id;

  GraphDef input_graph;

  if (!savedmodel_path_.empty()) {
    LOG(INFO) << "Reading SavedModel from: " << savedmodel_path_;
    if (!ReadSavedModelGraphDef(savedmodel_path_, &input_graph)) {
      LOG(WARNING) << "Failed to read SavedModel, falling back to Grappler graph";
      input_graph = grappler_item.graph;
    }
  } else {
    input_graph = grappler_item.graph;
  }

  LOG(INFO) << "Input graph has " << input_graph.node_size() << " nodes";

  std::string input_graphdef_file = GenerateTempFilename("annc_input_graph_");
  std::string output_graphdef_file = GenerateTempFilename(kOutputPrefix);

  auto cleanup_temps = [&]() {
    if (!keep_temp_files_) {
      CleanupTempFiles({input_graphdef_file, output_graphdef_file});
    }
  };

  Status write_status = WriteGraphDefToFile(input_graph, &input_graphdef_file);
  if (!write_status.ok()) {
    LOG(WARNING) << "Failed to write input GraphDef: " << write_status.message();
    cleanup_temps();
    *output = grappler_item.graph;
    return OkStatus();
  }

  Status status = InvokePipeline(input_graphdef_file, output_graphdef_file);
  if (!status.ok()) {
    LOG(WARNING) << "annc-tf-pipeline graph rewrite failed: " << status.message()
                 << ", returning original graph";
    *output = grappler_item.graph;
    cleanup_temps();
    return OkStatus();
  }

  status = ReadGraphDefFromFile(output_graphdef_file, output);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to read rewritten GraphDef from annc-tf-pipeline: "
                 << status.message() << ", returning original graph";
    *output = grappler_item.graph;
  } else {
    for (int i = 0; i < output->node_size(); ++i) {
      auto* node = output->mutable_node(i);
      if (node->device().empty()) {
        node->set_device("/job:localhost/replica:0/task:0/device:CPU:0");
      }
    }
    LOG(INFO) << "ANNCOptimizer graph rewrite completed successfully, "
              << "node count: " << output->node_size();
  }

  cleanup_temps();
  return OkStatus();
}

Status ANNCOptimizer::WriteGraphDefToFile(const GraphDef& graph_def,
                                             std::string* filepath) {
  std::string graph_def_str;
  if (!graph_def.SerializeToString(&graph_def_str)) {
    return errors::Internal("Failed to serialize GraphDef");
  }

  int fd = open(filepath->c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) {
    return errors::Internal(absl::StrCat("Failed to open file for writing: ",
                            *filepath, ": ", strerror(errno)));
  }

  ssize_t written = write(fd, graph_def_str.data(), graph_def_str.size());
  close(fd);

  if (written != static_cast<ssize_t>(graph_def_str.size())) {
    return errors::Internal(absl::StrCat("Failed to write GraphDef to file: ",
                            *filepath));
  }

  LOG(INFO) << "Wrote GraphDef to file: " << *filepath
            << " (size: " << graph_def_str.size() << " bytes)";

  return OkStatus();
}

Status ANNCOptimizer::ReadGraphDefFromFile(const std::string& filepath,
                                              GraphDef* graph_def) {
  std::string graph_def_str;

  int fd = open(filepath.c_str(), O_RDONLY);
  if (fd < 0) {
    return errors::Internal(absl::StrCat("Failed to open file for reading: ",
                            filepath, ": ", strerror(errno)));
  }

  const size_t kBufSize = 4096;
  char buf[kBufSize];
  ssize_t n;
  while ((n = read(fd, buf, kBufSize)) > 0) {
    graph_def_str.append(buf, n);
  }
  close(fd);

  if (n < 0) {
    return errors::Internal(absl::StrCat("Failed to read GraphDef from file: ",
                            filepath));
  }

  if (!graph_def->ParseFromString(graph_def_str)) {
    return errors::Internal(absl::StrCat("Failed to parse GraphDef from file: ", filepath));
  }

  LOG(INFO) << "Read GraphDef from file: " << filepath
            << " (size: " << graph_def_str.size() << " bytes)";

  return OkStatus();
}

Status ANNCOptimizer::InvokePipeline(const std::string& input_file,
                                     const std::string& output_file) {
  LOG(INFO) << "Invoking annc-tf-pipeline graph rewrite: " << pipeline_path_
            << " with input=" << input_file
            << ", output=" << output_file;

  pid_t pid = fork();

  if (pid < 0) {
    return errors::Internal(absl::StrCat("Failed to fork child process: ",
                            strerror(errno)));
  } else if (pid == 0) {
    if (!annc_verbose_) {
      int null_fd = open("/dev/null", O_RDWR);
      if (null_fd >= 0) {
        dup2(null_fd, STDOUT_FILENO);
        dup2(null_fd, STDERR_FILENO);
        close(null_fd);
      }
    }

    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(pipeline_path_.c_str()));
    argv.push_back(const_cast<char*>("--input_graphdef"));
    argv.push_back(const_cast<char*>(input_file.c_str()));
    argv.push_back(const_cast<char*>("--output_graphdef"));
    argv.push_back(const_cast<char*>(output_file.c_str()));
    if (!annc_work_dir_.empty()) {
      argv.push_back(const_cast<char*>("--work_dir"));
      argv.push_back(const_cast<char*>(annc_work_dir_.c_str()));
    }
    if (keep_temp_files_ || !annc_work_dir_.empty()) {
      argv.push_back(const_cast<char*>("--keep_temps"));
    }
    if (annc_verbose_) {
      argv.push_back(const_cast<char*>("--verbose"));
    }
    if (batch_size_ > 0) {
      static std::string batch_size_str;
      batch_size_str = std::to_string(batch_size_);
      argv.push_back(const_cast<char*>("--batch_size"));
      argv.push_back(const_cast<char*>(batch_size_str.c_str()));
    }
    if (!backend_.empty()) {
      argv.push_back(const_cast<char*>("--backend"));
      argv.push_back(const_cast<char*>(backend_.c_str()));
    }
    argv.push_back(nullptr);

    unsetenv("LD_PRELOAD");
    execv(pipeline_path_.c_str(), argv.data());
    _exit(127);
  } else {
    return WaitForProcess(pid, "annc-tf-pipeline");
  }
}

Status ANNCOptimizer::WaitForProcess(pid_t pid, const std::string& process_name) {
  int status;
  pid_t ret;

  auto start = std::chrono::steady_clock::now();

  while ((ret = waitpid(pid, &status, WNOHANG)) == 0) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start).count();
    if (elapsed > timeout_seconds_) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      return errors::Internal(absl::StrCat(process_name,
                              " timed out after ", timeout_seconds_, "s"));
    }
    usleep(100000);
  }

  if (ret < 0) {
    return errors::Internal(absl::StrCat("waitpid failed for ", process_name,
                            ": ", strerror(errno)));
  }

  if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
    return OkStatus();
  }

  int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  return errors::Internal(absl::StrCat(process_name,
                          " process exited with code: ", exit_code));
}

std::string ANNCOptimizer::GenerateTempFilename(const std::string& prefix) {
  static std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<uint64_t> dist;
  uint64_t rand_val = dist(rng);
  auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  return absl::StrCat(temp_dir_, "/", prefix, now, "_", rand_val, ".pb");
}

void ANNCOptimizer::CleanupTempFile(const std::string& filepath) {
  if (unlink(filepath.c_str()) != 0 && errno != ENOENT) {
    LOG(WARNING) << "Failed to delete temp file: " << filepath
                 << ", error: " << strerror(errno);
  } else {
    LOG(INFO) << "Deleted temp file: " << filepath;
  }
}

void ANNCOptimizer::CleanupTempFiles(const std::vector<std::string>& filepaths) {
  for (const auto& filepath : filepaths) {
    CleanupTempFile(filepath);
  }
}

}  // namespace grappler
}  // namespace tensorflow
