#include "annc_optimizer.h"

#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <random>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "absl/strings/str_cat.h"

namespace tensorflow {
namespace grappler {

namespace {
// 默认配置常量
constexpr int kDefaultTimeoutSeconds = 300;
constexpr char kDefaultTempDir[] = "/tmp";
constexpr char kOutputPrefix[] = "tf_graph_output_";
}  // namespace

Status ANNCOptimizer::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  // 从配置中读取参数
  if (config == nullptr) {
    return errors::InvalidArgument("ANNCOptimizer config is null");
  }

  enabled_ = true;
  keep_temp_files_ = false;
  annc_verbose_ = false;
  timeout_seconds_ = kDefaultTimeoutSeconds;
  temp_dir_ = kDefaultTempDir;
  pipeline_path_ = "/usr/local/bin/annc-tf-pipeline";
  annc_work_dir_.clear();

  // 解析自定义配置参数
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
    }
  }

  LOG(INFO) << "ANNCOptimizer initialized: "
            << "pipeline_path=" << pipeline_path_
            << ", temp_dir=" << temp_dir_
            << ", annc_work_dir=" << annc_work_dir_
            << ", timeout=" << timeout_seconds_ << "s"
            << ", enabled=" << enabled_
            << ", annc_verbose=" << annc_verbose_;

  return OkStatus();
}

Status ANNCOptimizer::Optimize(Cluster* cluster,
                                  const GrapplerItem& grappler_item,
                                  GraphDef* output) {
  (void)cluster;

  // 检查是否启用
  if (!enabled_) {
    LOG(INFO) << "ANNCOptimizer is disabled, skipping optimization";
    *output = grappler_item.graph;
    return OkStatus();
  }

  LOG(INFO) << "Running ANNCOptimizer on graph: " << grappler_item.id;

  std::string input_graphdef_file = GenerateTempFilename("annc_input_graph_");
  std::string output_graphdef_file = GenerateTempFilename(kOutputPrefix);
  TF_RETURN_IF_ERROR(WriteGraphDefToFile(grappler_item.graph,
                                         &input_graphdef_file));

  Status status = InvokePipeline(input_graphdef_file, output_graphdef_file);
  if (!status.ok()) {
    LOG(WARNING) << "annc-tf-pipeline graph rewrite failed: " << status.message()
                 << ", returning original graph";
    *output = grappler_item.graph;
    if (!keep_temp_files_) {
      CleanupTempFiles({input_graphdef_file, output_graphdef_file});
    }
    return OkStatus();
  }

  status = ReadGraphDefFromFile(output_graphdef_file, output);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to read rewritten GraphDef from annc-tf-pipeline: "
                 << status.message() << ", returning original graph";
    *output = grappler_item.graph;
  } else {
    LOG(INFO) << "ANNCOptimizer loaded rewritten GraphDef from: "
              << output_graphdef_file;
  }

  if (!keep_temp_files_) {
    CleanupTempFiles({input_graphdef_file, output_graphdef_file});
  } else {
    LOG(INFO) << "Keeping ANNCOptimizer temp files: "
              << "\n  Input GraphDef: " << input_graphdef_file
              << "\n  Output GraphDef: " << output_graphdef_file;
  }

  return OkStatus();
}

std::string ANNCOptimizer::GenerateTempFilename(const std::string& prefix) {
  // 生成随机文件名
  auto now = std::chrono::system_clock::now();
  auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
      now.time_since_epoch()).count();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(100000, 999999);

  return absl::StrCat(temp_dir_, "/", prefix, timestamp, "_", dis(gen),
                         ".pb");
}

Status ANNCOptimizer::WriteGraphDefToFile(const GraphDef& graph_def,
                                             std::string* filepath) {
  // 将 GraphDef 序列化为字符串
  std::string graph_def_str;
  if (!graph_def.SerializeToString(&graph_def_str)) {
    return errors::Internal("Failed to serialize GraphDef to string");
  }

  // 写入文件
  Status write_status = WriteStringToFile(Env::Default(), *filepath,
                                          graph_def_str);
  if (!write_status.ok()) {
    return errors::Internal(absl::StrCat("Failed to write GraphDef to file: ",
                            write_status.message()));
  }

  LOG(INFO) << "Wrote GraphDef to file: " << *filepath
            << " (size: " << graph_def_str.size() << " bytes)";

  return OkStatus();
}

Status ANNCOptimizer::ReadGraphDefFromFile(const std::string& filepath,
                                              GraphDef* graph_def) {
  // 从文件读取
  std::string graph_def_str;
  Status read_status = ReadFileToString(Env::Default(), filepath,
                                        &graph_def_str);
  if (!read_status.ok()) {
    return errors::Internal(absl::StrCat("Failed to read GraphDef from file: ",
                            read_status.message()));
  }

  // 反序列化
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
    if (keep_temp_files_) {
      argv.push_back(const_cast<char*>("--keep_temps"));
    }
    if (annc_verbose_) {
      argv.push_back(const_cast<char*>("--verbose"));
    }
    argv.push_back(nullptr);

    execv(pipeline_path_.c_str(), argv.data());
    _exit(127);
  } else {
    // 父进程
    return WaitForProcess(pid, "annc-tf-pipeline");
  }
}

Status ANNCOptimizer::WaitForProcess(pid_t pid, const std::string& process_name) {
  int status;
  pid_t ret;

  // 等待子进程完成，带超时
  auto start = std::chrono::steady_clock::now();

  while ((ret = waitpid(pid, &status, WNOHANG)) == 0) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start).count();

    if (elapsed > timeout_seconds_) {
      LOG(ERROR) << process_name << " process timeout after " << elapsed
                 << " seconds, killing process";
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
      return errors::DeadlineExceeded(process_name + " process timeout");
    }

    usleep(100000);  // 100ms
  }

  if (ret < 0) {
    return errors::Internal(absl::StrCat("waitpid failed: ", strerror(errno)));
  }

  if (WIFEXITED(status)) {
    int exit_code = WEXITSTATUS(status);
    if (exit_code != 0) {
      return errors::Internal(absl::StrCat(process_name, " process exited with code: ",
                              exit_code));
    }
  } else if (WIFSIGNALED(status)) {
      return errors::Internal(absl::StrCat(process_name, " process terminated by signal: ",
                            WTERMSIG(status)));
  }

  LOG(INFO) << process_name << " process completed successfully";
  return OkStatus();
}

void ANNCOptimizer::CleanupTempFile(const std::string& filepath) {
  if (unlink(filepath.c_str()) != 0) {
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
