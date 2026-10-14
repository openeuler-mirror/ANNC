// annc_ep_compile.cc — invoking the ANNC compile driver and caching its
// output for the EP's runtime-compile mode (see annc_ep_compile.h).
//
// Everything here runs inside GetCapability, before any node is claimed: the
// EP must know which sites exist (and that their kernels are loadable) before
// it commits to them, because a failure in Compile() fails the whole session.
#include "annc_ep_compile.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include <nlohmann/json.hpp>

namespace annc_ep {
namespace {

constexpr const char* kDriverName = "annc-onnx-pipeline";
constexpr const char* kKernelLibraryName = "libannc_kernels.so";
constexpr const char* kReportName = "sites.json";
constexpr const char* kDoneMarker = "DONE";

std::string EnvString(const char* name, const std::string& fallback = "") {
  const char* value = std::getenv(name);
  return (value != nullptr && value[0] != '\0') ? std::string(value) : fallback;
}

bool EnvFlag(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

std::string CacheRoot() {
  std::string root = EnvString("ANNC_EP_CACHE_DIR");
  if (!root.empty()) return root;
  const std::string xdg = EnvString("XDG_CACHE_HOME");
  if (!xdg.empty()) return xdg + "/annc-ep";
  const std::string home = EnvString("HOME", "/tmp");
  return home + "/.cache/annc-ep";
}

int TimeoutSeconds() {
  const std::string value = EnvString("ANNC_EP_COMPILE_TIMEOUT_S", "600");
  const int seconds = std::atoi(value.c_str());
  return seconds > 0 ? seconds : 600;
}

bool FileExists(const std::string& path) {
  struct stat info {};
  return ::stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode);
}

bool DirectoryExists(const std::string& path) {
  struct stat info {};
  return ::stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

bool MakeDirs(const std::string& path) {
  if (path.empty()) return false;
  std::string current = (path[0] == '/') ? "/" : "";
  size_t start = 0;
  while (start < path.size()) {
    const size_t slash = path.find('/', start);
    const std::string part = path.substr(
        start, slash == std::string::npos ? std::string::npos : slash - start);
    if (!part.empty()) {
      if (current.empty()) current = part;
      else if (current == "/") current += part;
      else current += "/" + part;
      if (::mkdir(current.c_str(), 0777) != 0 && errno != EEXIST) return false;
    }
    if (slash == std::string::npos) break;
    start = slash + 1;
  }
  return DirectoryExists(path);
}

// The driver executable: explicit override first, then next to this shared
// library (the deployment ships them together), then PATH.
std::string FindDriver() {
  const std::string override = EnvString("ANNC_EP_PIPELINE");
  if (!override.empty()) return FileExists(override) ? override : std::string();

  Dl_info info {};
  if (::dladdr(reinterpret_cast<void*>(&FindDriver), &info) != 0 &&
      info.dli_fname != nullptr) {
    std::string dir(info.dli_fname);
    const size_t slash = dir.find_last_of('/');
    dir = (slash == std::string::npos) ? std::string(".") : dir.substr(0, slash);
    const std::string candidate = dir + "/" + kDriverName;
    if (FileExists(candidate)) return candidate;
    // An installed layout keeps the EP in lib/ and the tools in bin/.
    const std::string sibling = dir + "/../bin/" + kDriverName;
    if (FileExists(sibling)) return sibling;
  }

  const std::string path = EnvString("PATH");
  std::stringstream stream(path);
  std::string dir;
  while (std::getline(stream, dir, ':')) {
    if (dir.empty()) continue;
    const std::string candidate = dir + "/" + kDriverName;
    if (FileExists(candidate)) return candidate;
  }
  return {};
}

// Runs `argv` with the current environment, capturing stdout, killing the
// process group if it outlives the timeout. Returns the exit status, or -1.
int RunCaptured(const std::vector<std::string>& argv, int timeout_seconds,
                std::string* captured) {
  int pipe_fds[2];
  if (::pipe(pipe_fds) != 0) return -1;

  std::vector<char*> args;
  args.reserve(argv.size() + 1);
  for (const std::string& arg : argv) args.push_back(const_cast<char*>(arg.c_str()));
  args.push_back(nullptr);

  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(pipe_fds[0]);
    ::close(pipe_fds[1]);
    return -1;
  }
  if (pid == 0) {
    ::setpgid(0, 0);  // own group, so the timeout can kill the whole tree
    ::close(pipe_fds[0]);
    ::dup2(pipe_fds[1], STDOUT_FILENO);
    ::close(pipe_fds[1]);
    ::execvp(args[0], args.data());
    std::fprintf(stderr, "[annc-ep] cannot execute %s: %s\n", args[0], std::strerror(errno));
    ::_exit(127);
  }

  // Establish the process group from the parent too; the child can otherwise
  // race past setpgid before a timeout tries to signal the group.
  (void)::setpgid(pid, pid);
  ::close(pipe_fds[1]);
  const int flags = ::fcntl(pipe_fds[0], F_GETFL, 0);
  if (flags >= 0) (void)::fcntl(pipe_fds[0], F_SETFL, flags | O_NONBLOCK);
  std::string output;
  char buffer[4096];
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
  int status = 0;
  bool reaped = false;
  while (true) {
    const pid_t done = ::waitpid(pid, &status, WNOHANG);
    if (done == pid) {
      reaped = true;
      break;
    }
    if (done < 0 && errno != EINTR) break;
    ssize_t got = ::read(pipe_fds[0], buffer, sizeof(buffer));
    while (got > 0) {
      output.append(buffer, static_cast<size_t>(got));
      got = ::read(pipe_fds[0], buffer, sizeof(buffer));
    }
    if (std::chrono::steady_clock::now() > deadline) {
      ::kill(-pid, SIGKILL);
      ::waitpid(pid, &status, 0);
      reaped = true;
      status = -1;  // report a timeout, not a signal death
      break;
    }
    struct pollfd poll_fd {pipe_fds[0], POLLIN, 0};
    (void)::poll(&poll_fd, 1, 20);
  }
  // Drain whatever is left.
  ssize_t got = ::read(pipe_fds[0], buffer, sizeof(buffer));
  while (got > 0) {
    output.append(buffer, static_cast<size_t>(got));
    got = ::read(pipe_fds[0], buffer, sizeof(buffer));
  }
  ::close(pipe_fds[0]);
  if (captured != nullptr) *captured = output;
  if (!reaped) return -1;
  if (status == -1) return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}

bool ReadSite(const nlohmann::json& entry, Site* site) {
  if (!entry.is_object()) return false;
  site->pattern = entry.value("pattern", "");
  site->kernelName = entry.value("kernel_name", "");
  if (site->kernelName.empty()) return false;
  site->executionMode = entry.value("execution_mode", "aot");
  site->abi = entry.value("abi", "mlir_ciface");
  site->memberNodes = entry.value("member_node_names", std::vector<std::string>{});
  if (site->memberNodes.empty()) return false;

  auto readArgs = [](const nlohmann::json& value, std::vector<Site::Arg>* out) {
    if (!value.is_array()) return false;
    for (const auto& item : value) {
      if (!item.is_object()) return false;
      Site::Arg arg;
      arg.name = item.value("tf_name", "");
      arg.role = item.value("role", "");
      arg.dtype = item.value("dtype", "");
      arg.rank = item.value("rank", int64_t{-1});
      if (item.contains("shape") && item["shape"].is_array()) {
        for (const auto& dim : item["shape"]) {
          if (!dim.is_number_integer()) return false;
          arg.shape.push_back(dim.get<int64_t>());
        }
      }
      out->push_back(std::move(arg));
    }
    return true;
  };
  if (!entry.contains("args") || !readArgs(entry["args"], &site->args) ||
      !entry.contains("outputs") || !readArgs(entry["outputs"], &site->outputs)) {
    return false;
  }
  if (!site->outputs.empty()) site->outputTensor = site->outputs.front().name;
  if (entry.contains("args") && entry["args"].is_array()) {
    int position = 0;
    for (const auto& arg : entry["args"]) {
      if (!arg.is_object()) {
        ++position;
        continue;
      }
      const std::string role = arg.value("role", "");
      const int64_t rank = arg.value("rank", int64_t{-1});
      if (role == "fixed" && rank == 2 && site->weightIndex < 0) site->weightIndex = position;
      if (role == "dynamic" && rank == 3 && site->flattenXIndex < 0) site->flattenXIndex = position;
      ++position;
    }
  }
  if (entry.contains("kernel_arg_order") && entry["kernel_arg_order"].is_array()) {
    for (const auto& value : entry["kernel_arg_order"]) {
      if (!value.is_number_integer()) return false;
      site->kernelArgOrder.push_back(value.get<int64_t>());
    }
  }
  return true;
}

}  // namespace

std::string CacheEntryPath(const std::string& signature) {
  if (signature.size() < 2) return {};
  return CacheRoot() + "/v1/" + signature.substr(0, 2) + "/" + signature;
}

std::string ProcessTempDir() {
  static std::atomic<uint64_t> next_id{0};
  const std::string dir = CacheRoot() + "/tmp-" + std::to_string(::getpid()) +
                          "-" + std::to_string(next_id.fetch_add(1));
  MakeDirs(dir);
  return dir;
}

bool CompileDriver::LocateDriver() {
  if (located_) return available_;
  located_ = true;
  driver_path_ = FindDriver();
  if (driver_path_.empty()) {
    unavailable_reason_ = std::string("no ") + kDriverName +
                          " found (set ANNC_EP_PIPELINE or put it next to libannc_ep.so)";
    return false;
  }
  // ANNC_EP_COMPILE=0 disables the mode; unset means enabled.
  const char* enabled = std::getenv("ANNC_EP_COMPILE");
  if (enabled != nullptr && std::strcmp(enabled, "0") == 0) {
    unavailable_reason_ = "ANNC_EP_COMPILE=0";
    return false;
  }
  available_ = true;
  return true;
}

bool CompileDriver::Available() const {
  return const_cast<CompileDriver*>(this)->LocateDriver();
}

std::string CompileDriver::ToolchainInfo() {
  if (!toolchain_info_.empty()) return toolchain_info_;
  if (!Available()) return {};
  std::string output;
  const int rc = RunCaptured({driver_path_, "--toolchain-info"}, 60, &output);
  if (rc != 0 || output.empty()) return {};
  // Hash the tool's report so the cache key changes when any pipeline binary is
  // rebuilt (the report is per-tool size + mtime).
  toolchain_info_ = Sha256String(output);
  return toolchain_info_;
}

bool CompileDriver::ReadReport(const std::string& path, SiteReport* report,
                               std::string* error) const {
  std::ifstream in(path);
  if (!in.is_open()) {
    *error = "cannot open '" + path + "'";
    return false;
  }
  nlohmann::json document;
  try {
    in >> document;
  } catch (const std::exception& e) {
    *error = std::string("invalid site report: ") + e.what();
    return false;
  }
  try {
    const std::string schema = document.value("schema", "");
    if (schema != "annc-ep-site-report/1") {
      *error = "unexpected site report schema '" + schema + "'";
      return false;
    }
    report->sites.clear();
    if (!document.contains("sites") || !document["sites"].is_array()) {
      *error = "site report has no sites array";
      return false;
    }
    size_t index = 0;
    for (const auto& entry : document["sites"]) {
      Site site;
      const size_t entry_index = index++;
      if (!ReadSite(entry, &site)) {
        *error = "invalid site entry at index " + std::to_string(entry_index);
        return false;
      }
      std::string validation_error;
      if (!ValidateSite(site, &validation_error)) {
        // A dropped site silently costs its node(s) a claim: the run falls
        // back to the CPU EP with no trace of why. Say it out loud.
        std::fprintf(stderr, "[annc-ep] dropping site '%s': %s\n",
                     site.kernelName.c_str(), validation_error.c_str());
        continue;
      }
      report->sites.push_back(std::move(site));
    }
  } catch (const std::exception& e) {
    *error = std::string("invalid site report fields: ") + e.what();
    report->sites.clear();
    return false;
  }
  return true;
}

bool CompileDriver::Compile(const std::string& modelPath, const std::string& signature,
                            std::string* kernelLibrary, SiteReport* report,
                            std::string* error) {
  if (!Available()) {
    *error = unavailable_reason_;
    return false;
  }
  const std::string entry = CacheEntryPath(signature);
  if (entry.empty()) {
    *error = "empty model signature";
    return false;
  }
  const std::string reportPath = entry + "/" + kReportName;
  const std::string libraryPath = entry + "/" + kKernelLibraryName;
  const std::string donePath = entry + "/" + kDoneMarker;
  const std::string work_dir = entry + ".work";

  auto loadPublished = [&]() {
    if (!FileExists(donePath) || !FileExists(reportPath)) return false;
    SiteReport cached;
    std::string report_error;
    if (!ReadReport(reportPath, &cached, &report_error)) return false;
    // A zero-site model is a successful pipeline result and intentionally has
    // no shared library. Non-empty reports must have the library before the
    // marker is considered usable.
    if (!cached.sites.empty() && !FileExists(libraryPath)) return false;
    *report = std::move(cached);
    *kernelLibrary = report->sites.empty() ? std::string() : libraryPath;
    return true;
  };

  // Fast path: a published entry.
  if (loadPublished()) return true;

  // Cold path: one compiler per signature across processes. The lock file is
  // opened O_CLOEXEC -- the compile children are grandchildren of this process
  // and flock lives on the open file description, so an inherited fd would hold
  // the lock for as long as the child tree runs.
  if (!MakeDirs(entry.substr(0, entry.find_last_of('/')))) {
    *error = "cannot create cache shard directory for '" + entry + "'";
    return false;
  }
  const std::string lockPath = entry + ".lock";
  const int lock_fd = ::open(lockPath.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
  if (lock_fd < 0) {
    *error = "cannot open cache lock '" + lockPath + "'";
    return false;
  }
  if (::flock(lock_fd, LOCK_EX) != 0) {
    ::close(lock_fd);
    *error = "cannot lock cache entry";
    return false;
  }
  // Re-check under the lock: another process may have published meanwhile.
  if (loadPublished()) {
    ::flock(lock_fd, LOCK_UN);
    ::close(lock_fd);
    return true;
  }

  bool ok = false;
  std::string stage_error;
  do {
    // A crashed prior compiler may have left a partial work tree behind.
    // Start each attempt from an empty directory so stale artifacts cannot be
    // mistaken for outputs from this invocation.
    {
      std::error_code ignored;
      std::filesystem::remove_all(work_dir, ignored);
    }
    if (!MakeDirs(work_dir)) {
      stage_error = "cannot create work dir '" + work_dir + "'";
      break;
    }
    std::string output;
    const int rc = RunCaptured({driver_path_, "--model", modelPath, "--out-dir", work_dir},
                               TimeoutSeconds(), &output);
    if (rc != 0) {
      stage_error = "compile driver failed (exit " + std::to_string(rc) + ")";
      const size_t keep = 400;
      if (output.size() > keep) output.erase(0, output.size() - keep);
      if (!output.empty()) stage_error += ": " + output;
      break;
    }
    std::string report_error;
    if (!ReadReport(work_dir + "/" + kReportName, report, &report_error)) {
      stage_error = report_error;
      break;
    }
    if (!report->sites.empty() &&
        !FileExists(work_dir + "/" + kKernelLibraryName)) {
      stage_error = "compile driver reported sites without a kernel library";
      break;
    }
    // Publish atomically: clear any partial entry, rename the work dir into
    // place, then write the marker readers check for.
    {
      std::error_code ignored;
      std::filesystem::remove_all(entry, ignored);
    }
    if (::rename(work_dir.c_str(), entry.c_str()) != 0) {
      stage_error = "cannot publish cache entry '" + entry + "': " + std::strerror(errno);
      break;
    }
    std::ofstream done(donePath);
    if (done) done << "ok\n";
    done.close();
    if (!done) {
      stage_error = "cannot publish cache marker '" + donePath + "'";
      break;
    }
    *kernelLibrary = report->sites.empty() ? std::string() : libraryPath;
    ok = true;
  } while (false);

  // Cleanup must remain under the signature lock: the next compiler can
  // recreate this same directory as soon as the lock is released.
  if (!ok && !EnvFlag("ANNC_EP_KEEP_TEMPS")) {
    std::error_code ignored;
    std::filesystem::remove_all(work_dir, ignored);
  }
  ::flock(lock_fd, LOCK_UN);
  ::close(lock_fd);
  if (!ok) {
    *error = stage_error;
    return false;
  }
  return true;
}

}  // namespace annc_ep
