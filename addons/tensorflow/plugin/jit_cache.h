#ifndef ANNC_TENSORFLOW_PLUGIN_JIT_CACHE_H_
#define ANNC_TENSORFLOW_PLUGIN_JIT_CACHE_H_

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "cxx_api.h"

namespace annc {
namespace jit {

struct JitArgumentSignature {
  std::vector<int64_t> dims;
};

struct JitCacheKey {
  std::string digest;

  bool operator==(const JitCacheKey &other) const {
    return digest == other.digest;
  }
  bool operator!=(const JitCacheKey &other) const { return !(*this == other); }
};

bool BuildJitCacheKey(const std::string &templateFingerprint,
                      const std::vector<JitArgumentSignature> &arguments,
                      JitCacheKey *key, std::string *error);

class JitExecutable final {
 public:
  JitExecutable(annc::tensorflow::JitKernelPtr kernel,
                const AnncJitKernelResult &result);
  ~JitExecutable();

  JitExecutable(const JitExecutable &) = delete;
  JitExecutable &operator=(const JitExecutable &) = delete;

  const std::string &workDir() const { return work_dir_; }
  const std::string &sharedLibraryPath() const { return shared_library_path_; }
  void *kernelFunction() const { return kernel_function_; }
  void *setThreadPoolFunction() const { return set_thread_pool_function_; }
  void *getThreadPoolFunction() const { return get_thread_pool_function_; }

 private:
  // The custom deleter releases support's object through the C API, keeping
  // the loaded library alive for all cache and active-call shared_ptr owners.
  annc::tensorflow::JitKernelPtr kernel_;
  std::string work_dir_;
  std::string shared_library_path_;
  void *kernel_function_;
  void *set_thread_pool_function_;
  void *get_thread_pool_function_;
};

struct JitCompileResult {
  std::shared_ptr<JitExecutable> executable;
  std::string error;

  bool ok() const { return executable != nullptr && error.empty(); }
};

enum class CacheEvent {
  kMiss,
  kWait,
  kHit,
};

struct JitCacheLookup {
  std::shared_ptr<JitExecutable> executable;
  std::string error;
  CacheEvent event = CacheEvent::kMiss;
  std::vector<std::string> evictedKeys;

  bool ok() const { return executable != nullptr && error.empty(); }
};

struct JitCacheStats {
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t waits = 0;
  uint64_t compiles = 0;
  uint64_t errors = 0;
  uint64_t evictions = 0;
  size_t entries = 0;
};

class JitCompilationCache final {
 public:
  using CompileFunction = std::function<JitCompileResult()>;

  explicit JitCompilationCache(size_t capacity);
  JitCacheLookup GetOrCompile(const JitCacheKey &key,
                              const CompileFunction &compile);
  JitCacheStats stats() const;
  size_t size() const;

 private:
  struct Entry;

  void touchReadyEntry(const std::shared_ptr<Entry> &entry);

  size_t capacity_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<Entry>> entries_;
  std::list<std::string> lru_;
  JitCacheStats stats_;
};

}  // namespace jit
}  // namespace annc

#endif  // ANNC_TENSORFLOW_PLUGIN_JIT_CACHE_H_
