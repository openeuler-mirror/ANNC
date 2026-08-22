#include "annc_jit_cache.h"

#include <dlfcn.h>
#include <openssl/evp.h>

#include <array>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <utility>

namespace annc::jit {
namespace {

namespace fs = std::filesystem;

constexpr char kKeySchema[] = "annc-jit-cache-key-v3";

void AppendField(std::ostringstream &output, const std::string &name,
                 const std::string &value) {
  output << name.size() << ':' << name << value.size() << ':' << value;
}

std::string Sha256Hex(const std::string &content) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context ||
      EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), content.data(), content.size()) != 1) {
    return "";
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digestSize = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digestSize) != 1) {
    return "";
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < digestSize; ++i) {
    output << std::setw(2) << static_cast<unsigned int>(digest[i]);
  }
  return output.str();
}

JitCompileResult InvokeCompile(const JitCompilationCache::CompileFunction &compile) {
  try {
    return compile();
  } catch (const std::exception &error) {
    return {nullptr,
            std::string("JIT compilation callback threw: ") + error.what()};
  } catch (...) {
    return {nullptr, "JIT compilation callback threw an unknown exception"};
  }
}

}  // namespace

bool BuildJitCacheKey(const std::string &templateFingerprint,
                      const std::vector<JitArgumentSignature> &arguments,
                      JitCacheKey *key, std::string *error) {
  if (!key || !error) return false;
  key->digest.clear();
  error->clear();
  if (templateFingerprint.empty()) {
    *error = "template fingerprint is empty";
    return false;
  }
  for (const JitArgumentSignature &argument : arguments) {
    for (int64_t dimension : argument.dims) {
      if (dimension < 0) {
        *error = "runtime argument shape contains a negative dimension";
        return false;
      }
    }
  }

  std::ostringstream serialized;
  AppendField(serialized, "schema", kKeySchema);
  AppendField(serialized, "template", templateFingerprint);
  AppendField(serialized, "argument_count", std::to_string(arguments.size()));
  for (size_t i = 0; i < arguments.size(); ++i) {
    const JitArgumentSignature &argument = arguments[i];
    AppendField(serialized, "argument_index", std::to_string(i));
    AppendField(serialized, "rank", std::to_string(argument.dims.size()));
    for (int64_t dimension : argument.dims) {
      AppendField(serialized, "dimension", std::to_string(dimension));
    }
  }
  key->digest = Sha256Hex(serialized.str());
  if (key->digest.empty()) {
    *error = "cannot hash JIT cache key inputs";
    return false;
  }
  return true;
}

JitExecutable::JitExecutable(std::string workDir, std::string sharedLibraryPath,
                             void *libraryHandle, void *kernelFunction,
                             void *setThreadPoolFunction,
                             void *getThreadPoolFunction, bool keepTemps)
    : work_dir_(std::move(workDir)),
      shared_library_path_(std::move(sharedLibraryPath)),
      library_handle_(libraryHandle),
      kernel_function_(kernelFunction),
      set_thread_pool_function_(setThreadPoolFunction),
      get_thread_pool_function_(getThreadPoolFunction),
      keep_temps_(keepTemps) {}

JitExecutable::~JitExecutable() {
  if (library_handle_) dlclose(library_handle_);
  if (!keep_temps_ && !work_dir_.empty()) {
    std::error_code error;
    fs::remove_all(work_dir_, error);
  }
}

struct JitCompilationCache::Entry {
  enum class State {
    kCompiling,
    kReady,
    kFailed,
  };

  explicit Entry(std::string cacheKey) : key(std::move(cacheKey)) {}

  std::string key;
  std::mutex mutex;
  std::condition_variable condition;
  State state = State::kCompiling;
  std::shared_ptr<JitExecutable> executable;
  std::string error;
  std::list<std::string>::iterator lruPosition;
  bool inLru = false;
};

JitCompilationCache::JitCompilationCache(size_t capacity)
    : capacity_(capacity) {}

void JitCompilationCache::touchReadyEntry(const std::shared_ptr<Entry> &entry) {
  if (entry->inLru) lru_.erase(entry->lruPosition);
  lru_.push_front(entry->key);
  entry->lruPosition = lru_.begin();
  entry->inLru = true;
}

JitCacheLookup JitCompilationCache::GetOrCompile(
    const JitCacheKey &key, const CompileFunction &compile) {
  if (key.digest.empty()) {
    return JitCacheLookup{nullptr, "JIT cache key is empty", CacheEvent::kMiss};
  }
  if (capacity_ == 0) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++stats_.misses;
      ++stats_.compiles;
    }
    JitCompileResult result = InvokeCompile(compile);
    if (!result.ok()) {
      std::lock_guard<std::mutex> lock(mutex_);
      ++stats_.errors;
    }
    return JitCacheLookup{std::move(result.executable), std::move(result.error),
                          CacheEvent::kMiss};
  }

  std::shared_ptr<Entry> entry;
  bool owner = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = entries_.find(key.digest);
    if (found == entries_.end()) {
      entry = std::make_shared<Entry>(key.digest);
      entries_.emplace(key.digest, entry);
      ++stats_.misses;
      ++stats_.compiles;
      owner = true;
    } else {
      entry = found->second;
      if (entry->state == Entry::State::kReady) {
        ++stats_.hits;
        touchReadyEntry(entry);
        return JitCacheLookup{entry->executable, "", CacheEvent::kHit};
      }
      ++stats_.waits;
    }
  }

  if (!owner) {
    std::unique_lock<std::mutex> entryLock(entry->mutex);
    entry->condition.wait(
        entryLock, [&] { return entry->state != Entry::State::kCompiling; });
    if (entry->state == Entry::State::kReady) {
      return JitCacheLookup{entry->executable, "", CacheEvent::kWait};
    }
    return JitCacheLookup{nullptr, entry->error, CacheEvent::kWait};
  }

  JitCompileResult result = InvokeCompile(compile);
  std::vector<std::shared_ptr<Entry>> evicted;
  std::vector<std::string> evictedKeys;
  bool ready = false;
  std::string failure;
  std::shared_ptr<JitExecutable> executable;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::lock_guard<std::mutex> entryLock(entry->mutex);
    if (result.ok()) {
      entry->state = Entry::State::kReady;
      entry->executable = result.executable;
      executable = entry->executable;
      ready = true;
      touchReadyEntry(entry);
      while (lru_.size() > capacity_) {
        std::string evictedKey = lru_.back();
        lru_.pop_back();
        auto found = entries_.find(evictedKey);
        if (found == entries_.end()) continue;
        found->second->inLru = false;
        evictedKeys.push_back(evictedKey);
        evicted.push_back(found->second);
        entries_.erase(found);
        ++stats_.evictions;
      }
    } else {
      entry->state = Entry::State::kFailed;
      entry->error = result.error.empty() ? "JIT compilation failed"
                                          : std::move(result.error);
      failure = entry->error;
      entries_.erase(key.digest);
      ++stats_.errors;
    }
  }
  entry->condition.notify_all();

  if (ready) {
    return JitCacheLookup{std::move(executable), "", CacheEvent::kMiss,
                          std::move(evictedKeys)};
  }
  return JitCacheLookup{nullptr, std::move(failure), CacheEvent::kMiss};
}

JitCacheStats JitCompilationCache::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  JitCacheStats snapshot = stats_;
  snapshot.entries = entries_.size();
  return snapshot;
}

size_t JitCompilationCache::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return entries_.size();
}

}  // namespace annc::jit
