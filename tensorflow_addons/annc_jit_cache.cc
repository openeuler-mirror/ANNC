#include "annc_jit_cache.h"

#include <dlfcn.h>

#include <cstring>
#include <exception>
#include <filesystem>
#include <utility>

namespace annc::jit {
namespace {

namespace fs = std::filesystem;

// Key layout, version 4: the digest IS the exact serialized identity
// (schema tag, length-prefixed fingerprint, fixed-width argument dims).
// Byte-exact map equality means distinct compilations can never share a
// digest, unlike the previous SHA-256 hex scheme which cost ~3.5-4.4 us per
// call in ostringstream + hashing and dominated the multi-shape hot path.
constexpr char kKeySchema[] = "annc-jit-cache-key-v4";

void AppendU64(std::string &output, uint64_t value) {
  char buffer[sizeof(uint64_t)];
  std::memcpy(buffer, &value, sizeof(value));
  output.append(buffer, sizeof(buffer));
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
  size_t dimensionCount = 0;
  for (const JitArgumentSignature &argument : arguments) {
    dimensionCount += argument.dims.size();
    for (int64_t dimension : argument.dims) {
      if (dimension < 0) {
        *error = "runtime argument shape contains a negative dimension";
        return false;
      }
    }
  }

  std::string &serialized = key->digest;
  serialized.reserve(
      sizeof(kKeySchema) - 1 + templateFingerprint.size() +
      sizeof(uint64_t) * (2 + arguments.size() + dimensionCount));
  serialized.append(kKeySchema);
  AppendU64(serialized, templateFingerprint.size());
  serialized.append(templateFingerprint);
  AppendU64(serialized, arguments.size());
  for (const JitArgumentSignature &argument : arguments) {
    AppendU64(serialized, argument.dims.size());
    for (int64_t dimension : argument.dims) {
      AppendU64(serialized, static_cast<uint64_t>(dimension));
    }
  }
  return true;
}

std::string CacheKeySummary(const JitCacheKey &key) {
  // FNV-1a over the whole digest: the digest starts with the constant schema
  // tag, so hex-encoding a prefix would print the same value for every key.
  static constexpr char kHex[] = "0123456789abcdef";
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : key.digest) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  std::string summary;
  summary.reserve(16);
  for (int shift = 60; shift >= 0; shift -= 4) {
    summary.push_back(kHex[(hash >> shift) & 0xf]);
  }
  return summary;
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
  if (entry->inLru) {
    // Move the existing node instead of erase + push_front, which allocated
    // and freed a list node (plus copied the key) on every hot cache hit.
    lru_.splice(lru_.begin(), lru_, entry->lruPosition);
  } else {
    lru_.push_front(entry->key);
    entry->inLru = true;
  }
  entry->lruPosition = lru_.begin();
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
