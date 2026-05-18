#include "Dialect/Atir/OpVerify/runKpGenOp.h"
#include "Dialect/Atir/OpVerify/GenericMemRef.h"
#include "Dialect/Atir/OpVerify/IoDef.h"
#include "Dialect/Atir/AtirOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <dlfcn.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <cmath>
#include <chrono>
#include <memory>
#include <functional>

using namespace llvm;
using namespace mlir;

namespace atir {

// RAII wrapper for dynamic library handle
struct DlHandle {
  void* handle = nullptr;
  
  explicit DlHandle(const std::string &path) {
    handle = dlopen(path.c_str(), RTLD_NOW);
    if (!handle) {
      llvm::errs() << "dlopen failed: " << dlerror() << "\n";
    }
  }
  
  ~DlHandle() { if (handle) dlclose(handle); }
  
  DlHandle(const DlHandle&) = delete;
  DlHandle& operator=(const DlHandle&) = delete;
  
  void* getSymbol(const std::string &symbolName) {
    if (!handle) return nullptr;
    void* sym = dlsym(handle, symbolName.c_str());
    if (!sym) {
      llvm::errs() << "dlsym failed for '" << symbolName << "': " << dlerror() << "\n";
    }
    return sym;
  }
};

// RAII wrapper for malloc'd return descriptor buffer
struct RetDescBuffer {
  void* ptr = nullptr;
  
  ~RetDescBuffer() { if (ptr) free(ptr); }
  
  RetDescBuffer() = default;
  RetDescBuffer(const RetDescBuffer&) = delete;
  RetDescBuffer& operator=(const RetDescBuffer&) = delete;
  
  // Transfer ownership
  RetDescBuffer(RetDescBuffer&& other) noexcept : ptr(other.ptr) {
    other.ptr = nullptr;
  }
  RetDescBuffer& operator=(RetDescBuffer&& other) noexcept {
    std::swap(ptr, other.ptr);
    return *this;
  }
  
  void* release() {
    void* p = ptr;
    ptr = nullptr;
    return p;
  }
};

// Build input MemRef list
static std::optional<std::vector<GenericMemRef>> buildInputMemRefs(const IoTensorDef* inputs) {
  std::vector<GenericMemRef> memrefs;
  
  size_t inputIdx = 0;
  for (const auto& tensorData : inputs->getInputs()) {
    auto type = tensorData.second.getType();
    if (auto atirType = llvm::dyn_cast<atir::TensorType>(type)) {
      auto memref = createGenericMemRef(atirType.getCacheData());
      if (!memref) {
        llvm::errs() << "Failed to create memref for input operand\n";
        return std::nullopt;
      }
      memrefs.push_back(std::move(*memref));
      inputIdx++;
    }
  }
  
  return memrefs;
}

// Build output MemRef list
static std::optional<std::vector<GenericMemRef>> buildOutputMemRefs(const IoTensorDef* outputs) {
  std::vector<GenericMemRef> memrefs;
  
  for (size_t i = 0; i < outputs->getOutputs().size(); ++i) {
    const auto& tensorData = outputs->getOutputs()[i];
    auto type = tensorData.second.getType();
    if (auto atirType = llvm::dyn_cast<atir::TensorType>(type)) {
      auto cacheData = atirType.getCacheData();
      auto memref = createGenericMemRef(atirType.getCacheData());
      if (!memref) {
        llvm::errs() << "Failed to create memref for output operand\n";
        return std::nullopt;
      }
      memrefs.push_back(std::move(*memref));
    }
  }
  
  return memrefs;
}

// Build argument list (MLIR C Interface convention: return descriptor + input parameters)
// The first parameter is a buffer to store complete memref descriptors for all return values
static std::vector<void*> buildArgumentList(
    const std::vector<GenericMemRef>& outputMemrefs,
    const std::vector<GenericMemRef>& inputMemrefs) {
  std::vector<void*> args;
  
  // The MLIR C interface packs multiple return values into a single C struct,
  // where each StridedMemRef descriptor uses its actual rank (not maxRank).
  // We must allocate the buffer to match this packed struct layout exactly.
  if (!outputMemrefs.empty()) {
    size_t retDescSize = 0;
    for (const auto& memref : outputMemrefs) {
      size_t rank = 0;
      if (memref.descriptor.size() >= 3) {
        rank = (memref.descriptor.size() - 3) / 2;
      }
      // Each descriptor: 2 pointers + offset + sizes[rank] + strides[rank]
      retDescSize += sizeof(void*) * 2 + sizeof(int64_t) * (1 + 2 * rank);
    }
    void* retDescBuffer = malloc(retDescSize);
    memset(retDescBuffer, 0, retDescSize);
    args.push_back(retDescBuffer);
  }

  for (const auto& memref : inputMemrefs) {
    args.push_back(const_cast<int64_t*>(memref.descriptor.data()));
  }
  
  return args;
}

static bool invokeKernel(void* funcPtr, const std::vector<void*>& args) {
  if (args.empty()) {
    llvm::errs() << "No arguments provided for kernel invocation\n";
    return false;
  }
  
  // For MLIR C Interface, call with different signatures based on argument count
  switch (args.size()) {
    case 1: reinterpret_cast<void(*)(void*)>(funcPtr)(args[0]); break;
    case 2: reinterpret_cast<void(*)(void*, void*)>(funcPtr)(args[0], args[1]); break;
    case 3: reinterpret_cast<void(*)(void*, void*, void*)>(funcPtr)(args[0], args[1], args[2]); break;
    case 4: reinterpret_cast<void(*)(void*, void*, void*, void*)>(funcPtr)(args[0], args[1], args[2], args[3]); break;
    case 5: reinterpret_cast<void(*)(void*, void*, void*, void*, void*)>(funcPtr)(args[0], args[1], args[2], args[3], args[4]); break;
    case 6: reinterpret_cast<void(*)(void*, void*, void*, void*, void*, void*)>(funcPtr)(args[0], args[1], args[2], args[3], args[4], args[5]); break;
    case 7: reinterpret_cast<void(*)(void*, void*, void*, void*, void*, void*, void*)>(funcPtr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6]); break;
    case 8: reinterpret_cast<void(*)(void*, void*, void*, void*, void*, void*, void*, void*)>(funcPtr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]); break;
    case 9: reinterpret_cast<void(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*)>(funcPtr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]); break;
    case 10: reinterpret_cast<void(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(funcPtr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]); break;
    case 11: reinterpret_cast<void(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(funcPtr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]); break;
    case 12: reinterpret_cast<void(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(funcPtr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11]); break;
    case 13: reinterpret_cast<void(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(funcPtr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]); break;
    case 14: reinterpret_cast<void(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(funcPtr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13]); break;
    case 15: reinterpret_cast<void(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(funcPtr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14]); break;
    case 16: reinterpret_cast<void(*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*)>(funcPtr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15]); break;
    default:
      llvm::errs() << "Unsupported argument count: " << args.size() << " (max 16)\n";
      return false;
  }
  return true;
}

// Single measurement function - measures time only
static double measureOnce(std::function<bool()> kernelFunc) {
  auto startTime = std::chrono::high_resolution_clock::now();
  bool success = kernelFunc();
  auto endTime = std::chrono::high_resolution_clock::now();
  if (!success) return -1.0;
  return std::chrono::duration<double, std::milli>(endTime - startTime).count();
}

// Performance profiler
// TODO: Add more performance metrics
class PerformanceProfiler {
public:
  struct Config {
    int totalRuns = 100;
  };
  
  explicit PerformanceProfiler(const Config& cfg) : config_(cfg) {}
  
  void recordRun(double timeMs) {
    allTimes_.push_back(timeMs);
  }
  
  PerformanceMetrics getMetrics(const std::string& kernelPath) const {
    if (allTimes_.empty()) {
      return PerformanceMetrics(kernelPath, 0.0);
    }
    
    // Calculate average time only
    double avgTime = std::accumulate(allTimes_.begin(), allTimes_.end(), 0.0) / allTimes_.size();
    
    return PerformanceMetrics(kernelPath, avgTime);
  }
  
  int getRunCount() const { return (int)allTimes_.size(); }
  bool shouldStop() const { return (int)allTimes_.size() >= config_.totalRuns; }

private:
  Config config_;
  std::vector<double> allTimes_;
};

// Extract all output results to outputResult
// Extract all output results from the return descriptor buffer written by the kernel.
// The kernel reuses the data pointers from our pre-allocated outputMemrefs (via the
// return descriptor), so we do NOT free the pointers in the descriptor — they point
// to memory owned by outputMemrefs' DataBuffer RAII.
static bool extractAllOutputsToAttributes(
    void* retDescBuffer,
    const std::vector<GenericMemRef>& outputMemrefs,
    const IoTensorDef* outputTemplate,
    IoTensorDef* outputResult) {
  
  if (outputMemrefs.empty()) {
    llvm::outs() << "[Info] No outputs to extract\n";
    return true;
  }
  
  if (outputMemrefs.size() != outputTemplate->getOutputCount()) {
    llvm::errs() << "Error: output count mismatch (expected " << outputTemplate->getOutputCount()
                << ", got " << outputMemrefs.size() << ")\n";
    return false;
  }

  // Compute per-output offsets matching the packed struct layout (must match buildArgumentList)
  // The MLIR C interface uses each output's actual rank, not a uniform maxRank.
  auto* descBase = static_cast<char*>(retDescBuffer);
  auto getDescOffset = [&](size_t idx) -> size_t {
    size_t offset = 0;
    for (size_t j = 0; j < idx; ++j) {
      size_t rank = 0;
      if (outputMemrefs[j].descriptor.size() >= 3) {
        rank = (outputMemrefs[j].descriptor.size() - 3) / 2;
      }
      offset += sizeof(void*) * 2 + sizeof(int64_t) * (1 + 2 * rank);
    }
    return offset;
  };
  
  for (size_t i = 0; i < outputMemrefs.size(); ++i) {
    const auto& outMemRef = outputMemrefs[i];
    auto outValue = outputTemplate->getOutputs()[i].second;
    auto outAtirType = llvm::cast<atir::TensorType>(outValue.getType());
    auto outShape = outAtirType.getShape();
    auto outElemType = outAtirType.getElementType();
    
    size_t elemBytes = getElementBytes(outElemType);
    size_t numElements = 1;
    for (auto dim : outShape) numElements *= dim;
    size_t totalBytes = numElements * elemBytes;

    if (descBase) {
      // Parse descriptor from return descriptor buffer using per-output offset
      char* desc = descBase + getDescOffset(i);
      void* alignedPtr = *reinterpret_cast<void**>(desc + sizeof(void*));

      if (alignedPtr && totalBytes > 0 && alignedPtr != outMemRef.data_buf.ptr) {
        // Kernel allocated its own memory — copy to our buffer
        memcpy(outMemRef.data_buf.ptr, alignedPtr, totalBytes);
      }
    }
    
    auto rankedTensorType = mlir::RankedTensorType::get(outShape, outElemType);
    auto denseAttr = extractResultToAttr(outMemRef, rankedTensorType);

    outAtirType.setCacheData(denseAttr);
    outputResult->addOutput(outputTemplate->getOutputs()[i].first, outValue);
  }
  
  return true;
}

bool runKpGenOp(const std::string &libPath, 
                const IoTensorDef *inputs,
                const IoTensorDef *outputTemplate,
                IoTensorDef *outputResult,
                PerformanceMetrics &metrics,
                const WarmupConfig &config) {
  
  DlHandle handle(libPath);
  if (!handle.handle) {
    return false;
  }
  
  llvm::dbgs() << "Input count: " << inputs->getInputCount() 
              << ", Output count: " << outputTemplate->getOutputCount() << "\n";
  
  auto funcName = "_mlir_ciface_" + getFuncNameFromPath(libPath);
  auto funcPtr = handle.getSymbol(funcName);
  if (!funcPtr) {
    return false;
  }
  
  // Build input and output MemRefs
  auto inputMemrefsOpt = buildInputMemRefs(inputs);
  if (!inputMemrefsOpt) {
    llvm::errs() << "buildInputMemRefs failed\n";
    return false;
  }
  auto& inputMemrefs = *inputMemrefsOpt;
  
  // Create output MemRef using outputTemplate (contains return value definitions)
  auto outputMemrefsOpt = buildOutputMemRefs(outputTemplate);
  if (!outputMemrefsOpt) {
    llvm::errs() << "[Error] buildOutputMemRefs failed\n";
    return false;
  }
  auto& outputMemrefs = *outputMemrefsOpt;
  
  // Build argument list (outputs first, then inputs)
  auto args = buildArgumentList(outputMemrefs, inputMemrefs);
  
  if (args.empty()) {
    llvm::errs() << "No arguments for kernel invocation\n";
    return false;
  }
  
  // Performance profiling with adaptive warmup
  std::vector<double> warmupTimes;
  warmupTimes.reserve(config.maxWarmupRuns);
  
  // Adaptive warmup loop
  int warmupIter = 0;
  bool stable = false;
  
  llvm::outs() << "[Profiling] Starting adaptive warmup...\n";
  
  while (warmupIter < config.maxWarmupRuns) {
    // Rebuild input and output MemRefs for each warmup iteration
    auto iterationInputMemrefs = buildInputMemRefs(inputs);
    if (!iterationInputMemrefs) {
      llvm::errs() << "buildInputMemRefs failed in warmup iteration " << (warmupIter + 1) << "\n";
      return false;
    }
    
    auto iterationOutputMemrefs = buildOutputMemRefs(outputTemplate);
    if (!iterationOutputMemrefs) {
      llvm::errs() << "buildOutputMemRefs failed in warmup iteration " << (warmupIter + 1) << "\n";
      return false;
    }
    
    auto warmupArgs = buildArgumentList(*iterationOutputMemrefs, *iterationInputMemrefs);
    
    double timeMs = measureOnce([&]() -> bool {
      bool success = invokeKernel(funcPtr, warmupArgs);
      return success;
    });
    
    // Free return descriptor buffer
    if (!warmupArgs.empty() && warmupArgs[0]) {
      free(warmupArgs[0]);
    }
    
    if (timeMs < 0) {
      llvm::errs() << "Kernel execution failed during warmup run " << (warmupIter + 1) << "\n";
      return false;
    }
    
    warmupTimes.push_back(timeMs);
    warmupIter++;
    
    // Check if we have enough runs for stability check
    if (warmupIter >= config.minWarmupRuns) {
      // Get last 5 runs
      double minTime = warmupTimes[warmupIter - config.minWarmupRuns];
      double maxTime = warmupTimes[warmupIter - config.minWarmupRuns];
      for (int j = 1; j < config.minWarmupRuns; ++j) {
        double t = warmupTimes[warmupIter - config.minWarmupRuns + j];
        if (t < minTime) minTime = t;
        if (t > maxTime) maxTime = t;
      }
      
      // Scheme B: Check if max-min difference is within threshold
      double avgTime = 0.0;
      for (int j = 0; j < config.minWarmupRuns; ++j) {
        avgTime += warmupTimes[warmupIter - config.minWarmupRuns + j];
      }
      avgTime /= config.minWarmupRuns;
      
      double diff = maxTime - minTime;
      double ratio = (avgTime > 0) ? (diff / avgTime) : 0.0;
      
      if (ratio <= config.stabilityThreshold) {
        stable = true;
        llvm::outs() << "[Profiling] Warmup stable after " << warmupIter << " iterations\n";
        llvm::outs() << "[Profiling] Last " << config.minWarmupRuns << " runs: "
                    << "min=" << minTime << "ms, max=" << maxTime << "ms, "
                    << "avg=" << avgTime << "ms, diff_ratio=" << (ratio * 100) << "%\n";
        break;
      }
    }
  }
  
  if (!stable) {
    llvm::outs() << "[Profiling] Warmup reached max iterations (" << config.maxWarmupRuns << "), proceeding with statistics\n";
    if (warmupIter >= config.minWarmupRuns) {
      double minTime = warmupTimes[warmupIter - config.minWarmupRuns];
      double maxTime = warmupTimes[warmupIter - config.minWarmupRuns];
      for (int j = 1; j < config.minWarmupRuns; ++j) {
        double t = warmupTimes[warmupIter - config.minWarmupRuns + j];
        if (t < minTime) minTime = t;
        if (t > maxTime) maxTime = t;
      }
      llvm::outs() << "[Profiling] Last " << config.minWarmupRuns << " runs: "
                  << "min=" << minTime << "ms, max=" << maxTime << "ms\n";
    }
  }
  
  // Run m iterations for statistics (starting from n+1)
  PerformanceProfiler::Config profilerConfig;
  profilerConfig.totalRuns = config.statRuns;
  PerformanceProfiler profiler(profilerConfig);
  
  llvm::outs() << "[Profiling] Running " << config.statRuns << " iterations for statistics...\n";
  
  // Run statistics iterations
  for (int i = 0; i < config.statRuns; ++i) {
    // Rebuild input and output MemRefs for each iteration to ensure fresh buffers
    auto iterationInputMemrefs = buildInputMemRefs(inputs);
    if (!iterationInputMemrefs) {
      llvm::errs() << "buildInputMemRefs failed in iteration " << (i + 1) << "\n";
      return false;
    }
    
    auto iterationOutputMemrefs = buildOutputMemRefs(outputTemplate);
    if (!iterationOutputMemrefs) {
      llvm::errs() << "buildOutputMemRefs failed in iteration " << (i + 1) << "\n";
      return false;
    }
    
    auto profilingArgs = buildArgumentList(*iterationOutputMemrefs, *iterationInputMemrefs);
    
    double timeMs = measureOnce([&]() -> bool {
      bool success = invokeKernel(funcPtr, profilingArgs);
      
      // Profiling iteration completed (no per-iteration output to reduce noise)
      return success;
    });
    
    // Free return descriptor buffer (RAII will handle it)
    // profilingArgs[0] is malloc'd buffer, needs to be freed
    if (!profilingArgs.empty() && profilingArgs[0]) {
      free(profilingArgs[0]);
    }
    
    if (timeMs < 0) {
      llvm::errs() << "Kernel execution failed during profiling run " << (i + 1) << "\n";
      return false;
    }
    
    profiler.recordRun(timeMs);
  }
  
  // Get performance metrics
  metrics = profiler.getMetrics(libPath);
  
  llvm::outs() << "[Perf] Avg time: " << metrics.execTimeMs << " ms\n";
  
  // === Single run for result extraction ===
  // Rebuild input and output MemRefs for result extraction
  auto resultInputMemrefs = buildInputMemRefs(inputs);
  if (!resultInputMemrefs) {
    llvm::errs() << "buildInputMemRefs failed for result extraction\n";
    return false;
  }
  
  auto resultOutputMemrefs = buildOutputMemRefs(outputTemplate);
  if (!resultOutputMemrefs) {
    llvm::errs() << "buildOutputMemRefs failed for result extraction\n";
    return false;
  }
  
  RetDescBuffer retDescBuf;
  auto resultArgs = buildArgumentList(*resultOutputMemrefs, *resultInputMemrefs);
  
  if (!invokeKernel(funcPtr, resultArgs)) {
    llvm::errs() << "Kernel execution failed during result extraction\n";
    return false;
  }
  
  // Save return descriptor for result extraction
  if (!resultArgs.empty()) {
    retDescBuf.ptr = resultArgs[0];
    resultArgs[0] = nullptr;
  }
  
  // Extract results
  if (!extractAllOutputsToAttributes(retDescBuf.ptr, *resultOutputMemrefs, outputTemplate, outputResult)) {
    return false;
  }
  
  // retDescBuf is automatically freed by RAII
  
  return true;
}

}
