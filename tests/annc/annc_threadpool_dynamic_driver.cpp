#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Kernel/MemRefTypes.h"
#include "Kernel/threadpool/ThreadPool.h"

using KernelFn = void (*)(AnncMemRef2DF32*, AnncMemRef2DF32*, AnncMemRef2DF32*);
using SetThreadPoolFn = void (*)(annc::threadpool::AnncThreadPool*);
using GetThreadPoolFn = annc::threadpool::AnncThreadPool* (*)();
using ClearThreadPoolFn = void (*)();

struct DriverConfig {
    std::string soPath = "./demo.so";
    std::int64_t m = 128;
    std::int64_t k = 64;
    std::int64_t n = 128;
    int threads = 4;
    float postAdd = 0.0f;
    bool relu = false;
};

struct FakePoolTrace {
    int parallelForCalls = 0;
    std::vector<std::pair<std::int64_t, std::int64_t>> chunks;
    std::optional<std::int64_t> lastCostPerUnit;
    std::optional<std::int64_t> lastGrainSize;
    std::string dispatchPath;

    void reset() {
        parallelForCalls = 0;
        chunks.clear();
        lastCostPerUnit.reset();
        lastGrainSize.reset();
        dispatchPath.clear();
    }
};

struct FakeTfThreadPool {
    int numThreads = 4;
    int currentThreadId = -1;
    bool inParallel = false;
    FakePoolTrace trace;

    int NumThreads() const {
        return numThreads;
    }

    int CurrentThreadId() const {
        return currentThreadId;
    }

    bool InParallelRegion() const {
        return inParallel;
    }

    void ParallelFor(std::int64_t total,
                     const std::function<void(std::int64_t, std::int64_t)>& fn) {
        trace.dispatchPath = "tf_default";
        trace.lastCostPerUnit.reset();
        trace.lastGrainSize = 1;
        dispatch(total, 1, fn);
    }

    void ParallelForAdaptive(std::int64_t total,
                             std::int64_t costPerUnit,
                             const std::function<void(std::int64_t, std::int64_t)>& fn) {
        trace.dispatchPath = "tf_cost";
        trace.lastCostPerUnit = costPerUnit;
        trace.lastGrainSize.reset();
        const std::int64_t adaptiveBlock =
            std::max<std::int64_t>(1, total / std::max(numThreads, 1));
        dispatch(total, adaptiveBlock, fn);
    }

    void ParallelForFixed(std::int64_t total,
                          std::int64_t blockSize,
                          const std::function<void(std::int64_t, std::int64_t)>& fn) {
        trace.dispatchPath = "tf_fixed";
        trace.lastCostPerUnit.reset();
        trace.lastGrainSize = blockSize;
        dispatch(total, blockSize, fn);
    }

private:
    void dispatch(std::int64_t total,
                  std::int64_t blockSize,
                  const std::function<void(std::int64_t, std::int64_t)>& fn) {
        trace.parallelForCalls++;

        const std::int64_t chunk = std::max<std::int64_t>(blockSize, 1);
        currentThreadId = 0;
        inParallel = true;
        for (std::int64_t begin = 0; begin < total; begin += chunk) {
            const std::int64_t end = std::min(begin + chunk, total);
            trace.chunks.emplace_back(begin, end);
            fn(begin, end);
        }
        inParallel = false;
        currentThreadId = -1;
    }
};

struct FakeTfThreadPoolAdapter : public annc::threadpool::AnncThreadPool {
    explicit FakeTfThreadPoolAdapter(FakeTfThreadPool* pool) : pool(pool) {}

    int num_threads() const override {
        return pool->NumThreads();
    }

    bool in_parallel_region() const override {
        return pool->InParallelRegion();
    }

    void parallel_for(
        std::int64_t total,
        const annc::threadpool::ParallelForOptions& options,
        const std::function<void(std::int64_t, std::int64_t)>& fn) override {
        if (options.cost_per_unit.has_value()) {
            pool->ParallelForAdaptive(total, *options.cost_per_unit, fn);
            return;
        }
        if (options.grain_size.has_value()) {
            pool->ParallelForFixed(total, *options.grain_size, fn);
            return;
        }
        pool->ParallelFor(total, fn);
    }

private:
    FakeTfThreadPool* pool;
};

struct PreparedInputs {
    std::vector<float> lhs;
    std::vector<float> rhs;
    std::vector<float> expected;
};

void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "  --so <path>         Shared library path\n"
        << "  --m <value>         MatMul M dimension\n"
        << "  --k <value>         MatMul K dimension\n"
        << "  --n <value>         MatMul N dimension\n"
        << "  --threads <value>   Fake frontend thread count\n"
        << "  --post-add <value>  Add a scalar after matmul when building expected output\n"
        << "  --relu              Apply relu after matmul when building expected output\n";
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::int64_t parseI64(const char* text, const char* flag) {
    char* end = nullptr;
    long long value = std::strtoll(text, &end, 10);
    require(end != text && *end == '\0', std::string("invalid integer for ") + flag + ": " + text);
    return static_cast<std::int64_t>(value);
}

int parseInt(const char* text, const char* flag) {
    return static_cast<int>(parseI64(text, flag));
}

float parseFloat(const char* text, const char* flag) {
    char* end = nullptr;
    float value = std::strtof(text, &end);
    require(end != text && *end == '\0', std::string("invalid float for ") + flag + ": " + text);
    return value;
}

DriverConfig parseArgs(int argc, char** argv) {
    DriverConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto needValue = [&](const char* flag) -> const char* {
            require(i + 1 < argc, std::string("missing value for ") + flag);
            return argv[++i];
        };

        if (arg == "--so") {
            cfg.soPath = needValue("--so");
        } else if (arg == "--m") {
            cfg.m = parseI64(needValue("--m"), "--m");
        } else if (arg == "--k") {
            cfg.k = parseI64(needValue("--k"), "--k");
        } else if (arg == "--n") {
            cfg.n = parseI64(needValue("--n"), "--n");
        } else if (arg == "--threads") {
            cfg.threads = parseInt(needValue("--threads"), "--threads");
        } else if (arg == "--post-add") {
            cfg.postAdd = parseFloat(needValue("--post-add"), "--post-add");
        } else if (arg == "--relu") {
            cfg.relu = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            usage(argv[0]);
            require(false, std::string("unknown argument: ") + arg);
        }
    }

    require(cfg.m > 0 && cfg.k > 0 && cfg.n > 0, "m, k, n must be positive");
    require(cfg.threads > 0, "threads must be positive");
    return cfg;
}

std::string inferKernelSymbolFromSoPath(const std::string& soPath) {
    std::string base = soPath;
    const std::size_t slash = base.find_last_of('/');
    if (slash != std::string::npos) {
        base = base.substr(slash + 1);
    }
    if (base.rfind("./", 0) == 0) {
        base = base.substr(2);
    }
    require(base.size() > 3 && base.substr(base.size() - 3) == ".so",
            "shared library path must end with .so: " + soPath);
    base.resize(base.size() - 3);
    require(!base.empty(), "failed to infer kernel symbol from .so path: " + soPath);
    return "_mlir_ciface_" + base;
}

AnncMemRef2DF32 makeMemRef2D(float* data, std::int64_t rows, std::int64_t cols) {
    AnncMemRef2DF32 memref{};
    memref.allocated = data;
    memref.aligned = data;
    memref.offset = 0;
    memref.sizes[0] = rows;
    memref.sizes[1] = cols;
    memref.strides[0] = cols;
    memref.strides[1] = 1;
    return memref;
}

std::vector<float> makeLhs(std::int64_t m, std::int64_t k) {
    std::vector<float> data(static_cast<std::size_t>(m * k));
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < k; ++j) {
            data[static_cast<std::size_t>(i * k + j)] =
                static_cast<float>(((i * 7 + j * 3) % 17) - 8) * 0.25f;
        }
    }
    return data;
}

std::vector<float> makeRhs(std::int64_t k, std::int64_t n) {
    std::vector<float> data(static_cast<std::size_t>(k * n));
    for (std::int64_t i = 0; i < k; ++i) {
        for (std::int64_t j = 0; j < n; ++j) {
            data[static_cast<std::size_t>(i * n + j)] =
                static_cast<float>(((i * 5 + j * 11) % 23) - 11) * 0.2f;
        }
    }
    return data;
}

float applyPostOps(float value, const DriverConfig& cfg) {
    value += cfg.postAdd;
    if (cfg.relu && value < 0.0f) {
        value = 0.0f;
    }
    return value;
}

std::vector<float> referenceMatmul(const std::vector<float>& lhs,
                                   const std::vector<float>& rhs,
                                   const DriverConfig& cfg) {
    std::vector<float> output(static_cast<std::size_t>(cfg.m * cfg.n), 0.0f);
    for (std::int64_t i = 0; i < cfg.m; ++i) {
        for (std::int64_t j = 0; j < cfg.n; ++j) {
            float sum = 0.0f;
            for (std::int64_t p = 0; p < cfg.k; ++p) {
                sum += lhs[static_cast<std::size_t>(i * cfg.k + p)] *
                       rhs[static_cast<std::size_t>(p * cfg.n + j)];
            }
            output[static_cast<std::size_t>(i * cfg.n + j)] = applyPostOps(sum, cfg);
        }
    }
    return output;
}

PreparedInputs prepareInputs(const DriverConfig& cfg) {
    PreparedInputs prepared;
    prepared.lhs = makeLhs(cfg.m, cfg.k);
    prepared.rhs = makeRhs(cfg.k, cfg.n);
    prepared.expected = referenceMatmul(prepared.lhs, prepared.rhs, cfg);
    return prepared;
}

bool nearlyEqual(float lhs, float rhs, float eps = 1e-4f) {
    return std::fabs(lhs - rhs) <= eps;
}

void verifyOutput(const std::vector<float>& output, const std::vector<float>& expected) {
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (!nearlyEqual(output[i], expected[i])) {
            std::cerr << "FAIL: output mismatch at index " << i
                      << ", got " << output[i]
                      << ", expected " << expected[i] << "\n";
            std::exit(1);
        }
    }
}

void printTrace(const char* label,
                const DriverConfig& cfg,
                const std::string& soPath,
                const std::string& symbol,
                const FakePoolTrace& trace,
                float firstOutput,
                float lastOutput) {
    std::cout << "PASS: " << label << "\n";
    std::cout << "  so=" << soPath << "\n";
    std::cout << "  symbol=" << symbol << "\n";
    std::cout << "  m=" << cfg.m << " k=" << cfg.k << " n=" << cfg.n
              << " threads=" << cfg.threads
              << " post_add=" << cfg.postAdd
              << " relu=" << (cfg.relu ? "true" : "false") << "\n";
    std::cout << "  dispatch=" << trace.dispatchPath;
    if (trace.lastCostPerUnit.has_value()) {
        std::cout << " cost_per_unit=" << *trace.lastCostPerUnit;
    }
    if (trace.lastGrainSize.has_value()) {
        std::cout << " grain_size=" << *trace.lastGrainSize;
    }
    std::cout << "\n";
    std::cout << "  parallel_for_calls=" << trace.parallelForCalls
              << " chunks=" << trace.chunks.size() << "\n";
    std::cout << "  chunk_ranges=";
    for (const auto& chunk : trace.chunks) {
        std::cout << " [" << chunk.first << ", " << chunk.second << ")";
    }
    std::cout << "\n";
    std::cout << "  first_output=" << firstOutput
              << " last_output=" << lastOutput << "\n";
}

void runKernelWithAdapter(const char* label,
                          KernelFn kernel,
                          SetThreadPoolFn setThreadPool,
                          GetThreadPoolFn getThreadPool,
                          ClearThreadPoolFn clearThreadPool,
                          annc::threadpool::AnncThreadPool& threadPool,
                          const FakePoolTrace& trace,
                          const PreparedInputs& prepared,
                          const DriverConfig& cfg,
                          const std::string& soPath,
                          const std::string& symbol) {
    std::vector<float> output(static_cast<std::size_t>(cfg.m * cfg.n), 0.0f);
    auto lhsMemRef = makeMemRef2D(const_cast<float*>(prepared.lhs.data()), cfg.m, cfg.k);
    auto rhsMemRef = makeMemRef2D(const_cast<float*>(prepared.rhs.data()), cfg.k, cfg.n);
    auto outMemRef = makeMemRef2D(output.data(), cfg.m, cfg.n);

    setThreadPool(&threadPool);
    require(getThreadPool() == &threadPool, "threadpool was not installed into TLS bridge");

    kernel(&lhsMemRef, &rhsMemRef, &outMemRef);

    require(getThreadPool() == &threadPool, "threadpool changed unexpectedly during kernel call");
    clearThreadPool();
    require(getThreadPool() == nullptr, "threadpool was not cleared from TLS bridge");

    verifyOutput(output, prepared.expected);
    require(trace.parallelForCalls > 0, "kernel did not use frontend-provided threadpool callback");
    require(!trace.chunks.empty(), "parallel_for produced no work chunks");

    printTrace(label, cfg, soPath, symbol, trace, output.front(), output.back());
}

int main(int argc, char** argv) {
    const DriverConfig cfg = parseArgs(argc, argv);
    const std::string symbol = inferKernelSymbolFromSoPath(cfg.soPath);
    const PreparedInputs prepared = prepareInputs(cfg);

    void* handle = dlopen(cfg.soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    const char* dlopenError = dlerror();
    require(handle != nullptr, dlopenError != nullptr ? dlopenError : "dlopen failed");

    auto kernel = reinterpret_cast<KernelFn>(dlsym(handle, symbol.c_str()));
    auto setThreadPool = reinterpret_cast<SetThreadPoolFn>(dlsym(handle, "annc_set_current_threadpool"));
    auto getThreadPool = reinterpret_cast<GetThreadPoolFn>(dlsym(handle, "annc_get_current_threadpool"));
    auto clearThreadPool = reinterpret_cast<ClearThreadPoolFn>(dlsym(handle, "annc_clear_current_threadpool"));

    require(kernel != nullptr, std::string("failed to load kernel symbol: ") + symbol);
    require(setThreadPool != nullptr, "failed to load annc_set_current_threadpool");
    require(getThreadPool != nullptr, "failed to load annc_get_current_threadpool");
    require(clearThreadPool != nullptr, "failed to load annc_clear_current_threadpool");

    FakeTfThreadPool tfPool{};
    tfPool.numThreads = cfg.threads;
    FakeTfThreadPoolAdapter tfAdapter(&tfPool);
    runKernelWithAdapter("threadpool-aware kernel driver (tensorflow adapter)",
                         kernel,
                         setThreadPool,
                         getThreadPool,
                         clearThreadPool,
                         tfAdapter,
                         tfPool.trace,
                         prepared,
                         cfg,
                         cfg.soPath,
                         symbol);

    dlclose(handle);
    return 0;
}
