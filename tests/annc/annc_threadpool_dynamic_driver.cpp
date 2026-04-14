#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Minimal ABI mirror used by the simulated frontend adapter.
struct AnncThreadPoolCtx {
    void* impl = nullptr;
    int (*num_threads)(void* impl) = nullptr;
    bool (*in_parallel_region)(void* impl) = nullptr;
    void (*parallel_for)(
        void* impl,
        std::int64_t begin,
        std::int64_t end,
        std::int64_t grain_size,
        void (*fn)(std::int64_t, std::int64_t, int, void*),
        void* user_data) = nullptr;
};

struct AnncMemRef2DF32 {
    float* allocated;
    float* aligned;
    std::int64_t offset;
    std::int64_t sizes[2];
    std::int64_t strides[2];
};

using KernelFn = void (*)(AnncMemRef2DF32*, AnncMemRef2DF32*, AnncMemRef2DF32*);
using SetThreadPoolCtxFn = void (*)(AnncThreadPoolCtx*);
using GetThreadPoolCtxFn = AnncThreadPoolCtx* (*)();
using ClearThreadPoolCtxFn = void (*)();

// Runtime knobs so one driver can cover multiple .so/shape cases.
struct DriverConfig {
    std::string soPath = "./demo.so";
    std::int64_t m = 128;
    std::int64_t k = 64;
    std::int64_t n = 128;
    int threads = 4;
    float postAdd = 0.0f;
    bool relu = false;
};

// Fake frontend-owned threadpool plus a few counters for validation.
struct FakeTfThreadPool {
    int numThreads = 4;
    bool inParallel = false;
    int parallelForCalls = 0;
    std::vector<std::pair<std::int64_t, std::int64_t>> chunks;
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

// Test helper: fail fast on any mismatch.
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

// Parse command-line options once so main() stays linear.
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

// Frontend threadpool callbacks exposed through AnncThreadPoolCtx.
int fakeNumThreads(void* impl) {
    return static_cast<FakeTfThreadPool*>(impl)->numThreads;
}

bool fakeInParallelRegion(void* impl) {
    return static_cast<FakeTfThreadPool*>(impl)->inParallel;
}

void fakeParallelFor(void* impl,
                     std::int64_t begin,
                     std::int64_t end,
                     std::int64_t grainSize,
                     void (*fn)(std::int64_t, std::int64_t, int, void*),
                     void* userData) {
    auto* pool = static_cast<FakeTfThreadPool*>(impl);
    pool->parallelForCalls++;

    const std::int64_t total = end - begin;
    const int workers = std::max(pool->numThreads, 1);
    const std::int64_t baseChunk = (total + workers - 1) / workers;
    const std::int64_t chunk = std::max(baseChunk, std::max<std::int64_t>(grainSize, 1));

    int workerId = 0;
    for (std::int64_t chunkBegin = begin; chunkBegin < end; chunkBegin += chunk) {
        const std::int64_t chunkEnd = std::min(chunkBegin + chunk, end);
        pool->chunks.emplace_back(chunkBegin, chunkEnd);
        fn(chunkBegin, chunkEnd, workerId++, userData);
    }
}

// Build a row-major memref descriptor over an existing buffer.
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

// Deterministic inputs keep the reference stable across runs.
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

// Optional post-ops keep the driver reusable for simple fused examples.
float applyPostOps(float value, const DriverConfig& cfg) {
    value += cfg.postAdd;
    if (cfg.relu && value < 0.0f) {
        value = 0.0f;
    }
    return value;
}

// Scalar reference used to validate the loaded kernel.
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

bool nearlyEqual(float lhs, float rhs, float eps = 1e-4f) {
    return std::fabs(lhs - rhs) <= eps;
}

int main(int argc, char** argv) {
    const DriverConfig cfg = parseArgs(argc, argv);
    const std::string symbol = inferKernelSymbolFromSoPath(cfg.soPath);

    // 1. Load the generated kernel .so.
    void* handle = dlopen(cfg.soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    const char* dlopenError = dlerror();
    require(handle != nullptr, dlopenError != nullptr ? dlopenError : "dlopen failed");

    // 2. Resolve the kernel entry and threadpool bridge APIs.
    auto kernel = reinterpret_cast<KernelFn>(dlsym(handle, symbol.c_str()));
    auto setThreadPoolCtx = reinterpret_cast<SetThreadPoolCtxFn>(dlsym(handle, "annc_set_current_threadpool_ctx"));
    auto getThreadPoolCtx = reinterpret_cast<GetThreadPoolCtxFn>(dlsym(handle, "annc_get_current_threadpool_ctx"));
    auto clearThreadPoolCtx = reinterpret_cast<ClearThreadPoolCtxFn>(dlsym(handle, "annc_clear_current_threadpool_ctx"));

    require(kernel != nullptr, std::string("failed to load kernel symbol: ") + symbol);
    require(setThreadPoolCtx != nullptr, "failed to load annc_set_current_threadpool_ctx");
    require(getThreadPoolCtx != nullptr, "failed to load annc_get_current_threadpool_ctx");
    require(clearThreadPoolCtx != nullptr, "failed to load annc_clear_current_threadpool_ctx");

    // 3. Build a frontend-owned threadpool and wrap it in AnncThreadPoolCtx.
    FakeTfThreadPool fakeTfPool{};
    fakeTfPool.numThreads = cfg.threads;

    AnncThreadPoolCtx threadPoolCtx{};
    threadPoolCtx.impl = &fakeTfPool;
    threadPoolCtx.num_threads = &fakeNumThreads;
    threadPoolCtx.in_parallel_region = &fakeInParallelRegion;
    threadPoolCtx.parallel_for = &fakeParallelFor;

    // 4. Prepare inputs, output buffer, and scalar reference.
    std::vector<float> lhs = makeLhs(cfg.m, cfg.k);
    std::vector<float> rhs = makeRhs(cfg.k, cfg.n);
    std::vector<float> expected = referenceMatmul(lhs, rhs, cfg);
    std::vector<float> output(static_cast<std::size_t>(cfg.m * cfg.n), 0.0f);

    auto lhsMemRef = makeMemRef2D(lhs.data(), cfg.m, cfg.k);
    auto rhsMemRef = makeMemRef2D(rhs.data(), cfg.k, cfg.n);
    auto outMemRef = makeMemRef2D(output.data(), cfg.m, cfg.n);

    // 5. Install the frontend threadpool into the exported TLS bridge.
    setThreadPoolCtx(&threadPoolCtx);
    require(getThreadPoolCtx() == &threadPoolCtx, "threadpool ctx was not installed into TLS bridge");

    // 6. Call the generated MLIR C interface symbol.
    kernel(&lhsMemRef, &rhsMemRef, &outMemRef);

    // 7. The adapter owns cleanup of the TLS bridge.
    require(getThreadPoolCtx() == &threadPoolCtx, "threadpool ctx changed unexpectedly during kernel call");
    clearThreadPoolCtx();
    require(getThreadPoolCtx() == nullptr, "threadpool ctx was not cleared from TLS bridge");

    // 8. Check numerical correctness.
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (!nearlyEqual(output[i], expected[i])) {
            std::cerr << "FAIL: output mismatch at index " << i
                      << ", got " << output[i]
                      << ", expected " << expected[i] << "\n";
            std::exit(1);
        }
    }

    // 9. Check that the kernel actually used the injected threadpool.
    require(fakeTfPool.parallelForCalls > 0, "kernel did not use frontend-provided threadpool callback");
    require(!fakeTfPool.chunks.empty(), "parallel_for produced no work chunks");

    std::cout << "PASS: threadpool-aware kernel driver\n";
    std::cout << "  so=" << cfg.soPath << "\n";
    std::cout << "  symbol=" << symbol << "\n";
    std::cout << "  m=" << cfg.m << " k=" << cfg.k << " n=" << cfg.n
              << " threads=" << cfg.threads
              << " post_add=" << cfg.postAdd
              << " relu=" << (cfg.relu ? "true" : "false") << "\n";
    std::cout << "  parallel_for_calls=" << fakeTfPool.parallelForCalls
              << " chunks=" << fakeTfPool.chunks.size() << '\n';
    std::cout << "  chunk_ranges=";
    for (const auto& chunk : fakeTfPool.chunks) {
        std::cout << " [" << chunk.first << ", " << chunk.second << ")";
    }
    std::cout << "\n";
    std::cout << "  first_output=" << output.front()
              << " last_output=" << output.back() << "\n";

    dlclose(handle);
    return 0;
}
