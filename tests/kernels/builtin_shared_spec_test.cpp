#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>

#include "Kernel/KernelRegistry.h"
#include "Kernel/KernelSymbolResolver.h"
#include "Kernel/MemRefTypes.h"

using namespace annc::kernels;

#define ANNC_SPEC_TOKEN_PREFIX matmul_kernel_specs
#define ANNC_BUILTIN_KERNEL_SPECS_FILE "annc/lib/Kernel/builtin_kernels/matmul_kernel_specs.inc"
#include "Kernel/BuiltinKernelSpecDeclare.h"

namespace {

template <typename Fn>
Fn loadKernelApi(const char* symbol) {
    std::string apiSymbol = std::string("_mlir_ciface_") + symbol;
    dlerror();
    void* address = dlsym(RTLD_DEFAULT, apiSymbol.c_str());
    const char* error = dlerror();
    if (error != nullptr || address == nullptr) {
        throw std::runtime_error("Failed to resolve kernel API symbol: " + apiSymbol);
    }
    return reinterpret_cast<Fn>(address);
}

AnncMemRef makeMemRef2D(void* data, int64_t dim0, int64_t dim1) {
    AnncMemRef memref{};
    memref.allocated = data;
    memref.aligned = data;
    memref.offset = 0;
    memref.sizes[0] = dim0;
    memref.sizes[1] = dim1;
    memref.strides[0] = dim1;
    memref.strides[1] = 1;
    memref.rank = 2;
    return memref;
}

} // namespace

int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "... "; \
    try { \
        test_##name(); \
        std::cout << "PASSED\n"; \
        tests_passed++; \
    } catch (const std::exception& e) { \
        std::cout << "FAILED: " << e.what() << "\n"; \
        tests_failed++; \
    } catch (...) { \
        std::cout << "FAILED: unknown error\n"; \
        tests_failed++; \
    } \
} while (0)

#define EXPECT_TRUE(cond) if (!(cond)) throw std::runtime_error("EXPECT_TRUE failed: " #cond)
#define EXPECT_FALSE(cond) if (cond) throw std::runtime_error("EXPECT_FALSE failed: " #cond)
#define EXPECT_EQ(a, b) if (!((a) == (b))) throw std::runtime_error("EXPECT_EQ failed: " #a " != " #b)
#define EXPECT_STREQ(a, b) if ((a) != (b)) throw std::runtime_error("EXPECT_STREQ failed: " #a " != " #b)
#define EXPECT_HAS_PREFIX(value, prefix) if ((value).rfind(prefix, 0) != 0) throw std::runtime_error("EXPECT_HAS_PREFIX failed")
#define EXPECT_HAS_SUFFIX(value, suffix) do { const auto& _value = (value); const std::string _suffix = (suffix); if (_value.size() < _suffix.size() || _value.compare(_value.size() - _suffix.size(), _suffix.size(), _suffix) != 0) throw std::runtime_error("EXPECT_HAS_SUFFIX failed"); } while (0)

TEST(BuiltinRegistryExposesMatMulDefault) {
    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("MatMul", "aarch64");
    EXPECT_TRUE(symbolOpt.has_value());
    EXPECT_HAS_PREFIX(*symbolOpt, "ANNCKernel_matmul_");
    EXPECT_HAS_SUFFIX(*symbolOpt, "_auto");
}

TEST(BuiltinResolverFindsMatMulDefault) {
    KernelSymbolResolverRequest request;
    request.op_type = "MatMul";
    request.backend = "aarch64";

    auto symbol = ResolveKernelSymbol(request);
    EXPECT_TRUE(symbol.has_value());
    EXPECT_HAS_PREFIX(*symbol, "ANNCKernel_matmul_");
    EXPECT_HAS_SUFFIX(*symbol, "_auto");
}

TEST(BuiltinMatMulWrapperRuns) {
    float lhsData[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float rhsData[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float outputData[9] = {0.0f};

    AnncMemRef lhs = makeMemRef2D(lhsData, 2, 3);
    AnncMemRef rhs = makeMemRef2D(rhsData, 3, 2);
    AnncMemRef output = makeMemRef2D(outputData, 2, 2);

    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("MatMul", "aarch64");
    EXPECT_TRUE(symbolOpt.has_value());

    auto kernel = loadKernelApi<void (*)(AnncMemRef*, AnncMemRef*, AnncMemRef*)>(
        symbolOpt->c_str());
    kernel(&lhs, &rhs, &output);

    EXPECT_EQ(outputData[0], 58.0f);
    EXPECT_EQ(outputData[1], 64.0f);
    EXPECT_EQ(outputData[2], 139.0f);
    EXPECT_EQ(outputData[3], 154.0f);
}

int main() {
    std::cout << "=== Builtin Shared Spec Tests ===\n\n";

    RUN_TEST(BuiltinRegistryExposesMatMulDefault);
    RUN_TEST(BuiltinResolverFindsMatMulDefault);
    RUN_TEST(BuiltinMatMulWrapperRuns);

    std::cout << "\n=== Results ===\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";

    return tests_failed == 0 ? 0 : 1;
}
