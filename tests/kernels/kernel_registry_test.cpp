#include <cassert>
#include <cstdint>
#include <atomic>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "Kernel/KernelMacros.h"
#include "Kernel/KernelRegistry.h"
#include "Kernel/KernelSymbolResolver.h"
#include "Kernel/MemRefTypes.h"
#include "tests/kernels/auto_kernel_support.h"

using namespace annc::kernels;

namespace {

KernelQuery makeQuery(std::string opType,
                      std::vector<TypeConstraintInfo> constraints,
                      std::string backend = "aarch64") {
    KernelQuery query;
    query.op_type = std::move(opType);
    query.backend = std::move(backend);
    query.type_constraints = std::move(constraints);
    return query;
}

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
#define EXPECT_NE(a, b) if ((a) == (b)) throw std::runtime_error("EXPECT_NE failed: " #a " == " #b)
#define EXPECT_STREQ(a, b) if ((a) != (b)) throw std::runtime_error("EXPECT_STREQ failed: " #a " != " #b)
#define EXPECT_HAS_PREFIX(value, prefix) if ((value).rfind(prefix, 0) != 0) throw std::runtime_error("EXPECT_HAS_PREFIX failed")
#define EXPECT_HAS_SUFFIX(value, suffix) do { const auto& _value = (value); const std::string _suffix = (suffix); if (_value.size() < _suffix.size() || _value.compare(_value.size() - _suffix.size(), _suffix.size(), _suffix) != 0) throw std::runtime_error("EXPECT_HAS_SUFFIX failed"); } while (0)

void setup() {
    KernelRegistry::instance().clear();
}

void teardown() {
    KernelRegistry::instance().clear();
}

TEST(AutoWrapperDefaultRegistrationAndForwarding) {
    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("AutoDefaultOp", "aarch64");
    EXPECT_TRUE(symbolOpt.has_value());
    EXPECT_HAS_PREFIX(*symbolOpt, "ANNCKernel_auto_kernel_specs_");
    EXPECT_HAS_SUFFIX(*symbolOpt, "_auto");

    float inputData[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float outputData[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    AnncMemRef1DF32 input{inputData, inputData, 0, {4}, {1}};
    AnncMemRef1DF32 output{outputData, outputData, 0, {4}, {1}};

    auto defaultKernel = loadKernelApi<void (*)(AnncMemRef1DF32*, AnncMemRef1DF32*)>(
        symbolOpt->c_str());
    defaultKernel(&output, &input);

    EXPECT_EQ(outputData[0], 3.0f);
    EXPECT_EQ(outputData[1], 6.0f);
    EXPECT_EQ(outputData[2], 9.0f);
    EXPECT_EQ(outputData[3], 12.0f);
}

TEST(AutoWrapperSpecializationRegistrationAndForwarding) {
    EXPECT_FALSE(KernelRegistry::instance().lookupKernelSymbol("AutoSpecOp", "aarch64").has_value());

    auto f32SymbolOpt = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("AutoSpecOp", {{"Tdata", "float"}}));
    EXPECT_TRUE(f32SymbolOpt.has_value());
    const std::string& f32Symbol = *f32SymbolOpt;
    EXPECT_HAS_PREFIX(f32Symbol, "ANNCKernel_auto_kernel_specs_");
    EXPECT_HAS_SUFFIX(f32Symbol, "_auto");

    auto i32SymbolOpt = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("AutoSpecOp", {{"Tdata", "int32_t"}}));
    EXPECT_TRUE(i32SymbolOpt.has_value());
    const std::string& i32Symbol = *i32SymbolOpt;
    EXPECT_HAS_PREFIX(i32Symbol, "ANNCKernel_auto_kernel_specs_");
    EXPECT_HAS_SUFFIX(i32Symbol, "_auto");
    EXPECT_FALSE(f32Symbol == i32Symbol);

    g_auto_spec_f32_calls = 0;
    g_auto_spec_i32_calls = 0;

    auto f32Kernel = loadKernelApi<void (*)()>(f32Symbol.c_str());
    auto i32Kernel = loadKernelApi<void (*)()>(i32Symbol.c_str());
    f32Kernel();
    i32Kernel();

    EXPECT_EQ(g_auto_spec_f32_calls, 1);
    EXPECT_EQ(g_auto_spec_i32_calls, 1);
}

TEST(BasicDefaultRegistration) {
    setup();

    auto info = Name("TestOp").Backend("aarch64").Build("test_kernel", "test.cpp", 1);
    KernelRegistry::instance().registerKernel(std::move(info));

    EXPECT_TRUE(KernelRegistry::instance().hasKernel("TestOp", "aarch64"));
    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("TestOp", "aarch64");
    EXPECT_TRUE(symbolOpt.has_value());
    EXPECT_STREQ(*symbolOpt, "test_kernel");

    teardown();
}

TEST(NotFound) {
    setup();

    EXPECT_FALSE(KernelRegistry::instance().hasKernel("NonExistentOp", "aarch64"));
    EXPECT_FALSE(KernelRegistry::instance().lookupKernelSymbol("NonExistentOp", "aarch64").has_value());

    teardown();
}

TEST(ListKernels) {
    setup();

    KernelRegistry::instance().registerKernel(Name("Op1").Backend("aarch64").Build("kernel1", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(Name("Op2").Backend("aarch64").Build("kernel2", "test.cpp", 2));

    auto kernels = KernelRegistry::instance().listKernels();
    EXPECT_EQ(kernels.size(), static_cast<size_t>(2));

    teardown();
}

TEST(DefaultOverwrite) {
    setup();

    KernelRegistry::instance().registerKernel(Name("OverwriteOp").Backend("aarch64").Build("test_kernel_v1", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(Name("OverwriteOp").Backend("aarch64").Build("test_kernel_v2", "test.cpp", 2));

    EXPECT_TRUE(KernelRegistry::instance().hasKernel("OverwriteOp", "aarch64"));
    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("OverwriteOp", "aarch64");
    EXPECT_TRUE(symbolOpt.has_value());
    EXPECT_STREQ(*symbolOpt, "test_kernel_v2");

    teardown();
}

TEST(SpecializationLookup) {
    setup();

    KernelRegistry::instance().registerKernel(Name("MixedOp").Backend("aarch64").Build("mixed_default", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("MixedOp").Backend("aarch64").TypeConstraint<float>("T0").Build("mixed_f32", "test.cpp", 2));
    KernelRegistry::instance().registerKernel(
        Name("MixedOp").Backend("aarch64").TypeConstraint<std::int32_t>("T0").Build("mixed_i32", "test.cpp", 3));

    auto defaultSymbolOpt = KernelRegistry::instance().lookupKernelSymbol("MixedOp", "aarch64");
    EXPECT_TRUE(defaultSymbolOpt.has_value());
    EXPECT_STREQ(*defaultSymbolOpt, "mixed_default");

    auto f32SymbolOpt = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("MixedOp", {{"T0", "float"}}));
    EXPECT_TRUE(f32SymbolOpt.has_value());
    EXPECT_STREQ(*f32SymbolOpt, "mixed_f32");

    auto i32SymbolOpt = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("MixedOp", {{"T0", "int32_t"}}));
    EXPECT_TRUE(i32SymbolOpt.has_value());
    EXPECT_STREQ(*i32SymbolOpt, "mixed_i32");

    teardown();
}

TEST(SpecializationRequiresExactMatch) {
    setup();

    KernelRegistry::instance().registerKernel(Name("FallbackOp").Backend("aarch64").Build("fallback_default", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("FallbackOp").Backend("aarch64").TypeConstraint<float>("T0").Build("fallback_f32", "test.cpp", 2));

    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("FallbackOp", {{"T0", "int32_t"}}));
    EXPECT_FALSE(symbolOpt.has_value());

    teardown();
}

TEST(OnlySpecializationsWithoutDefaultIsNotImplicitlyResolvable) {
    setup();

    KernelRegistry::instance().registerKernel(
        Name("SpecOnlyOp").Backend("aarch64").TypeConstraint<float>("T0").Build("spec_f32", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("SpecOnlyOp").Backend("aarch64").TypeConstraint<std::int32_t>("T0").Build("spec_i32", "test.cpp", 2));

    EXPECT_FALSE(KernelRegistry::instance().lookupKernelSymbol("SpecOnlyOp", "aarch64").has_value());
    auto specSymbolOpt = KernelRegistry::instance().lookupKernelSymbol(makeQuery("SpecOnlyOp", {{"T0", "float"}}));
    EXPECT_TRUE(specSymbolOpt.has_value());
    EXPECT_STREQ(*specSymbolOpt, "spec_f32");

    teardown();
}

TEST(BuilderMacroSupportsDefaultRegistration) {
    setup();

    auto info = Name("MacroOp").Backend("aarch64").Build("macro_kernel", "macro.cpp", 7);
    KernelRegistry::instance().registerKernel(std::move(info));

    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("MacroOp", "aarch64");
    EXPECT_TRUE(symbolOpt.has_value());
    EXPECT_STREQ(*symbolOpt, "macro_kernel");

    teardown();
}

TEST(SymbolResolverFindsDefaultKernel) {
    setup();

    KernelRegistry::instance().registerKernel(
        Name("ResolveDefaultOp").Backend("aarch64").Build("resolve_default", "test.cpp", 1));

    KernelSymbolResolverRequest request;
    request.op_type = "ResolveDefaultOp";
    request.backend = "aarch64";

    auto symbol = ResolveKernelSymbol(request);
    EXPECT_TRUE(symbol.has_value());
    EXPECT_STREQ(*symbol, "resolve_default");

    teardown();
}

TEST(SymbolResolverFindsSpecializedKernel) {
    setup();

    KernelRegistry::instance().registerKernel(
        Name("ResolveSpecOp").Backend("aarch64").Build("resolve_spec_default", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("ResolveSpecOp").Backend("aarch64").TypeConstraint<float>("Tdata").Build(
            "resolve_spec_f32", "test.cpp", 2));

    KernelSymbolResolverRequest request;
    request.op_type = "ResolveSpecOp";
    request.backend = "aarch64";
    request.type_constraints = {{"Tdata", "float"}};

    auto symbol = ResolveKernelSymbol(request);
    EXPECT_TRUE(symbol.has_value());
    EXPECT_STREQ(*symbol, "resolve_spec_f32");

    teardown();
}

TEST(SymbolResolverSpecializationMissReturnsNullopt) {
    setup();

    KernelRegistry::instance().registerKernel(
        Name("ResolveSpecOp").Backend("aarch64").Build("resolve_spec_default", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("ResolveSpecOp").Backend("aarch64").TypeConstraint<float>("Tdata").Build(
            "resolve_spec_f32", "test.cpp", 2));

    KernelSymbolResolverRequest request;
    request.op_type = "ResolveSpecOp";
    request.backend = "aarch64";
    request.type_constraints = {{"Tdata", "int32_t"}};

    auto symbol = ResolveKernelSymbol(request);
    EXPECT_FALSE(symbol.has_value());

    teardown();
}

TEST(SymbolResolverReturnsNulloptWhenMissing) {
    setup();

    KernelSymbolResolverRequest request;
    request.op_type = "MissingOp";
    request.backend = "aarch64";

    auto symbol = ResolveKernelSymbol(request);
    EXPECT_FALSE(symbol.has_value());

    teardown();
}

TEST(ConcurrentAccess) {
    setup();

    KernelRegistry::instance().registerKernel(Name("ConcurrentOp").Backend("aarch64").Build("concurrent_kernel", "test.cpp", 1));

    const int numThreads = 4;
    const int opsPerThread = 16;
    std::vector<std::thread> threads;
    std::vector<bool> results(numThreads, false);

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([i, opsPerThread, &results]() {
            try {
                for (int j = 0; j < opsPerThread; ++j) {
                    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("ConcurrentOp", "aarch64");
                    if (!symbolOpt.has_value() || *symbolOpt != "concurrent_kernel") {
                        return;
                    }
                }
                results[i] = true;
            } catch (...) {
                results[i] = false;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (bool result : results) {
        EXPECT_TRUE(result);
    }

    teardown();
}

TEST(SingleSpecializationWithoutDefaultReturnsNullopt) {
    setup();

    KernelRegistry::instance().registerKernel(
        Name("SingleSpecOp").Backend("aarch64").TypeConstraint<float>("T").Build("single_spec", "test.cpp", 1));

    EXPECT_FALSE(KernelRegistry::instance().lookupKernelSymbol("SingleSpecOp", "aarch64").has_value());

    auto specOpt = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("SingleSpecOp", {{"T", "float"}}));
    EXPECT_TRUE(specOpt.has_value());
    EXPECT_STREQ(*specOpt, "single_spec");

    teardown();
}

TEST(EmptyStringInputs) {
    setup();

    KernelRegistry::instance().registerKernel(Name("").Backend("aarch64").Build("empty_op_kernel", "test.cpp", 1));
    auto emptyOpt = KernelRegistry::instance().lookupKernelSymbol("", "aarch64");
    EXPECT_TRUE(emptyOpt.has_value());
    EXPECT_STREQ(*emptyOpt, "empty_op_kernel");

    EXPECT_FALSE(KernelRegistry::instance().lookupKernelSymbol("NonExistent", "").has_value());

    teardown();
}

TEST(MultipleClearAndReRegister) {
    setup();

    for (int i = 0; i < 3; ++i) {
        KernelRegistry::instance().registerKernel(
            Name("RepeatOp").Backend("aarch64").Build("repeat_v" + std::to_string(i), "test.cpp", i));

        auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("RepeatOp", "aarch64");
        EXPECT_TRUE(symbolOpt.has_value());
        EXPECT_STREQ(*symbolOpt, "repeat_v" + std::to_string(i));

        KernelRegistry::instance().clear();
        EXPECT_FALSE(KernelRegistry::instance().lookupKernelSymbol("RepeatOp", "aarch64").has_value());
    }

    teardown();
}

TEST(BackendIsolation) {
    setup();

    KernelRegistry::instance().registerKernel(
        Name("MultiBackendOp").Backend("aarch64").Build("aarch64_impl", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("MultiBackendOp").Backend("cuda").Build("cuda_impl", "test.cpp", 2));

    auto aarch64Opt = KernelRegistry::instance().lookupKernelSymbol("MultiBackendOp", "aarch64");
    EXPECT_TRUE(aarch64Opt.has_value());
    EXPECT_STREQ(*aarch64Opt, "aarch64_impl");

    auto cudaOpt = KernelRegistry::instance().lookupKernelSymbol("MultiBackendOp", "cuda");
    EXPECT_TRUE(cudaOpt.has_value());
    EXPECT_STREQ(*cudaOpt, "cuda_impl");

    EXPECT_FALSE(KernelRegistry::instance().lookupKernelSymbol("MultiBackendOp", "x86").has_value());

    teardown();
}

TEST(MultipleTypeConstraintsExactMatch) {
    setup();

    KernelRegistry::instance().registerKernel(
        Name("MultiConstraintOp").Backend("aarch64")
            .TypeConstraint<float>("T1")
            .TypeConstraint<int32_t>("T2")
            .Build("f32_i32_impl", "test.cpp", 1));

    KernelRegistry::instance().registerKernel(
        Name("MultiConstraintOp").Backend("aarch64")
            .TypeConstraint<double>("T1")
            .TypeConstraint<int64_t>("T2")
            .Build("d_i64_impl", "test.cpp", 2));

    auto match1 = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("MultiConstraintOp", {{"T1", "float"}, {"T2", "int32_t"}}));
    EXPECT_TRUE(match1.has_value());
    EXPECT_STREQ(*match1, "f32_i32_impl");

    auto match2 = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("MultiConstraintOp", {{"T1", "double"}, {"T2", "int64_t"}}));
    EXPECT_TRUE(match2.has_value());
    EXPECT_STREQ(*match2, "d_i64_impl");

    auto noMatch = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("MultiConstraintOp", {{"T1", "float"}, {"T2", "int64_t"}}));
    EXPECT_FALSE(noMatch.has_value());

    teardown();
}

TEST(ConcurrentMixedRegisterAndLookup) {
    setup();

    const int numThreads = 8;
    const int opsPerThread = 100;
    std::vector<std::thread> threads;
    std::atomic<int> successCount{0};

    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([t, opsPerThread, &successCount]() {
            try {
                for (int i = 0; i < opsPerThread; ++i) {
                    if (i % 2 == 0) {
                        std::string opName = "ConcurrentMixedOp_" + std::to_string(t) + "_" + std::to_string(i);
                        KernelRegistry::instance().registerKernel(
                            Name(opName.c_str()).Backend("aarch64").Build("kernel_" + std::to_string(i), "test.cpp", i));
                    } else {
                        auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("ConcurrentMixedOp_0_0", "aarch64");
                        if (!symbolOpt.has_value()) {
                            return;
                        }
                    }
                }
                successCount.fetch_add(1);
            } catch (...) {
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(successCount.load(), numThreads);

    teardown();
}

TEST(LastRegistrationWinsForSameSignature) {
    setup();

    KernelRegistry::instance().registerKernel(
        Name("WinningOp").Backend("aarch64").Build("first_version", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("WinningOp").Backend("aarch64").Build("second_version", "test.cpp", 2));
    KernelRegistry::instance().registerKernel(
        Name("WinningOp").Backend("aarch64").Build("third_version", "test.cpp", 3));

    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("WinningOp", "aarch64");
    EXPECT_TRUE(symbolOpt.has_value());
    EXPECT_STREQ(*symbolOpt, "third_version");

    teardown();
}

TEST(ListKernelsAfterMultipleRegistrations) {
    setup();

    KernelRegistry::instance().registerKernel(Name("OpA").Backend("aarch64").Build("kernel_a", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(Name("OpB").Backend("cuda").Build("kernel_b", "test.cpp", 2));
    KernelRegistry::instance().registerKernel(
        Name("OpA").Backend("cuda").Build("kernel_a_cuda", "test.cpp", 3));

    auto kernels = KernelRegistry::instance().listKernels();
    EXPECT_EQ(kernels.size(), static_cast<size_t>(2));

    bool foundOpA = false;
    bool foundOpB = false;
    for (const auto& kernel : kernels) {
        if (kernel.find("OpA") != std::string::npos) foundOpA = true;
        if (kernel.find("OpB") != std::string::npos) foundOpB = true;
    }
    EXPECT_TRUE(foundOpA);
    EXPECT_TRUE(foundOpB);

    teardown();
}

TEST(BuilderChainingVariousCombinations) {
    setup();

    auto info1 = Name("ChainOp").Build("default_backend", "test.cpp", 1);
    KernelRegistry::instance().registerKernel(std::move(info1));

    auto info2 = Name("ChainOp").Backend("custom").Build("custom_backend", "test.cpp", 2);
    KernelRegistry::instance().registerKernel(std::move(info2));

    auto info3 = Name("ChainOp")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .Build("specialized_f32", "test.cpp", 3);
    KernelRegistry::instance().registerKernel(std::move(info3));

    auto defaultOpt = KernelRegistry::instance().lookupKernelSymbol("ChainOp");
    EXPECT_TRUE(defaultOpt.has_value());
    EXPECT_STREQ(*defaultOpt, "default_backend");

    auto customOpt = KernelRegistry::instance().lookupKernelSymbol("ChainOp", "custom");
    EXPECT_TRUE(customOpt.has_value());
    EXPECT_STREQ(*customOpt, "custom_backend");

    auto specOpt = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("ChainOp", {{"T", "float"}}));
    EXPECT_TRUE(specOpt.has_value());
    EXPECT_STREQ(*specOpt, "specialized_f32");

    teardown();
}

int main() {
    std::cout << "=== ANNC Kernel Registry Tests ===\n\n";

    RUN_TEST(AutoWrapperDefaultRegistrationAndForwarding);
    RUN_TEST(AutoWrapperSpecializationRegistrationAndForwarding);
    RUN_TEST(BasicDefaultRegistration);
    RUN_TEST(NotFound);
    RUN_TEST(ListKernels);
    RUN_TEST(DefaultOverwrite);
    RUN_TEST(SpecializationLookup);
    RUN_TEST(SpecializationRequiresExactMatch);
    RUN_TEST(OnlySpecializationsWithoutDefaultIsNotImplicitlyResolvable);
    RUN_TEST(BuilderMacroSupportsDefaultRegistration);
    RUN_TEST(SymbolResolverFindsDefaultKernel);
    RUN_TEST(SymbolResolverFindsSpecializedKernel);
    RUN_TEST(SymbolResolverSpecializationMissReturnsNullopt);
    RUN_TEST(SymbolResolverReturnsNulloptWhenMissing);
    RUN_TEST(ConcurrentAccess);
    RUN_TEST(SingleSpecializationWithoutDefaultReturnsNullopt);
    RUN_TEST(EmptyStringInputs);
    RUN_TEST(MultipleClearAndReRegister);
    RUN_TEST(BackendIsolation);
    RUN_TEST(MultipleTypeConstraintsExactMatch);
    RUN_TEST(ConcurrentMixedRegisterAndLookup);
    RUN_TEST(LastRegistrationWinsForSameSignature);
    RUN_TEST(ListKernelsAfterMultipleRegistrations);
    RUN_TEST(BuilderChainingVariousCombinations);

    std::cout << "\n=== Results ===\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";

    return tests_failed == 0 ? 0 : 1;
}
