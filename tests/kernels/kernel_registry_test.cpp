#include <atomic>
#include <cstdint>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "Kernel/KernelMacros.h"
#include "Kernel/KernelRegistry.h"
#include "Kernel/MemRefTypes.h"
#include "tests/kernels/auto_kernel_support.h"

using namespace annc::kernels;

namespace annc::kernels::test {
void registerAutoKernelSpecsForTest();
} // namespace annc::kernels::test

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

class KernelRegistryTest : public ::testing::Test {
protected:
    void SetUp() override { KernelRegistry::instance().clear(); }
    void TearDown() override { KernelRegistry::instance().clear(); }
};

class AutoKernelRegistryTest : public KernelRegistryTest {
protected:
    void SetUp() override {
        KernelRegistryTest::SetUp();
        annc::kernels::test::registerAutoKernelSpecsForTest();
    }
};

} // namespace

TEST_F(AutoKernelRegistryTest, AutoWrapperDefaultRegistrationAndForwarding) {
    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("AutoDefaultOp", "aarch64");
    ASSERT_TRUE(symbolOpt.has_value());
    EXPECT_TRUE(symbolOpt->rfind("ANNCKernel_auto_kernel_specs_", 0) == 0);
    EXPECT_TRUE(symbolOpt->size() >= 5 && symbolOpt->compare(symbolOpt->size() - 5, 5, "_auto") == 0);

    float inputData[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float outputData[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    AnncMemRef1DF32 input{inputData, inputData, 0, {4}, {1}};
    AnncMemRef1DF32 output{outputData, outputData, 0, {4}, {1}};

    auto defaultKernel = loadKernelApi<void (*)(AnncMemRef1DF32*, AnncMemRef1DF32*)>(
        symbolOpt->c_str());
    defaultKernel(&output, &input);

    EXPECT_FLOAT_EQ(outputData[0], 3.0f);
    EXPECT_FLOAT_EQ(outputData[1], 6.0f);
    EXPECT_FLOAT_EQ(outputData[2], 9.0f);
    EXPECT_FLOAT_EQ(outputData[3], 12.0f);
}

TEST_F(AutoKernelRegistryTest, AutoWrapperSpecializationRegistrationAndForwarding) {
    EXPECT_FALSE(KernelRegistry::instance().lookupKernelSymbol("AutoSpecOp", "aarch64").has_value());

    auto f32SymbolOpt = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("AutoSpecOp", {{"Tdata", "float"}}));
    ASSERT_TRUE(f32SymbolOpt.has_value());
    EXPECT_TRUE(f32SymbolOpt->rfind("ANNCKernel_auto_kernel_specs_", 0) == 0);

    auto i32SymbolOpt = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("AutoSpecOp", {{"Tdata", "int32_t"}}));
    ASSERT_TRUE(i32SymbolOpt.has_value());
    EXPECT_TRUE(i32SymbolOpt->rfind("ANNCKernel_auto_kernel_specs_", 0) == 0);
    EXPECT_NE(*f32SymbolOpt, *i32SymbolOpt);

    g_auto_spec_f32_calls = 0;
    g_auto_spec_i32_calls = 0;

    auto f32Kernel = loadKernelApi<void (*)()>(f32SymbolOpt->c_str());
    auto i32Kernel = loadKernelApi<void (*)()>(i32SymbolOpt->c_str());
    f32Kernel();
    i32Kernel();

    EXPECT_EQ(g_auto_spec_f32_calls, 1);
    EXPECT_EQ(g_auto_spec_i32_calls, 1);
}

TEST_F(KernelRegistryTest, BasicDefaultRegistration) {
    KernelRegistry::instance().registerKernel(
        Name("TestOp").Backend("aarch64").Build("test_kernel", "test.cpp", 1));

    EXPECT_TRUE(KernelRegistry::instance().hasKernel("TestOp", "aarch64"));
    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("TestOp", "aarch64");
    ASSERT_TRUE(symbolOpt.has_value());
    EXPECT_EQ(*symbolOpt, "test_kernel");
}

TEST_F(KernelRegistryTest, NotFound) {
    EXPECT_FALSE(KernelRegistry::instance().hasKernel("NonExistentOp", "aarch64"));
    EXPECT_FALSE(KernelRegistry::instance().lookupKernelSymbol("NonExistentOp", "aarch64").has_value());
}

TEST_F(KernelRegistryTest, ListKernels) {
    KernelRegistry::instance().registerKernel(Name("Op1").Backend("aarch64").Build("kernel1", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(Name("Op2").Backend("aarch64").Build("kernel2", "test.cpp", 2));

    auto kernels = KernelRegistry::instance().listKernels();
    EXPECT_EQ(kernels.size(), static_cast<size_t>(2));
}

TEST_F(KernelRegistryTest, DefaultOverwrite) {
    KernelRegistry::instance().registerKernel(Name("OverwriteOp").Backend("aarch64").Build("test_kernel_v1", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(Name("OverwriteOp").Backend("aarch64").Build("test_kernel_v2", "test.cpp", 2));

    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("OverwriteOp", "aarch64");
    ASSERT_TRUE(symbolOpt.has_value());
    EXPECT_EQ(*symbolOpt, "test_kernel_v2");
}

TEST_F(KernelRegistryTest, SpecializationLookup) {
    KernelRegistry::instance().registerKernel(Name("MixedOp").Backend("aarch64").Build("mixed_default", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("MixedOp").Backend("aarch64").TypeConstraint<float>("T0").Build("mixed_f32", "test.cpp", 2));
    KernelRegistry::instance().registerKernel(
        Name("MixedOp").Backend("aarch64").TypeConstraint<std::int32_t>("T0").Build("mixed_i32", "test.cpp", 3));

    auto defaultSymbolOpt = KernelRegistry::instance().lookupKernelSymbol("MixedOp", "aarch64");
    ASSERT_TRUE(defaultSymbolOpt.has_value());
    EXPECT_EQ(*defaultSymbolOpt, "mixed_default");

    auto f32SymbolOpt = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("MixedOp", {{"T0", "float"}}));
    ASSERT_TRUE(f32SymbolOpt.has_value());
    EXPECT_EQ(*f32SymbolOpt, "mixed_f32");

    auto i32SymbolOpt = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("MixedOp", {{"T0", "int32_t"}}));
    ASSERT_TRUE(i32SymbolOpt.has_value());
    EXPECT_EQ(*i32SymbolOpt, "mixed_i32");
}

TEST_F(KernelRegistryTest, SpecializationRequiresExactMatch) {
    KernelRegistry::instance().registerKernel(Name("FallbackOp").Backend("aarch64").Build("fallback_default", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("FallbackOp").Backend("aarch64").TypeConstraint<float>("T0").Build("fallback_f32", "test.cpp", 2));

    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("FallbackOp", {{"T0", "int32_t"}}));
    EXPECT_FALSE(symbolOpt.has_value());
}

TEST_F(KernelRegistryTest, OnlySpecializationsWithoutDefaultIsNotImplicitlyResolvable) {
    KernelRegistry::instance().registerKernel(
        Name("SpecOnlyOp").Backend("aarch64").TypeConstraint<float>("T0").Build("spec_f32", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("SpecOnlyOp").Backend("aarch64").TypeConstraint<std::int32_t>("T0").Build("spec_i32", "test.cpp", 2));

    EXPECT_FALSE(KernelRegistry::instance().lookupKernelSymbol("SpecOnlyOp", "aarch64").has_value());
    auto specSymbolOpt = KernelRegistry::instance().lookupKernelSymbol(makeQuery("SpecOnlyOp", {{"T0", "float"}}));
    ASSERT_TRUE(specSymbolOpt.has_value());
    EXPECT_EQ(*specSymbolOpt, "spec_f32");
}

TEST_F(KernelRegistryTest, BuilderMacroSupportsDefaultRegistration) {
    KernelRegistry::instance().registerKernel(
        Name("MacroOp").Backend("aarch64").Build("macro_kernel", "macro.cpp", 7));

    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("MacroOp", "aarch64");
    ASSERT_TRUE(symbolOpt.has_value());
    EXPECT_EQ(*symbolOpt, "macro_kernel");
}

TEST_F(KernelRegistryTest, ConcurrentAccess) {
    KernelRegistry::instance().registerKernel(
        Name("ConcurrentOp").Backend("aarch64").Build("concurrent_kernel", "test.cpp", 1));

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
}

TEST_F(KernelRegistryTest, SingleSpecializationWithoutDefaultReturnsNullopt) {
    KernelRegistry::instance().registerKernel(
        Name("SingleSpecOp").Backend("aarch64").TypeConstraint<float>("T").Build("single_spec", "test.cpp", 1));

    EXPECT_FALSE(KernelRegistry::instance().lookupKernelSymbol("SingleSpecOp", "aarch64").has_value());

    auto specOpt = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("SingleSpecOp", {{"T", "float"}}));
    ASSERT_TRUE(specOpt.has_value());
    EXPECT_EQ(*specOpt, "single_spec");
}

TEST_F(KernelRegistryTest, EmptyStringInputs) {
    KernelRegistry::instance().registerKernel(Name("").Backend("aarch64").Build("empty_op_kernel", "test.cpp", 1));
    auto emptyOpt = KernelRegistry::instance().lookupKernelSymbol("", "aarch64");
    ASSERT_TRUE(emptyOpt.has_value());
    EXPECT_EQ(*emptyOpt, "empty_op_kernel");

    EXPECT_FALSE(KernelRegistry::instance().lookupKernelSymbol("NonExistent", "").has_value());
}

TEST_F(KernelRegistryTest, MultipleClearAndReRegister) {
    for (int i = 0; i < 3; ++i) {
        KernelRegistry::instance().registerKernel(
            Name("RepeatOp").Backend("aarch64").Build("repeat_v" + std::to_string(i), "test.cpp", i));

        auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("RepeatOp", "aarch64");
        ASSERT_TRUE(symbolOpt.has_value());
        EXPECT_EQ(*symbolOpt, "repeat_v" + std::to_string(i));

        KernelRegistry::instance().clear();
        EXPECT_FALSE(KernelRegistry::instance().lookupKernelSymbol("RepeatOp", "aarch64").has_value());
    }
}

TEST_F(KernelRegistryTest, BackendIsolation) {
    KernelRegistry::instance().registerKernel(
        Name("MultiBackendOp").Backend("aarch64").Build("aarch64_impl", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("MultiBackendOp").Backend("cuda").Build("cuda_impl", "test.cpp", 2));

    auto aarch64Opt = KernelRegistry::instance().lookupKernelSymbol("MultiBackendOp", "aarch64");
    ASSERT_TRUE(aarch64Opt.has_value());
    EXPECT_EQ(*aarch64Opt, "aarch64_impl");

    auto cudaOpt = KernelRegistry::instance().lookupKernelSymbol("MultiBackendOp", "cuda");
    ASSERT_TRUE(cudaOpt.has_value());
    EXPECT_EQ(*cudaOpt, "cuda_impl");

    EXPECT_FALSE(KernelRegistry::instance().lookupKernelSymbol("MultiBackendOp", "x86").has_value());
}

TEST_F(KernelRegistryTest, MultipleTypeConstraintsExactMatch) {
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
    ASSERT_TRUE(match1.has_value());
    EXPECT_EQ(*match1, "f32_i32_impl");

    auto match2 = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("MultiConstraintOp", {{"T1", "double"}, {"T2", "int64_t"}}));
    ASSERT_TRUE(match2.has_value());
    EXPECT_EQ(*match2, "d_i64_impl");

    auto noMatch = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("MultiConstraintOp", {{"T1", "float"}, {"T2", "int64_t"}}));
    EXPECT_FALSE(noMatch.has_value());
}

TEST_F(KernelRegistryTest, ConcurrentMixedRegisterAndLookup) {
    KernelRegistry::instance().registerKernel(
        Name("ConcurrentMixedAnchor").Backend("aarch64").Build("anchor_kernel", "test.cpp", 0));

    const int numThreads = 4;
    const int opsPerThread = 12;
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
                        auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("ConcurrentMixedAnchor", "aarch64");
                        if (!symbolOpt.has_value() || *symbolOpt != "anchor_kernel") {
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
}

TEST_F(KernelRegistryTest, LastRegistrationWinsForSameSignature) {
    KernelRegistry::instance().registerKernel(
        Name("WinningOp").Backend("aarch64").Build("first_version", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("WinningOp").Backend("aarch64").Build("second_version", "test.cpp", 2));
    KernelRegistry::instance().registerKernel(
        Name("WinningOp").Backend("aarch64").Build("third_version", "test.cpp", 3));

    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol("WinningOp", "aarch64");
    ASSERT_TRUE(symbolOpt.has_value());
    EXPECT_EQ(*symbolOpt, "third_version");
}

TEST_F(KernelRegistryTest, ListKernelsAfterMultipleRegistrations) {
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
}

TEST_F(KernelRegistryTest, BuilderChainingVariousCombinations) {
    KernelRegistry::instance().registerKernel(Name("ChainOp").Build("default_backend", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(Name("ChainOp").Backend("custom").Build("custom_backend", "test.cpp", 2));
    KernelRegistry::instance().registerKernel(
        Name("ChainOp").Backend("aarch64").TypeConstraint<float>("T").Build("specialized_f32", "test.cpp", 3));

    auto defaultOpt = KernelRegistry::instance().lookupKernelSymbol("ChainOp");
    ASSERT_TRUE(defaultOpt.has_value());
    EXPECT_EQ(*defaultOpt, "default_backend");

    auto customOpt = KernelRegistry::instance().lookupKernelSymbol("ChainOp", "custom");
    ASSERT_TRUE(customOpt.has_value());
    EXPECT_EQ(*customOpt, "custom_backend");

    auto specOpt = KernelRegistry::instance().lookupKernelSymbol(
        makeQuery("ChainOp", {{"T", "float"}}));
    ASSERT_TRUE(specOpt.has_value());
    EXPECT_EQ(*specOpt, "specialized_f32");
}
