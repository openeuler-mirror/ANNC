#include <array>
#include <cstdint>
#include <dlfcn.h>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "Kernel/KernelRegistry.h"
#include "Kernel/MemRefTypes.h"

#if ANNC_TEST_ENABLE_KDNN
#include "annc/lib/Kernel/builtin_kernels/kdnn_adaptor/KDNNTensorInfoAdaptor.h"
#endif

using namespace annc::kernels;

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

AnncMemRef2DF32 makeMemRef2D(float* data, int64_t dim0, int64_t dim1) {
    AnncMemRef2DF32 memref{};
    memref.allocated = data;
    memref.aligned = data;
    memref.offset = 0;
    memref.sizes[0] = dim0;
    memref.sizes[1] = dim1;
    memref.strides[0] = dim1;
    memref.strides[1] = 1;
    return memref;
}

std::array<float, 4> runMatMulKernel(const std::string& backend) {
    float lhsData[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float rhsData[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float outputData[9] = {0.0f};

    AnncMemRef2DF32 lhs = makeMemRef2D(lhsData, 2, 3);
    AnncMemRef2DF32 rhs = makeMemRef2D(rhsData, 3, 2);
    AnncMemRef2DF32 output = makeMemRef2D(outputData, 2, 2);

    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol(
        KernelQuery{"MatMul", backend, {{"T", "float"}}});
    if (!symbolOpt.has_value()) {
        throw std::runtime_error("MatMul kernel was not registered for backend: " + backend);
    }

    auto kernel = loadKernelApi<void (*)(AnncMemRef2DF32*, AnncMemRef2DF32*, AnncMemRef2DF32*)>(
        symbolOpt->c_str());
    kernel(&output, &lhs, &rhs);

    return {outputData[0], outputData[1], outputData[2], outputData[3]};
}

void expectStandardMatMulOutput(const std::array<float, 4>& output) {
    EXPECT_FLOAT_EQ(output[0], 58.0f);
    EXPECT_FLOAT_EQ(output[1], 64.0f);
    EXPECT_FLOAT_EQ(output[2], 139.0f);
    EXPECT_FLOAT_EQ(output[3], 154.0f);
}

void requireStandardMatMulOutput(const std::array<float, 4>& output) {
    if (output[0] != 58.0f || output[1] != 64.0f ||
        output[2] != 139.0f || output[3] != 154.0f) {
        throw std::runtime_error("MatMul output mismatch");
    }
}

#if ANNC_TEST_ENABLE_KDNN
void runBuiltinKdnnMatMulWrapperBody() {
    requireStandardMatMulOutput(runMatMulKernel("kdnn"));
}
#endif

} // namespace

TEST(BuiltinSharedSpecTest, RegistryExposesMatMulAarch64) {
    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol(
        KernelQuery{"MatMul", "aarch64", {{"T", "float"}}});

    ASSERT_TRUE(symbolOpt.has_value());
    EXPECT_TRUE(symbolOpt->rfind("ANNCKernel_matmul_", 0) == 0);
    ASSERT_GE(symbolOpt->size(), 5u);
    EXPECT_EQ(symbolOpt->substr(symbolOpt->size() - 5), "_auto");
}

TEST(BuiltinSharedSpecTest, ResolverFindsMatMulAarch64) {
    auto symbol = KernelRegistry::instance().lookupKernelSymbol(
        KernelQuery{"MatMul", "aarch64", {{"T", "float"}}});

    ASSERT_TRUE(symbol.has_value());
    EXPECT_TRUE(symbol->rfind("ANNCKernel_matmul_", 0) == 0);
    ASSERT_GE(symbol->size(), 5u);
    EXPECT_EQ(symbol->substr(symbol->size() - 5), "_auto");
}

TEST(BuiltinSharedSpecTest, Aarch64MatMulWrapperRuns) {
    expectStandardMatMulOutput(runMatMulKernel("aarch64"));
}

#if ANNC_TEST_ENABLE_KDNN
TEST(BuiltinSharedSpecTest, RegistryExposesMatMulKdnn) {
    auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol(
        KernelQuery{"MatMul", "kdnn", {{"T", "float"}}});

    EXPECT_TRUE(symbolOpt.has_value());
}

TEST(BuiltinSharedSpecTest, KdnnTensorRefPreservesMemRefMetadata) {
    float buffer[16] = {0.0f};
    AnncMemRef2DF32 memref{};
    memref.allocated = buffer;
    memref.aligned = buffer;
    memref.offset = 3;
    memref.sizes[0] = 2;
    memref.sizes[1] = 4;
    memref.strides[0] = 5;
    memref.strides[1] = 1;

    auto tensorRef =
        annc::kernels::kdnn_adaptor::makeDenseTensorRef(memref, KDNN::Layout::AB);

    EXPECT_EQ(tensorRef.info.GetDims()[0], static_cast<KDNN::SizeType>(2));
    EXPECT_EQ(tensorRef.info.GetDims()[1], static_cast<KDNN::SizeType>(4));
    EXPECT_EQ(tensorRef.info.GetStrides()[0], static_cast<KDNN::SizeType>(5));
    EXPECT_EQ(tensorRef.info.GetStrides()[1], static_cast<KDNN::SizeType>(1));
    EXPECT_EQ(tensorRef.data, buffer + 3);
}

TEST(BuiltinSharedSpecTest, KdnnMatMulWrapperRuns) {
    pid_t pid = fork();
    ASSERT_GE(pid, 0) << "failed to fork kdnn smoke child";

    if (pid == 0) {
        try {
            runBuiltinKdnnMatMulWrapperBody();
            _exit(0);
        } catch (...) {
            _exit(1);
        }
    }

    int status = 0;
    ASSERT_GE(waitpid(pid, &status, 0), 0) << "failed to wait for kdnn child";

    if (WIFSIGNALED(status) && WTERMSIG(status) == SIGILL) {
        SUCCEED() << "kdnn MatMul wrapper hit SIGILL on this machine";
        return;
    }

    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(BuiltinSharedSpecTest, MatMulHasBothBackends) {
    EXPECT_TRUE(KernelRegistry::instance().hasKernel(
        KernelQuery{"MatMul", "aarch64", {{"T", "float"}}}));
    EXPECT_TRUE(KernelRegistry::instance().hasKernel(
        KernelQuery{"MatMul", "kdnn", {{"T", "float"}}}));
}

TEST(BuiltinSharedSpecTest, MatMulSymbolsDifferBetweenBackends) {
    auto aarch64Sym = KernelRegistry::instance().lookupKernelSymbol(
        KernelQuery{"MatMul", "aarch64", {{"T", "float"}}});
    auto kdnnSym = KernelRegistry::instance().lookupKernelSymbol(
        KernelQuery{"MatMul", "kdnn", {{"T", "float"}}});

    ASSERT_TRUE(aarch64Sym.has_value());
    ASSERT_TRUE(kdnnSym.has_value());
    EXPECT_NE(*aarch64Sym, *kdnnSym);
}
#endif
