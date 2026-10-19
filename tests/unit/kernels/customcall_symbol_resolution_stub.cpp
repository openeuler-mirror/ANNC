#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "Kernel/KernelPriorityResolver.h"

using namespace annc::kernels;

namespace {

std::string resolveForCustomCallOrThrow(const std::string& opType, bool enableKdnn) {
    KernelResolveRequest req;
    req.op_type = opType;
    if (opType == "MatMul") {
        req.type_constraints = {{"T", "float"}};
    }
    auto info = resolveBestKernelInfo(req, enableKdnn);
    if (!info.has_value()) {
        throw std::runtime_error("failed to resolve symbol for op: " + opType);
    }
    return info->symbol_name;
}

} // namespace

TEST(CustomCallSymbolResolutionTest, ResolvesMatMulToBuiltinAutoKernel) {
    const std::string matmulSymbol = resolveForCustomCallOrThrow("MatMul", false);

    EXPECT_TRUE(matmulSymbol.rfind("ANNCKernel_matmul_", 0) == 0);
    ASSERT_GE(matmulSymbol.size(), 5u);
    EXPECT_EQ(matmulSymbol.substr(matmulSymbol.size() - 5), "_auto");
}

TEST(CustomCallSymbolResolutionTest, MissingOpThrows) {
    EXPECT_THROW(
        (void)resolveForCustomCallOrThrow("DefinitelyMissingOp", false),
        std::runtime_error);
}
