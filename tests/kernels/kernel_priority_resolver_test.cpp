#include <string>

#include <gtest/gtest.h>

#include "Kernel/KernelPriorityResolver.h"
#include "Kernel/KernelRegistry.h"

using namespace annc::kernels;

namespace {

KernelResolveRequest req(const std::string& opType) {
    KernelResolveRequest r;
    r.op_type = opType;
    return r;
}

class KernelPriorityResolverTest : public ::testing::Test {
protected:
    void SetUp() override { KernelRegistry::instance().clear(); }
    void TearDown() override { KernelRegistry::instance().clear(); }
};

} // namespace

TEST_F(KernelPriorityResolverTest, HasAnyAvailableKernelAarch64OnlyWithKdnnDisabledReturnsTrue) {
    KernelRegistry::instance().registerKernel(
        Name("HasTest1").Backend("aarch64").Build("test_aarch64", "test.cpp", 1));

    EXPECT_TRUE(hasAnyAvailableKernel(req("HasTest1"), false));
}

TEST_F(KernelPriorityResolverTest, HasAnyAvailableKernelAarch64OnlyWithKdnnEnabledReturnsTrue) {
    KernelRegistry::instance().registerKernel(
        Name("HasTest1b").Backend("aarch64").Build("test_aarch64", "test.cpp", 1));

    EXPECT_TRUE(hasAnyAvailableKernel(req("HasTest1b"), true));
}

TEST_F(KernelPriorityResolverTest, HasAnyAvailableKernelKdnnOnlyWithKdnnEnabledReturnsTrue) {
    KernelRegistry::instance().registerKernel(
        Name("HasTest2").Backend("kdnn").Build("test_kdnn", "test.cpp", 1));

    EXPECT_TRUE(hasAnyAvailableKernel(req("HasTest2"), true));
}

TEST_F(KernelPriorityResolverTest, HasAnyAvailableKernelKdnnOnlyWithKdnnDisabledReturnsFalse) {
    KernelRegistry::instance().registerKernel(
        Name("HasTest3").Backend("kdnn").Build("test_kdnn", "test.cpp", 1));

    EXPECT_FALSE(hasAnyAvailableKernel(req("HasTest3"), false));
}

TEST_F(KernelPriorityResolverTest, HasAnyAvailableKernelBothBackendsWithKdnnEnabledReturnsTrue) {
    KernelRegistry::instance().registerKernel(
        Name("HasTest4").Backend("aarch64").Build("test_aarch64", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("HasTest4").Backend("kdnn").Build("test_kdnn", "test.cpp", 2));

    EXPECT_TRUE(hasAnyAvailableKernel(req("HasTest4"), true));
}

TEST_F(KernelPriorityResolverTest, HasAnyAvailableKernelBothBackendsWithKdnnDisabledReturnsTrue) {
    KernelRegistry::instance().registerKernel(
        Name("HasTest5").Backend("aarch64").Build("test_aarch64", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("HasTest5").Backend("kdnn").Build("test_kdnn", "test.cpp", 2));

    EXPECT_TRUE(hasAnyAvailableKernel(req("HasTest5"), false));
}

TEST_F(KernelPriorityResolverTest, HasAnyAvailableKernelNoKernelReturnsFalse) {
    EXPECT_FALSE(hasAnyAvailableKernel(req("NonExistentOp"), false));
}

TEST_F(KernelPriorityResolverTest, ResolveBestKernelInfoBothBackendsWithKdnnEnabledReturnsKdnn) {
    KernelRegistry::instance().registerKernel(
        Name("ResTest1").Backend("aarch64").Build("res_aarch64", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("ResTest1").Backend("kdnn").Build("res_kdnn", "test.cpp", 2));

    auto result = resolveBestKernelInfo(req("ResTest1"), true);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->symbol_name, "res_kdnn");
    EXPECT_EQ(result->backend, "kdnn");
}

TEST_F(KernelPriorityResolverTest, ResolveBestKernelInfoBothBackendsWithKdnnDisabledReturnsAarch64) {
    KernelRegistry::instance().registerKernel(
        Name("ResTest2").Backend("aarch64").Build("res_aarch64", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("ResTest2").Backend("kdnn").Build("res_kdnn", "test.cpp", 2));

    auto result = resolveBestKernelInfo(req("ResTest2"), false);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->symbol_name, "res_aarch64");
    EXPECT_EQ(result->backend, "aarch64");
}

TEST_F(KernelPriorityResolverTest, ResolveBestKernelInfoAarch64OnlyWithKdnnEnabledReturnsAarch64) {
    KernelRegistry::instance().registerKernel(
        Name("ResTest3").Backend("aarch64").Build("res_aarch64", "test.cpp", 1));

    auto result = resolveBestKernelInfo(req("ResTest3"), true);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->symbol_name, "res_aarch64");
}

TEST_F(KernelPriorityResolverTest, ResolveBestKernelInfoKdnnOnlyWithKdnnEnabledReturnsKdnn) {
    KernelRegistry::instance().registerKernel(
        Name("ResTest4").Backend("kdnn").Build("res_kdnn", "test.cpp", 1));

    auto result = resolveBestKernelInfo(req("ResTest4"), true);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->symbol_name, "res_kdnn");
}

TEST_F(KernelPriorityResolverTest, ResolveBestKernelInfoKdnnOnlyWithKdnnDisabledReturnsNullopt) {
    KernelRegistry::instance().registerKernel(
        Name("ResTest5").Backend("kdnn").Build("res_kdnn", "test.cpp", 1));

    auto result = resolveBestKernelInfo(req("ResTest5"), false);

    EXPECT_FALSE(result.has_value());
}

TEST_F(KernelPriorityResolverTest, ResolveBestKernelInfoNoKernelReturnsNullopt) {
    auto result = resolveBestKernelInfo(req("NonExistentOp"), false);

    EXPECT_FALSE(result.has_value());
}

TEST_F(KernelPriorityResolverTest, ResolveBestKernelInfoTypedRequestFallsBackOnlyToExplicitAnyType) {
    KernelRegistry::instance().registerKernel(
        Name("TypedFallbackOp").Backend("aarch64").Build("untyped_default", "test.cpp", 1));
    KernelRegistry::instance().registerKernel(
        Name("TypedFallbackOp").Backend("aarch64").AnyType("T").Build("generic_any", "test.cpp", 2));

    KernelResolveRequest request;
    request.op_type = "TypedFallbackOp";
    request.type_constraints = {{"T", "int32_t"}};

    auto result = resolveBestKernelInfo(request, false);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->symbol_name, "generic_any");
}
