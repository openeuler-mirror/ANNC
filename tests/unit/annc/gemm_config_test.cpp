#include <gtest/gtest.h>
#include <unistd.h>

#include <tuple>

#include "Support/GemmConfig.h"

namespace {

// Parse in fresh children so first-use caching cannot leak between cases.
TEST(GemmConfigDeathTest, ModesSelectIndependentOptimizations) {
  for (const char* value : {"0", "1", "2", "3", "99"}) {
    SCOPED_TRACE(value);
    EXPECT_EXIT(
        {
          setenv("ANNC_GEMM", value, 1);
          const auto& config = annc::gemmConfig();
          const int mode = std::atoi(value);
          const auto actual = std::make_tuple(static_cast<int>(config.mode),
                                              config.epilogue, config.prepack);
          const auto expected = std::make_tuple(mode, mode == 2 || mode == 99,
                                                mode == 3 || mode == 99);
          _exit(actual == expected ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "");
  }
}

TEST(GemmConfigDeathTest, UnsetAndEmptyDefaultToOff) {
  for (const char* value : {static_cast<const char*>(nullptr), "", " \t\n"}) {
    EXPECT_EXIT(
        {
          if (value)
            setenv("ANNC_GEMM", value, 1);
          else
            unsetenv("ANNC_GEMM");
          const auto& config = annc::gemmConfig();
          _exit(config.mode == annc::GemmMode::Off && !config.epilogue &&
                        !config.prepack
                    ? 0
                    : 1);
        },
        ::testing::ExitedWithCode(0), "");
  }
}

TEST(GemmConfigDeathTest, AcceptsSurroundingWhitespace) {
  EXPECT_EXIT(
      {
        setenv("ANNC_GEMM", " \t99\r\n", 1);
        const auto& config = annc::gemmConfig();
        _exit(config.mode == annc::GemmMode::All && config.epilogue &&
                      config.prepack
                  ? 0
                  : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(GemmConfigDeathTest, InvalidModesDiagnoseAndFallBackToOff) {
  for (const char* value : {"4", "98", "100", "-1", "+3", "03", "2x", "prepack",
                            "all", "99999999999999999999999"}) {
    SCOPED_TRACE(value);
    EXPECT_EXIT(
        {
          setenv("ANNC_GEMM", value, 1);
          setenv("ANNC_LOG_LEVEL", "off", 1);
          const auto& config = annc::gemmConfig();
          _exit(config.mode == annc::GemmMode::Off && !config.epilogue &&
                        !config.prepack
                    ? 0
                    : 1);
        },
        ::testing::ExitedWithCode(0),
        "invalid ANNC_GEMM.*expected 0=off, 1=basic, 2=epilogue, 3=prepack, "
        "99=all; falling back to 0=off");
  }
}

TEST(GemmConfigDeathTest, FirstUseFreezesAllFeatures) {
  for (const char* initial : {"0", "1", "2", "3", "99"}) {
    EXPECT_EXIT(
        {
          setenv("ANNC_GEMM", initial, 1);
          const auto first = annc::gemmConfig();
          setenv("ANNC_GEMM", *initial == '0' ? "99" : "0", 1);
          const auto& next = annc::gemmConfig();
          _exit(std::tie(first.mode, first.epilogue, first.prepack) ==
                        std::tie(next.mode, next.epilogue, next.prepack)
                    ? 0
                    : 1);
        },
        ::testing::ExitedWithCode(0), "");
  }
}

TEST(GemmConfigDeathTest, LogsModeNameAtInitialization) {
  EXPECT_EXIT(
      {
        setenv("ANNC_GEMM", "2", 1);
        setenv("ANNC_LOG_LEVEL", "info", 1);
        (void)annc::gemmConfig();
        (void)annc::gemmConfig();
        _exit(0);
      },
      ::testing::ExitedWithCode(0),
      "ANNC_GEMM=2 \\(epilogue\\), epilogue=1, prepack=0");
}

}  // namespace
