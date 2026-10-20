// anncGemmFusionLevel caches the env value on first use (function-local
// static), so each level is probed in a forked child with its own ANNC_GEMM.
#include "Helper.h"

#include "gtest/gtest.h"
#include <sys/wait.h>
#include <unistd.h>

namespace {

int levelInFork(const char *value) {
  pid_t pid = fork();
  if (pid == 0) {
    if (value) {
      setenv("ANNC_GEMM", value, 1);
    } else {
      unsetenv("ANNC_GEMM");
    }
    _exit(annc::anncGemmFusionLevel());
  }
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

}  // namespace

TEST(GemmFusionLevel, MapsEnvValueToLevel) {
  EXPECT_EQ(0, levelInFork(nullptr));
  EXPECT_EQ(0, levelInFork("0"));
  EXPECT_EQ(1, levelInFork("1"));
  EXPECT_EQ(2, levelInFork("2"));
  EXPECT_EQ(0, levelInFork("3"));
  EXPECT_EQ(0, levelInFork(""));
  EXPECT_EQ(0, levelInFork("garbage"));
  EXPECT_EQ(1, levelInFork(" 1 "));
}
