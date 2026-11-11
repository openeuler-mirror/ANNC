// Shared CHECK harness for the pb2stablehlo unit suites (wheel-isolated
// domain, no GTest by design -- see the suite CMakeLists). Inline linkage:
// each test binary gets its own counters.

#pragma once

#include <cstdio>
#include <string>

namespace pb2test {
inline int g_checks = 0;
inline int g_failures = 0;

inline void report(bool ok, const char* expr, int line) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "[pb2-test] FAIL line %d: %s\n", line, expr);
  }
}
}  // namespace pb2test

#define PB2_CHECK(expr) pb2test::report((expr), #expr, __LINE__)

#define PB2_CHECK_EQ(actual, expected)                                   \
  do {                                                                   \
    auto _a = (actual);                                                  \
    auto _b = (expected);                                                \
    pb2test::report(_a == _b, #actual " == " #expected, __LINE__);       \
    if (!(_a == _b))                                                     \
      std::fprintf(stderr, "  actual: %s, expected: %s\n",               \
                   std::to_string(_a).c_str(),                           \
                   std::to_string(_b).c_str());                          \
  } while (0)

#define PB2_CHECK_STREQ(actual, expected)                                \
  do {                                                                   \
    std::string _a = (actual);                                           \
    std::string _b = (expected);                                         \
    pb2test::report(_a == _b, #actual " == " #expected, __LINE__);       \
    if (!(_a == _b))                                                     \
      std::fprintf(stderr, "  actual: \"%s\", expected: \"%s\"\n",       \
                   _a.c_str(), _b.c_str());                              \
  } while (0)

// Exit-code helper for the forked fatal-path cases.
#include <sys/wait.h>
#include <unistd.h>
#define PB2_CHECK_FATAL_EXIT(child_stmt, want_code)                      \
  do {                                                                   \
    pid_t _pid = fork();                                                 \
    if (_pid == 0) {                                                     \
      child_stmt;                                                        \
      _exit(0); /* not reached when fatal */                             \
    }                                                                    \
    int _st = 0;                                                         \
    waitpid(_pid, &_st, 0); /* blocking: reap the child */               \
    pb2test::report(WIFEXITED(_st) && WEXITSTATUS(_st) == (want_code),   \
                    #child_stmt " exits " #want_code, __LINE__);         \
  } while (0)

// Print the summary line and return the process exit code.
#define PB2_TEST_SUMMARY(prefix)                                         \
  do {                                                                   \
    std::fprintf(stderr, "[" prefix "] %d checks, %d failure(s)\n",      \
                 pb2test::g_checks, pb2test::g_failures);                \
    return pb2test::g_failures == 0 ? 0 : 1;                             \
  } while (0)
