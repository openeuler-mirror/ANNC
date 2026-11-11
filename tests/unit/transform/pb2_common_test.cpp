// Unit tests for the pb2stablehlo common layer's deterministic functions.
//
// This suite lives in the tests tree per the repo contract, but compiles in
// the transform's wheel-isolated domain (the vendored MLIR/LLVM headers and
// libtensorflow_cc.so.2 of the TF 2.20 wheel; see the transform CMakeLists
// for why the project toolchain must not be mixed in), hence plain CHECK
// macros instead of the project GTest.

#include <cstdio>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#include "common/pb2_common.h"

namespace {
int g_failures = 0;
int g_checks = 0;

void report(bool ok, const char* expr, int line) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "[common-test] FAIL line %d: %s\n", line, expr);
  }
}
#define CHECK_EQ(actual, expected)                                    \
  do {                                                                \
    auto _a = (actual);                                               \
    auto _b = (expected);                                             \
    report(_a == _b, #actual " == " #expected, __LINE__);             \
    if (!(_a == _b))                                                  \
      std::fprintf(stderr, "  actual: %s, expected: %s\n",            \
                   std::to_string(_a).c_str(),                        \
                   std::to_string(_b).c_str());                       \
  } while (0)
#define CHECK_STREQ(actual, expected)                                 \
  do {                                                                \
    std::string _a = (actual);                                        \
    std::string _b = (expected);                                      \
    report(_a == _b, #actual " == " #expected, __LINE__);             \
    if (!(_a == _b))                                                  \
      std::fprintf(stderr, "  actual: \"%s\", expected: \"%s\"\n",    \
                   _a.c_str(), _b.c_str());                           \
  } while (0)
}  // namespace

static void TestParseSlot() {
  CHECK_EQ(ParseSlot("0"), 0);
  CHECK_EQ(ParseSlot("7"), 7);
  CHECK_EQ(ParseSlot("99999999999999"), 2147483647);  // overflow clamped
  CHECK_EQ(ParseSlot("abc"), 0);  // no digits -> strtol 0
  CHECK_EQ(ParseSlot("-3"), 0);   // negative clamped to 0
}

static void TestResolveRef() {
  InputRef r = ResolveRef("dense");
  CHECK_STREQ(r.node, "dense");
  CHECK_EQ(r.slot, 0);
  CHECK_EQ(r.control, 0);

  r = ResolveRef("dense:2");
  CHECK_STREQ(r.node, "dense");
  CHECK_EQ(r.slot, 2);
  CHECK_EQ(r.control, 0);

  r = ResolveRef("^init");
  CHECK_STREQ(r.node, "init");
  CHECK_EQ(r.slot, 0);
  CHECK_EQ(r.control, 1);
}

// ResolveRef exits on structurally invalid encodings by design (a garbage
// InputRef would silently corrupt the graph downstream), so the fatal paths
// are asserted in a forked child via the exit code (kExitGraphStructure=4).
// The signature stays fatal-by-design on purpose: this file is a verbatim
// migration and every caller relies on the fail-fast contract.
static void TestResolveRefFatalPaths() {
  const char* bad[] = {"", "a:b:0", "^", "^a:b"};
  for (const char* input : bad) {
    pid_t pid = fork();
    if (pid == 0) {
      ResolveRef(input);
      _exit(0);  // not reached when fatal
    }
    int status = 0;
    waitpid(pid, &status, 0);  /* blocking: reap the child */
    report(WIFEXITED(status) && WEXITSTATUS(status) == 4, input,
           __LINE__);
  }
}

static void TestJsonEscape() {
  CHECK_STREQ(JsonEscape("plain"), "plain");
  CHECK_STREQ(JsonEscape("a\"b"), "a\\\"b");
  CHECK_STREQ(JsonEscape("a\\b"), "a\\\\b");
  CHECK_STREQ(JsonEscape("a\nb\r\t"), "a\\nb\\r\\t");
  CHECK_STREQ(JsonEscape(std::string("a\001b")), "a\\u0001b");
}

static void TestAttrValueToJson() {
  AttrValue v;
  v.set_i(-42);
  CHECK_STREQ(AttrValueToJson(v), "-42");

  v.Clear();
  v.set_f(1.5f);
  CHECK_STREQ(AttrValueToJson(v), "1.5");

  v.Clear();
  v.set_b(true);
  CHECK_STREQ(AttrValueToJson(v), "true");

  v.Clear();
  v.set_s("v\"x");
  CHECK_STREQ(AttrValueToJson(v), "\"v\\\"x\"");

  v.Clear();
  v.set_type(tensorflow::DT_FLOAT);
  CHECK_STREQ(AttrValueToJson(v), "\"DT_FLOAT\"");

  v.Clear();
  auto* dim = v.mutable_shape()->add_dim();
  dim->set_size(2);
  CHECK_STREQ(AttrValueToJson(v), "{\"dim\":[2]}");
  v.mutable_shape()->set_unknown_rank(true);
  v.mutable_shape()->clear_dim();
  CHECK_STREQ(AttrValueToJson(v), "{\"unknown_rank\":true}");

  v.Clear();
  v.mutable_list()->add_i(1);
  v.mutable_list()->add_i(2);
  CHECK_STREQ(AttrValueToJson(v), "[1,2]");

  // Tensors ride the lossless base64 channel.
  v.Clear();
  v.mutable_tensor()->set_dtype(tensorflow::DT_FLOAT);
  CHECK_STREQ(AttrValueToJson(v).substr(0, 15), "{\"__tensor__\":\"");

  // func attrs cannot ride the channel: null keeps the JSON parseable.
  v.Clear();
  v.mutable_func()->set_name("f");
  CHECK_STREQ(AttrValueToJson(v), "null");
}

static void TestReservedAttrNames() {
  CHECK_EQ(IsReservedAttrName("call_target_name"), 1);
  CHECK_EQ(IsReservedAttrName("residue_reason"), 1);
  CHECK_EQ(IsReservedAttrName("custom_op"), 1);
  CHECK_EQ(IsReservedAttrName("custom_attrs"), 1);
  CHECK_EQ(IsReservedAttrName("dtype_assumed"), 1);
  CHECK_EQ(IsReservedAttrName("T"), 0);
  CHECK_EQ(IsReservedAttrName("output_shapes"), 0);
}

static void TestAttrToMlir() {
  mlir::MLIRContext context;
  mlir::Builder b(&context);
  AttrValue v;
  v.set_s("abc");
  auto attr = AttrToMlir(v, b);
  report(attr != mlir::Attribute(), "AttrToMlir(s) produced an attribute",
         __LINE__);
  // Assert via the printed form: the wheel's vendored MLIR does not export
  // the typed isa/cast convenience members, so the attribute is compared as
  // its canonical text (a StringAttr prints quoted).
  if (attr) {
    std::string printed;
    llvm::raw_string_ostream os(printed);
    attr.print(os);
    CHECK_STREQ(printed, "\"abc\"");
  }
}

int main() {
  TestParseSlot();
  TestResolveRef();
  TestResolveRefFatalPaths();
  TestJsonEscape();
  TestAttrValueToJson();
  TestReservedAttrNames();
  TestAttrToMlir();
  std::fprintf(stderr, "[common-test] %d checks, %d failure(s)\n", g_checks,
               g_failures);
  return g_failures == 0 ? 0 : 1;
}
