// Unit tests for the pb2stablehlo common layer's deterministic functions.
//
// This suite lives in the tests tree per the repo contract, but compiles in
// the transform's wheel-isolated domain (the vendored MLIR/LLVM headers and
// libtensorflow_cc.so.2 of the TF 2.20 wheel; see the transform CMakeLists
// for why the project toolchain must not be mixed in), hence plain CHECK
// macros instead of the project GTest.

#include "pb2_test_harness.h"
#include "common/pb2_common.h"


static void TestParseSlot() {
  PB2_CHECK_EQ(ParseSlot("0"), 0);
  PB2_CHECK_EQ(ParseSlot("7"), 7);
  PB2_CHECK_EQ(ParseSlot("99999999999999"), 2147483647);  // overflow clamped
  PB2_CHECK_EQ(ParseSlot("abc"), 0);  // no digits -> strtol 0
  PB2_CHECK_EQ(ParseSlot("-3"), 0);   // negative clamped to 0
}

static void TestResolveRef() {
  InputRef r = ResolveRef("dense");
  PB2_CHECK_STREQ(r.node, "dense");
  PB2_CHECK_EQ(r.slot, 0);
  PB2_CHECK_EQ(r.control, 0);

  r = ResolveRef("dense:2");
  PB2_CHECK_STREQ(r.node, "dense");
  PB2_CHECK_EQ(r.slot, 2);
  PB2_CHECK_EQ(r.control, 0);

  r = ResolveRef("^init");
  PB2_CHECK_STREQ(r.node, "init");
  PB2_CHECK_EQ(r.slot, 0);
  PB2_CHECK_EQ(r.control, 1);
}

// ResolveRef exits on structurally invalid encodings by design (a garbage
// InputRef would silently corrupt the graph downstream), so the fatal paths
// are asserted in a forked child via the exit code (kExitGraphStructure=4).
// The signature stays fatal-by-design on purpose: this file is a verbatim
// migration and every caller relies on the fail-fast contract.
static void TestResolveRefFatalPaths() {
  for (const char* input : {"", "a:b:0", "^", "^a:b"})
    PB2_CHECK_FATAL_EXIT(ResolveRef(input), kExitGraphStructure);
}

static void TestJsonEscape() {
  PB2_CHECK_STREQ(JsonEscape("plain"), "plain");
  PB2_CHECK_STREQ(JsonEscape("a\"b"), "a\\\"b");
  PB2_CHECK_STREQ(JsonEscape("a\\b"), "a\\\\b");
  PB2_CHECK_STREQ(JsonEscape("a\nb\r\t"), "a\\nb\\r\\t");
  PB2_CHECK_STREQ(JsonEscape(std::string("a\001b")), "a\\u0001b");
}

static void TestAttrValueToJson() {
  AttrValue v;
  v.set_i(-42);
  PB2_CHECK_STREQ(AttrValueToJson(v), "-42");

  v.Clear();
  v.set_f(1.5f);
  PB2_CHECK_STREQ(AttrValueToJson(v), "1.5");

  v.Clear();
  v.set_b(true);
  PB2_CHECK_STREQ(AttrValueToJson(v), "true");

  v.Clear();
  v.set_s("v\"x");
  PB2_CHECK_STREQ(AttrValueToJson(v), "\"v\\\"x\"");

  v.Clear();
  v.set_type(tensorflow::DT_FLOAT);
  PB2_CHECK_STREQ(AttrValueToJson(v), "\"DT_FLOAT\"");

  v.Clear();
  auto* dim = v.mutable_shape()->add_dim();
  dim->set_size(2);
  PB2_CHECK_STREQ(AttrValueToJson(v), "{\"dim\":[2]}");
  v.mutable_shape()->set_unknown_rank(true);
  v.mutable_shape()->clear_dim();
  PB2_CHECK_STREQ(AttrValueToJson(v), "{\"unknown_rank\":true}");

  v.Clear();
  v.mutable_list()->add_i(1);
  v.mutable_list()->add_i(2);
  PB2_CHECK_STREQ(AttrValueToJson(v), "[1,2]");

  // Tensors ride the lossless base64 channel.
  v.Clear();
  v.mutable_tensor()->set_dtype(tensorflow::DT_FLOAT);
  PB2_CHECK_STREQ(AttrValueToJson(v).substr(0, 15), "{\"__tensor__\":\"");

  // func attrs cannot ride the channel: null keeps the JSON parseable.
  v.Clear();
  v.mutable_func()->set_name("f");
  PB2_CHECK_STREQ(AttrValueToJson(v), "null");
}

static void TestReservedAttrNames() {
  PB2_CHECK_EQ(IsReservedAttrName("call_target_name"), 1);
  PB2_CHECK_EQ(IsReservedAttrName("residue_reason"), 1);
  PB2_CHECK_EQ(IsReservedAttrName("custom_op"), 1);
  PB2_CHECK_EQ(IsReservedAttrName("custom_attrs"), 1);
  PB2_CHECK_EQ(IsReservedAttrName("dtype_assumed"), 1);
  PB2_CHECK_EQ(IsReservedAttrName("T"), 0);
  PB2_CHECK_EQ(IsReservedAttrName("output_shapes"), 0);
}

static void TestAttrToMlir() {
  mlir::MLIRContext context;
  mlir::Builder b(&context);
  AttrValue v;
  v.set_s("abc");
  auto attr = AttrToMlir(v, b);
  PB2_CHECK(attr != mlir::Attribute());
  // Assert via the printed form: the wheel's vendored MLIR does not export
  // the typed isa/cast convenience members, so the attribute is compared as
  // its canonical text (a StringAttr prints quoted).
  if (attr) {
    std::string printed;
    llvm::raw_string_ostream os(printed);
    attr.print(os);
    PB2_CHECK_STREQ(printed, "\"abc\"");
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
  PB2_TEST_SUMMARY("common-test");
}
