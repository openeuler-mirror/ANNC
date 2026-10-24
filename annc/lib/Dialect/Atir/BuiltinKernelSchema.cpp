#include "Dialect/Atir/BuiltinKernelSchema.h"

using namespace mlir;

namespace atir {

std::optional<CustomOpSchema> getContextKernelSchema(
    StringRef opName, llvm::ArrayRef<int64_t> attributes) {
  if (opName == "MatMulAdd" || opName == "MatMulAddRelu" ||
      opName == "MatMulAddReluWithAddOutput") {
    auto schema = CustomOpSchema::get(opName)
                      .TypeVar("T")
                      .MemRefArg("lhs", 2, "T")
                      .MemRefArg("rhs", 2, "T")
                      .MemRefArg("bias", 1, "T");
    if (opName != "MatMulAddRelu") schema.Result("add", 2, "T");
    if (opName != "MatMulAdd") schema.Result("relu", 2, "T");
    return schema;
  }
  if (opName == "KPFusedDnnEmbeddingWithHashBucket") {
    int64_t numBuckets = attributes.empty() ? 0 : attributes.front();
    return CustomOpSchema::get(opName)
        .TypeVar("T")
        .MemRefArg("input", 1, "")
        .MemRefArg("embedding_weight", 2, "T")
        .I64AttrArg("num_buckets", numBuckets)
        .Result("output", 2, "T");
  }
  if (opName == "KPFusedTargetBehaviorInteraction") {
    return CustomOpSchema::get(opName)
        .TypeVar("T")
        .MemRefArg("batch_input", 3, "T")
        .MemRefArg("weight", 2, "T")
        .MemRefArg("bias", 1, "T")
        .MemRefArg("tile_input", 3, "T")
        .Result("output", 3, "T");
  }
  for (int r1 = 1; r1 <= 2; ++r1) {
    for (int r2 = 1; r2 <= 2; ++r2) {
      const std::string name = "KPFusedEmbeddingActionIdGatherR" +
                               std::to_string(r1) + "x" + std::to_string(r2);
      if (opName != name) continue;
      return CustomOpSchema::get(opName)
          .TypeVar("T")
          .TypeVar("T1")
          .TypeVar("T2")
          .MemRefArg("params", 2, "T")
          .MemRefArg("indices1", r1, "T1")
          .MemRefArg("indices2", r2, "T2")
          .MemRefArg("pack_dim", 0, "")
          .MemRefArg("pack", 0, "")
          .Result("output", 2, "T");
    }
  }
  for (int n = 2; n <= 16; ++n) {
    if (opName != "KPFusedSparseDynamicStitchN" + std::to_string(n)) continue;
    auto schema =
        CustomOpSchema::get(opName).TypeVar("T").MemRefArg("x", 1, "");
    for (int i = 0; i < n; ++i)
      schema.MemRefArg("variable_" + std::to_string(i), 2, "T");
    return schema.Result("output", 2, "T");
  }
  for (StringRef suffix : {"I32Sum", "I32Mean", "I64Sum", "I64Mean"}) {
    if (opName != ("KPFusedSparseSegmentReduce" + suffix).str()) continue;
    return CustomOpSchema::get(opName)
        .TypeVar("T")
        .TypeVar("Tidx")
        .MemRefArg("keys", 2, "")
        .MemRefArg("begin", 1, "")
        .MemRefArg("data", 2, "T")
        .MemRefArg("indices", 1, "Tidx")
        .MemRefArg("begin_1", 1, "")
        .Result("output", 2, "T")
        .Result("slice_output", 0, "");
  }
  if (opName == "KPFusedEmbeddingPadding" ||
      opName == "KPFusedEmbeddingPaddingFast") {
    auto schema = CustomOpSchema::get(opName)
                      .TypeVar("T")
                      .MemRefArg("origin_shape", 1, "")
                      .MemRefArg("input_rows", 0, "")
                      .MemRefArg("pack", 0, "")
                      .MemRefArg("data", 2, "T")
                      .MemRefArg("reshape_sizes", 1, "")
                      .Result("out_padding_rows", 0, "");
    if (opName.ends_with("Fast"))
      return schema.Result("out_reshape_rows", 0, "");
    return schema.Result("out_data", 2, "T");
  }
  for (StringRef suffix : {"I32", "I64", "I32DynLimit", "I64DynLimit"}) {
    if (opName != ("KPFusedSparseReshape" + suffix).str()) continue;
    auto schema = CustomOpSchema::get(opName)
                      .TypeVar("T")
                      .MemRefArg("slice_input", 2, "")
                      .MemRefArg("begin", 1, "")
                      .MemRefArg("pack_const", 0, "T")
                      .MemRefArg("new_shape", 1, "");
    if (suffix.ends_with("DynLimit")) schema.MemRefArg("range_limit", 0, "");
    return schema.Result("out_indices", 2, "").Result("out_shape", 1, "");
  }
  if (opName == "KPFusedSparseSelect") {
    return CustomOpSchema::get(opName)
        .TypeVar("T")
        .MemRefArg("a", 1, "T")
        .MemRefArg("b", 1, "T")
        .MemRefArg("c", 1, "T")
        .MemRefArg("gt", 0, "T")
        .MemRefArg("eq1", 0, "T")
        .MemRefArg("eq2", 0, "T")
        .MemRefArg("eq3", 0, "T")
        .Result("out_x", 2, "")
        .Result("out_y", 2, "")
        .Result("out_w", 2, "");
  }

  if (opName == "MatMul") {
    return CustomOpSchema::get(opName)
        .TypeVar("T")
        .MemRefArg("lhs", 2, "T")
        .MemRefArg("rhs", 2, "T")
        .Result("output", 2, "T");
  }
  if (opName == "KPFusedGather") {
    return CustomOpSchema::get(opName)
        .TypeVar("T")
        .TypeVar("Tkeys")
        .TypeVar("Tbegin")
        .TypeVar("Tindices")
        .MemRefArg("data", 2, "T")
        .MemRefArg("keys", 2, "Tkeys")
        .MemRefArg("begin", 1, "Tbegin")
        .Result("unique_values", 1, "Tkeys")
        .Result("unique_indices", 1, "Tindices")
        .Result("gathered", 2, "T");
  }
  if (opName == "TensordotMatMul" && attributes.empty())
    return CustomOpSchema::get(opName)
        .TypeVar("T")
        .MemRefArg("input", 3, "T")
        .MemRefArg("weight", 2, "T")
        .Result("output", 3, "T");
  if (opName == "LayerNorm" && attributes.empty())
    return CustomOpSchema::get(opName)
        .TypeVar("T")
        .MemRefArg("input", 2, "T")
        .MemRefArg("gamma", 1, "T")
        .MemRefArg("beta", 1, "T")
        .MemRefArg("eps", 0, "T")
        .Result("output", 2, "T");
  if ((opName == "BatchedMatMulAdd" || opName == "BatchedMatMulAdd4") &&
      attributes.empty()) {
    int64_t rank = opName.ends_with("4") ? 4 : 3;
    return CustomOpSchema::get(opName)
        .TypeVar("T")
        .MemRefArg("data", rank, "T")
        .MemRefArg("weight", rank, "T")
        .MemRefArg("bias", rank, "T")
        .Result("output", rank, "T");
  }
  for (int arity : {1, 2, 3, 4, 6, 7, 11}) {
    for (bool flat : {false, true}) {
      if (flat && arity != 1 && arity != 3 && arity != 4) continue;
      if (opName != "MaskedMul" + std::to_string(arity) + (flat ? "Flat" : ""))
        continue;
      if (attributes.size() != 8 || attributes[0] < 1 ||
          attributes[0] > arity || attributes[1] < 1 || attributes[1] > 16 ||
          attributes[2] < 0 || attributes[2] > 64 - attributes[1])
        return std::nullopt;
      auto schema = CustomOpSchema::get(opName).TypeVar("T").TypeVar("TMask");
      for (int i = 0; i < arity; ++i)
        schema.MemRefArg("arg" + std::to_string(i), 3, "T");
      schema.MemRefArg("ids", 2, "TMask");
      for (int64_t i = 0; i < attributes[1] + attributes[2]; ++i)
        schema.Result("output" + std::to_string(i), 3, "T");
      if (flat) schema.Result("flat", 2, "T");
      const char *names[] = {"dataArgs",    "rawMuls", "fedMuls",
                             "argMap",      "sumMask", "interiorKind",
                             "divisorBits", "padding"};
      for (size_t i = 0; i < attributes.size(); ++i)
        schema.I64AttrArg(names[i], attributes[i]);
      return schema;
    }
  }
  return std::nullopt;
}

}  // namespace atir
