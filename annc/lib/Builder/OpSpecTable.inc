// OpSpecTable.inc — declarative TF-op -> ATIR-op table.
//
// Each row describes how a TF op maps to an ATIR op. 1:1 ops (uniform output +
// optional buffer + inputs + defaulted attrs) are realized by the generic
// table-driven construction path; transformer-driven ops are registered
// alongside their transformer implementations.

static void populateOpSpecTable(llvm::SmallVectorImpl<OpSpec>& specs) {
  // Constant: raw tensor bytes decoded by buildConstantNode.
  specs.push_back(makeSpec({"Const", "Constant"}, "atir.constant", 0, 0, 1,
                           /*needsBuffer=*/false, {}, {}, nullptr,
                           OpKind::Constant));
  // Generic 1:1 build with a buffer operand and defaulted attributes.
  specs.push_back(makeSpec({"Add", "AddV2", "AddN", "BiasAdd"}, "atir.Add", 1,
                           kUnbounded, 1, /*needsBuffer=*/true, kDefaultsAdd));
  // Generic 1:1 build without a buffer operand.
  specs.push_back(makeSpec({"Shape"}, "atir.Shape", 1, 1, 1, false));
  // Identity: also required by the frontend's per-output Identity boundary.
  specs.push_back(
      makeSpec({"Identity", "IdentityN", "ReadVariableOp", "StopGradient"},
               "atir.Identity", 1, 1, 1, true));

  // ---- transformer-driven ops (non-1:1 shapes / special semantics) ----
  specs.push_back(makeSpec({"MatMul"}, "atir.MatMul", 2, 3, 1, true, {}, {},
                           transformMatMul));
  specs.push_back(makeSpec({"ConcatV2"}, "atir.ConcatV2", 2, kUnbounded, 1,
                           true, {}, {}, transformConcatV2));
  specs.push_back(makeSpec({"Pack", "Stack"}, "atir.Pack", 1, kUnbounded, 1,
                           true, {}, {}, transformPack));
  // Gather (V1: params + indices) and GatherV2 (V1 + axis tensor) map to the
  // same atir.Gather; the transformer tells them apart by input count.
  specs.push_back(makeSpec({"Gather", "GatherV2"}, "atir.Gather", 2, 3, 1, true,
                           {}, {}, transformGather));
  specs.push_back(makeSpec({"StridedSlice"}, "atir.StridedSlice", 4, 4, 1, true,
                           {}, {}, transformStridedSlice));
  specs.push_back(makeSpec({"Pad", "PadV2"}, "atir.Pad", 2, 3, 1, true, {}, {},
                           transformPad));
  specs.push_back(makeSpec({"Reshape"}, "atir.Reshape", 2, 2, 1, true, {}, {},
                           transformReshape));
  specs.push_back(makeSpec({"Squeeze"}, "atir.Reshape", 1, 1, 1, true, {}, {},
                           transformSqueeze));
  specs.push_back(makeSpec({"Transpose"}, "atir.Transpose", 1, 2, 1, true, {},
                           {}, transformTranspose));
  specs.push_back(makeSpec({"ExpandDims"}, "atir.ExpandDims", 1, 2, 1, true, {},
                           {}, transformExpandDims));
  specs.push_back(makeSpec({"BroadcastTo", "Broadcast"}, "atir.broadcast", 1, 2,
                           1, true, {}, {}, transformBroadcast));
  specs.push_back(makeSpec({"Split"}, "atir.Split", 2, 2, kVariableOutputs,
                           false, {}, {}, transformSplit));
  specs.push_back(makeSpec({"Variable", "VariableV2", "VarHandleOp"},
                           "atir.variable", 0, 0, 1, false, {}, {},
                           transformVariable));
  specs.push_back(
      makeSpec({"Sum"}, "atir.Sum", 1, 2, 1, true, {}, {}, transformSum));
  specs.push_back(
      makeSpec({"Prod"}, "atir.Prod", 1, 2, 1, true, {}, {}, transformProd));
  // 1:1 shape but elementwise-special semantics.
  specs.push_back(makeSpec({"Square"}, "atir.Square", 1, 1, 1, true, {}, {},
                           transformSquare));
  specs.push_back(makeSpec({"SquaredDifference"}, "atir.SquaredDifference", 2,
                           2, 1, true, {}, {}, transformSquaredDifference));

  // ---- sparse / segment / control flow ----
  specs.push_back(makeSpec({"SparseToDense"}, "atir.SparseToDense", 4, 4, 1,
                           true, {}, {}, transformSparseToDense));
  specs.push_back(makeSpec({"SparseReshape"}, "atir.SparseReshape", 3, 3, 2,
                           false, {}, {}, transformSparseReshape));
  specs.push_back(makeSpec({"SparseFillEmptyRows"}, "atir.SparseFillEmptyRows",
                           4, 4, 4, false, {}, {},
                           transformSparseFillEmptyRows));
  specs.push_back(
      makeSpec({"SparseSegmentSum", "SparseSegmentSumWithNumSegments"},
               "atir.SparseSegmentSum", 3, 4, 1, true, {}, {},
               transformSparseSegmentSum));
  specs.push_back(
      makeSpec({"SparseSegmentMin", "SparseSegmentMinWithNumSegments"},
               "atir.SparseSegmentMin", 3, 4, 1, true, {}, {},
               transformSparseSegmentMin));
  specs.push_back(
      makeSpec({"SparseSegmentMean", "SparseSegmentMeanWithNumSegments"},
               "atir.SparseSegmentMean", 3, 4, 1, true, {}, {},
               transformSparseSegmentMean));
  specs.push_back(makeSpec({"ResourceGather"}, "atir.ResourceGather", 2,
                           kUnbounded, 1, true, {}, {},
                           transformResourceGather));
  specs.push_back(makeSpec({"StringToHashBucketFast"},
                           "atir.StringToHashBucketFast", 1, 1, 1, true, {}, {},
                           transformStringToHashBucketFast));
  specs.push_back(makeSpec({"Unique"}, "atir.Unique", 1, 1, 2, false, {}, {},
                           transformUnique));
  specs.push_back(makeSpec({"TopK", "TopKV2"}, "atir.TopK", 2, 2, 2, false, {},
                           {}, transformTopK));
  specs.push_back(makeSpec({"Merge"}, "atir.Merge", 1, kUnbounded, 2, false, {},
                           {}, transformMerge));
  specs.push_back(makeSpec({"DynamicPartition"}, "atir.DynamicPartition", 2, 2,
                           kVariableOutputs, false, {}, {},
                           transformDynamicPartition));
  specs.push_back(makeSpec({"ParallelDynamicStitch"},
                           "atir.ParallelDynamicStitch", 2, kUnbounded, 1, true,
                           {}, {}, transformParallelDynamicStitch));
}
