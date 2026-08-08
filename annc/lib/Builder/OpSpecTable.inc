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

  // ---- elementwise / activation ----
  specs.push_back(makeSpec({"Mul"}, "atir.Mul", 2, 2, 1, true));
  specs.push_back(makeSpec({"Sub"}, "atir.Sub", 2, 2, 1, true));
  specs.push_back(
      makeSpec({"Div", "RealDiv", "Divide"}, "atir.RealDiv", 2, 2, 1, true));
  specs.push_back(makeSpec({"FloorDiv"}, "atir.FloorDiv", 2, 2, 1, true));
  specs.push_back(makeSpec({"FloorMod"}, "atir.FloorMod", 2, 2, 1, true));
  specs.push_back(makeSpec({"Maximum"}, "atir.Maximum", 2, 2, 1, true));
  specs.push_back(makeSpec({"Minimum"}, "atir.Minimum", 2, 2, 1, true));
  specs.push_back(
      makeSpec({"Relu"}, "atir.Relu", 1, 1, 1, true, kDefaultsRelu));
  specs.push_back(
      makeSpec({"Sigmoid", "Logistic"}, "atir.Logistic", 1, 1, 1, true));
  specs.push_back(makeSpec({"Abs"}, "atir.Abs", 1, 1, 1, true));
  specs.push_back(makeSpec({"Rsqrt"}, "atir.Rsqrt", 1, 1, 1, true));

  // ---- comparisons (no buffer operand) ----
  specs.push_back(makeSpec({"Equal"}, "atir.Compare", 2, 2, 1, false,
                           kDefaultsPredEQ, kMappingIncompatibleShapeError));
  specs.push_back(makeSpec({"Less"}, "atir.Compare", 2, 2, 1, false,
                           kDefaultsPredLT, kMappingIncompatibleShapeError));
  specs.push_back(makeSpec({"NotEqual"}, "atir.Compare", 2, 2, 1, false,
                           kDefaultsPredNE, kMappingIncompatibleShapeError));
  specs.push_back(makeSpec({"Greater"}, "atir.Compare", 2, 2, 1, false,
                           kDefaultsPredGT, kMappingIncompatibleShapeError));
  specs.push_back(makeSpec({"GreaterEqual", "GreaterEqualV2"}, "atir.Compare",
                           2, 2, 1, false, kDefaultsPredGE,
                           kMappingIncompatibleShapeError));
  specs.push_back(makeSpec({"LessEqual"}, "atir.Compare", 2, 2, 1, false,
                           kDefaultsPredLE, kMappingIncompatibleShapeError));
  // LogicalAnd is a bitwise-AND on bool tensors (i32+bool encoding); atir.And
  // needs no buffer operand, same as the compare family.
  specs.push_back(makeSpec({"LogicalAnd"}, "atir.And", 2, 2, 1, false));

  // ---- shape / index ----
  specs.push_back(makeSpec({"Size"}, "atir.Size", 1, 1, 1, false));
  specs.push_back(makeSpec({"ZerosLike"}, "atir.ZerosLike", 1, 1, 1, false));
  specs.push_back(makeSpec({"Cast"}, "atir.Cast", 1, 1, 1, true));
  specs.push_back(makeSpec({"Fill"}, "atir.Fill", 2, 2, 1, true));
  specs.push_back(makeSpec({"Range"}, "atir.Range", 3, 3, 1, true));
  specs.push_back(makeSpec({"Tile"}, "atir.Tile", 2, 2, 1, true));
  specs.push_back(makeSpec({"Slice"}, "atir.Slice", 3, 3, 1, true));
  specs.push_back(makeSpec({"Concat"}, "atir.Concat", 1, kUnbounded, 1, true,
                           kDefaultsConcat));
  specs.push_back(
      makeSpec({"Where", "Select", "SelectV2"}, "atir.Where", 1, 3, 1, true));

  // ---- reduce / math ----
  specs.push_back(makeSpec({"Mean"}, "atir.ReduceMean", 2, 2, 1, true,
                           kDefaultsReduceMean, kMappingKeepDims));
  specs.push_back(makeSpec({"Softmax"}, "atir.Softmax", 1, 1, 1, true));
  specs.push_back(makeSpec({"Pow"}, "atir.Pow", 2, 2, 1, true));
  specs.push_back(makeSpec({"UnsortedSegmentMin"}, "atir.UnsortedSegmentMin", 3,
                           3, 1, true));
  specs.push_back(makeSpec({"TensorScatterUpdate"}, "atir.TensorScatterUpdate",
                           3, 3, 1, true));

  // ---- identity / matrix ----
  specs.push_back(makeSpec({"GatherNd"}, "atir.GatherNd", 2, 2, 1, true));
  specs.push_back(makeSpec({"BatchMatMul", "BatchMatMulV2"}, "atir.BatchMatMul",
                           2, 2, 1, true, {}, {}, transformBatchMatMul));
  specs.push_back(makeSpec({"Dot"}, "atir.Dot", 2, 2, 1, true));

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
  specs.push_back(makeSpec({"SparseTensorDenseMatMul"},
                           "atir.SparseTensorDenseMatMul", 4, 4, 1, true, {},
                           {}, transformSparseTensorDenseMatMul));
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
  // StringToNumber: 1:1 string->numeric; output dtype comes from the TF
  // out_type attr via the frontend (resolved result type), no ATIR attr.
  specs.push_back(
      makeSpec({"StringToNumber"}, "atir.StringToNumber", 1, 1, 1, true));
  specs.push_back(makeSpec({"StaticRegexReplace"}, "atir.StaticRegexReplace", 1,
                           1, 1, true, {}, {}, transformStaticRegexReplace));
  specs.push_back(makeSpec({"StringSplit"}, "atir.StringSplit", 2, 2, 3, false,
                           {}, {}, transformStringSplit));
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
