// onnx_op_spec_table.inc — declarative ONNX-op -> ATIR-op table.
//
// Kept separate from the TF table (annc/lib/Builder/OpSpecTable.inc) because
// the two frameworks disagree about the *same* op names: ONNX `Sum` is a
// variadic elementwise add while TF `Sum` reduces, and the same op name can
// carry different operands in the two frameworks (ONNX `Gather` takes its axis
// as the `axis` attribute with the TF `GatherV2` axis operand form absent).
// Sharing one table would silently mistranslate both. The engine
// (construction, metadata, diagnostics) is shared; only the rows differ.
//
// Rows are keyed by ONNX op name in the default `ai.onnx` domain. Rows that
// reuse an existing annc::transform* function rely on the adapter having
// normalized the node into the operand shape that transformer expects (for
// example the ONNX `Transpose.perm` attribute is materialized as a second
// operand). See onnx_node_info_adapter.cpp for the normalizations.

namespace {

// Attribute defaults for the generic path. applyAttrMappings only converts an
// incoming attribute into an existing one of the same MLIR type, so a mapped
// attribute needs a type anchor; the anchor's value is also what the emitted
// op keeps when the node carries no such attribute. Values therefore follow
// the ONNX-13 semantics where they differ from the ATIR declaration, and for
// an attribute the resolver itself falls back on (Concat.axis, a required
// attribute from Concat-4 on) the anchor is that same fallback.
Attribute onnxDefBoolFalse(OpBuilder& b) { return b.getBoolAttr(false); }
Attribute onnxDefBoolTrue(OpBuilder& b) { return b.getBoolAttr(true); }
Attribute onnxDefI32Zero(OpBuilder& b) { return b.getI32IntegerAttr(0); }
Attribute onnxDefPredEQ(OpBuilder& b) { return b.getStringAttr("EQ"); }
Attribute onnxDefPredGT(OpBuilder& b) { return b.getStringAttr("GT"); }
Attribute onnxDefPredLT(OpBuilder& b) { return b.getStringAttr("LT"); }
Attribute onnxDefF32NegOne(OpBuilder& b) { return b.getF32FloatAttr(-1.0f); }

const AttrDefault kOnnxDefaultsAdd[] = {{"do_relu", onnxDefBoolFalse},
                                        {"relu_limit", onnxDefF32NegOne}};
const AttrDefault kOnnxDefaultsConcat[] = {{"axis", onnxDefI32Zero},
                                           {"do_relu", onnxDefBoolFalse},
                                           {"relu_limit", onnxDefF32NegOne},
                                           {"round_mode", onnxDefI32Zero}};
const AttrDefault kOnnxDefaultsRelu[] = {{"relu_limit", onnxDefF32NegOne}};
const AttrDefault kOnnxDefaultsCompareEQ[] = {
    {"comparisonDirection", onnxDefPredEQ},
    {"incompatibleShapeError", onnxDefBoolTrue}};
const AttrDefault kOnnxDefaultsCompareGT[] = {
    {"comparisonDirection", onnxDefPredGT},
    {"incompatibleShapeError", onnxDefBoolTrue}};
const AttrDefault kOnnxDefaultsCompareLT[] = {
    {"comparisonDirection", onnxDefPredLT},
    {"incompatibleShapeError", onnxDefBoolTrue}};

const AttrMapping kOnnxMappingConcatAxis[] = {
    {"axis", "axis", AttrKind::Direct, nullptr}};

// Attributes a transformer or the resolver reads straight off NodeInfo::attrs.
// Declaring them as DocumentedDrop tells attachMetadata they are accounted for,
// so the "attribute not mapped to an ATIR attribute" warning stays meaningful
// for genuinely unexpected attributes instead of firing on every Cast/Reshape
// in the model. It also keeps the ONNX diagnostics independent of the TF
// framework whitelist (kHarmlessTfAttrs) that happens to contain some of these
// names.
const AttrMapping kOnnxDocumentedCast[] = {
    {"to", "to", AttrKind::DocumentedDrop, nullptr}};
const AttrMapping kOnnxDocumentedReshape[] = {
    {"allowzero", "allowzero", AttrKind::DocumentedDrop, nullptr}};
const AttrMapping kOnnxDocumentedSoftmax[] = {
    {"axis", "axis", AttrKind::DocumentedDrop, nullptr}};
const AttrMapping kOnnxDocumentedTranspose[] = {
    {"perm", "perm", AttrKind::DocumentedDrop, nullptr}};
const AttrMapping kOnnxDocumentedGather[] = {
    {"axis", "axis", AttrKind::DocumentedDrop, nullptr}};
const AttrMapping kOnnxDocumentedAxes[] = {
    {"axes", "axes", AttrKind::DocumentedDrop, nullptr}};
const AttrMapping kOnnxDocumentedMod[] = {
    {"fmod", "fmod", AttrKind::DocumentedDrop, nullptr}};
const AttrMapping kOnnxDocumentedReduce[] = {
    {"axes", "axes", AttrKind::DocumentedDrop, nullptr},
    {"keepdims", "keepdims", AttrKind::DocumentedDrop, nullptr},
    {"noop_with_empty_axes", "noop_with_empty_axes", AttrKind::DocumentedDrop,
     nullptr}};

}  // namespace

static void populateOnnxOpSpecTable(llvm::SmallVectorImpl<OpSpec>& specs) {
  // ---- constants --------------------------------------------------------
  // Both an initializer turned into a node by the adapter and a real ONNX
  // `Constant` node land here; raw bytes were decoded by the adapter.
  specs.push_back(makeSpec({"Constant"}, "atir.constant", 0, 0, 1,
                           /*needsBuffer=*/false, {}, {}, nullptr,
                           OpKind::Constant));

  // ---- elementwise / activations ----------------------------------------
  // ONNX `Sum` (variadic elementwise add) maps to atir.Add, never to atir.Sum
  // which is the TF reduce. This one row is why the tables are split.
  specs.push_back(makeSpec({"Sum", "Add"}, "atir.Add", 2, kUnbounded, 1, true,
                           kOnnxDefaultsAdd));
  specs.push_back(makeSpec({"Sub"}, "atir.Sub", 2, 2, 1, true));
  specs.push_back(makeSpec({"Mul"}, "atir.Mul", 2, 2, 1, true));
  specs.push_back(makeSpec({"Div"}, "atir.RealDiv", 2, 2, 1, true));
  // fmod=0 is the floor modulus (the sign follows the divisor), exactly what
  // atir.FloorMod computes. Under opset 13 that form is integer-only and
  // integer Mod is vetoed by the adapter's dtype gate; Mod-28 extended fmod=0
  // to floating point, which is the combination this row actually maps.
  specs.push_back(makeSpec({"Mod"}, "atir.FloorMod", 2, 2, 1, true, {},
                           kOnnxDocumentedMod));
  specs.push_back(makeSpec({"Pow"}, "atir.Pow", 2, 2, 1, true));
  specs.push_back(makeSpec({"Min"}, "atir.Minimum", 2, 2, 1, true));
  specs.push_back(makeSpec({"Max"}, "atir.Maximum", 2, 2, 1, true));
  specs.push_back(makeSpec({"Relu"}, "atir.Relu", 1, 1, 1, true,
                           kOnnxDefaultsRelu));
  specs.push_back(makeSpec({"Sigmoid"}, "atir.Logistic", 1, 1, 1, true));
  specs.push_back(makeSpec({"Sqrt"}, "atir.Sqrt", 1, 1, 1, true));
  specs.push_back(makeSpec({"Abs"}, "atir.Abs", 1, 1, 1, true));
  specs.push_back(makeSpec({"Identity"}, "atir.Identity", 1, 1, 1, true, {}, {},
                           annc::transformIdentity));
  specs.push_back(makeSpec({"Reciprocal"}, "atir.Reciprocal", 1, 1, 1, true));
  // Not becomes Compare(x, true, NE): the only boolean negation ATIR has, and
  // one that broadcasts, unlike a materialized boolean tensor.
  specs.push_back(makeSpec({"Not"}, "atir.Compare", 1, 1, 1, false, {}, {},
                           onnxTransformNot));
  // LeakyRelu is deliberately absent. Its ATIR form would be
  // Where(x >= 0, x, alpha * x), which needs atir.Mul on operands of different
  // shapes; atir.Mul has no broadcast trait, and the four measured models feed
  // it a dynamic leading dimension, so no constant can be sized for it. The op
  // therefore falls back to atir.opaque with reason NO_SPEC.

  // ---- comparisons / logic ----------------------------------------------
  specs.push_back(makeSpec({"Equal"}, "atir.Compare", 2, 2, 1, false,
                           kOnnxDefaultsCompareEQ));
  specs.push_back(makeSpec({"Greater"}, "atir.Compare", 2, 2, 1, false,
                           kOnnxDefaultsCompareGT));
  specs.push_back(makeSpec({"Less"}, "atir.Compare", 2, 2, 1, false,
                           kOnnxDefaultsCompareLT));
  specs.push_back(makeSpec({"And"}, "atir.And", 2, 2, 1, false));
  specs.push_back(makeSpec({"Where"}, "atir.Where", 3, 3, 1, true));

  // ---- shape arithmetic --------------------------------------------------
  specs.push_back(makeSpec({"Shape"}, "atir.Shape", 1, 1, 1, /*needsBuffer=*/false));
  specs.push_back(makeSpec({"Cast"}, "atir.Cast", 1, 1, 1, true, {},
                           kOnnxDocumentedCast));
  specs.push_back(makeSpec({"Reshape"}, "atir.Reshape", 2, 2, 1, true, {},
                           kOnnxDocumentedReshape, annc::transformReshape));
  specs.push_back(makeSpec({"Squeeze"}, "atir.Reshape", 1, 1, 1, true, {},
                           kOnnxDocumentedAxes, annc::transformSqueeze));
  specs.push_back(makeSpec({"Unsqueeze"}, "atir.ExpandDims", 2, 2, 1, true, {},
                           kOnnxDocumentedAxes, annc::transformExpandDims));
  specs.push_back(makeSpec({"Transpose"}, "atir.Transpose", 1, 1, 1, true, {},
                           kOnnxDocumentedTranspose, onnxTransformTranspose));
  specs.push_back(makeSpec({"Concat"}, "atir.Concat", 1, kUnbounded, 1, true,
                           kOnnxDefaultsConcat, kOnnxMappingConcatAxis));
  // Gather carries its axis as an attribute in every ONNX opset (1..24), so
  // the row accepts exactly the two operands the schema declares; a third one
  // is not an ONNX Gather and is refused by the adapter's arity check. No
  // attrDefaults: ONNX Gather has no batch_dims attribute (that one belongs to
  // GatherND), and the transformer bakes atir.Gather's batch_dims to 0 -- a
  // table default would be dead code and, at 1, wrong if it ever took effect.
  specs.push_back(makeSpec({"Gather"}, "atir.Gather", 2, 2, 1, true, {},
                           kOnnxDocumentedGather, onnxTransformGather));
  specs.push_back(makeSpec({"Tile"}, "atir.Tile", 2, 2, 1, true));
  specs.push_back(makeSpec({"Expand"}, "atir.broadcast", 2, 2, 1, true, {}, {},
                           annc::transformBroadcast));

  // atir.Slice is TF-shaped: full-rank (begin, size) with stride 1. The
  // transformer converts the ONNX starts/ends/axes/steps form; the adapter
  // vetoes steps != 1 before it gets there.
  specs.push_back(makeSpec({"Slice"}, "atir.Slice", 3, 5, 1, true, {}, {},
                           onnxTransformSlice));

  // ---- matrix ------------------------------------------------------------
  specs.push_back(makeSpec({"MatMul"}, "atir.MatMul", 2, 2, 1, true, {}, {},
                           annc::transformMatMul));

  // ---- reductions --------------------------------------------------------
  // Of the three reduce forms only ReduceSum carries axes as an operand in
  // opset 13; the transformer unifies ReduceMean/ReduceProd/GlobalAveragePool
  // onto the same operand shape. keepdims is applied by the transformer
  // (ONNX defaults it to 1, ATIR's keep_dims to 0), so no attr mapping here.
  specs.push_back(makeSpec({"ReduceSum"}, "atir.Sum", 1, 2, 1, true, {},
                           kOnnxDocumentedReduce, onnxTransformReduce));
  specs.push_back(makeSpec({"ReduceMean"}, "atir.ReduceMean", 1, 1, 1, true,
                           {}, kOnnxDocumentedReduce, onnxTransformReduce));
  specs.push_back(makeSpec({"ReduceProd"}, "atir.Prod", 1, 1, 1, true, {},
                           kOnnxDocumentedReduce, onnxTransformReduce));
  // GlobalAveragePool reduces every spatial dim with keepdims=1, which is
  // exactly atir.ReduceMean over axes [2..rank-1] with keep_dims set.
  specs.push_back(makeSpec({"GlobalAveragePool"}, "atir.ReduceMean", 1, 1, 1,
                           true, {}, kOnnxDocumentedReduce,
                           onnxTransformReduce));

  // ---- misc --------------------------------------------------------------
  // atir.Softmax is last-dimension only; the adapter rejects other axes.
  specs.push_back(makeSpec({"Softmax"}, "atir.Softmax", 1, 1, 1, true, {},
                           kOnnxDocumentedSoftmax));
}
