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
}
