#ifndef ATIR_ALL_PASSES_TD
#define ATIR_ALL_PASSES_TD

include "mlir/Pass/PassBase.td"

/* Identity Canonicalize */
def AtirIdentityCanonicalize : Pass<"atir-identity-canonicalize", "ModuleOp"> {
  let summary = "eliminate redundant ATIR Identity operations";
  let constructor = "createAtirIdentityCanonicalizePass()";
  let dependentDialects = [
    "AtirDialect"
  ];
}

/* Ordered, hardware-independent GEMM epilogue fusion. */
def AtirGemmEpilogueFusion
    : Pass<"atir-gemm-epilogue-fusion", "ModuleOp"> {
  let summary = "fuse an ordered GEMM elementwise epilogue";
  let constructor = "createAtirGemmEpilogueFusionPass()";
  let dependentDialects = ["AtirDialect"];
}

/* BatchNorm fold */
def AtirFoldBatchNorm : Pass<"atir-fold-batch-norm", "ModuleOp"> {
  let summary = "fold inference BatchNorm primitive chains into MatMul weights";
  let constructor = "createAtirFoldBatchNormPass()";
  let dependentDialects = [
    "AtirDialect"
  ];
}

/* Rank inference */
def AtirRankInference : Pass<"atir-rank-inference", "ModuleOp"> {
  let summary = "infer tensor ranks for rank_known=false values from TF op semantics";
  let constructor = "createAtirRankInferencePass()";
  let dependentDialects = [
    "AtirDialect"
  ];
}

/* Operation Fusion */
def AtirOpFusion : Pass<"atir-op-fusion", "ModuleOp"> {
  let summary = "operator fusion";
  let constructor = "createAtirOpFusionPass()";
  let dependentDialects = [
    "AtirDialect"
  ];
}

/* Configurable Fusion */
def AtirConfigFusion : Pass<"atir-config-fusion", "ModuleOp"> {
  let summary = "configurable operator fusion";
  let constructor = "createAtirConfigFusionPass()";
  let dependentDialects = [
    "AtirDialect"
  ];
  let options = [
    Option<"config", "config", "std::string", "",
           "Path to external fusion pattern JSON config.">,
    Option<"runBuiltinAfter", "run-builtin-after", "bool", "true",
           "Run builtin fusion after user patterns.">,
    Option<"report", "report", "std::string", "",
           "Optional path for the config fusion JSON report.">,
    Option<"warnLimit", "warn-limit", "int", "10",
           "Maximum skip warning samples per pattern; 0 disables samples.">
  ];
}

/* matmul add relu Fusion */
def AtirFastCodegen : Pass<"atir-fast-codegen", "ModuleOp"> {
  let summary = "fast codegen";
  let constructor = "createAtirFastCodegenPass()";
  let dependentDialects = [
    "AtirDialect"
  ];
  let options = [
    Option<"enableKdnn", "enable-kdnn", "bool",
           /*defaultValue=*/"false",
           "Enable KDNN optimized kernels where available. "
           "Kernels with a KDNN implementation will use it; "
           "others fall back to aarch64.">,
    ListOption<"enabledCustomOps", "enable-custom-ops", "std::string",
               "Only rewrite the listed custom op types. Empty enables all types.">,
    ListOption<"disabledCustomOps", "disable-custom-ops", "std::string",
               "Do not rewrite the listed custom op types. Takes precedence over enable-custom-ops.">
  ];
}

/* prune-func pass */
def AtirPruneFunc : Pass<"atir-prune-func", "ModuleOp"> {
  let summary = "prune func";
  let constructor = "createAtirPruneFuncPass()";
  let options = [
    Option<"executionMode", "execution-mode", "std::string", "",
           "Keep only ANNC kernel functions with this execution mode">
  ];
  let dependentDialects = [
    "AtirDialect"
  ];
}

/* Select one fused kernel from a fusion-only runtime template. */
def AtirSelectKernel : Pass<"atir-select-kernel", "ModuleOp"> {
  let summary = "select one fused kernel function for runtime compilation";
  let constructor = "createAtirSelectKernelPass()";
  let options = [
    Option<"kernelName", "kernel-name", "std::string", "",
           "Symbol name of the fused kernel to retain">
  ];
  let dependentDialects = ["AtirDialect"];
}

/* Bind runtime argument shapes before lowering. */
def AtirSpecializeShapes : Pass<"atir-specialize-shapes", "ModuleOp"> {
  let summary = "specialize fused kernel arguments with runtime shapes";
  let constructor = "createAtirSpecializeShapesPass()";
  let options = [
    Option<"shapeSpec", "shape-spec", "std::string", "",
           "Path to the runtime shape specification JSON">
  ];
  let dependentDialects = ["AtirDialect"];
}

#endif // ATIR_ALL_PASSES_TD
