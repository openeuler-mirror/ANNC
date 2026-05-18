#ifndef ATIR_ALL_PASSES_TD
#define ATIR_ALL_PASSES_TD

include "mlir/Pass/PassBase.td"

/* Tiling */
def AtirTiling : Pass<"atir-tiling", "ModuleOp"> {
  let summary = "tensor tiling";
  let constructor = "createAtirTilingPass()";
  let dependentDialects = [
    "AtirDialect"
  ];
}

/* Canonicalize */
def AtirCanonicalize : Pass<"atir-canonicalize", "ModuleOp"> {
  let summary = "constant canonicalize";
  let constructor = "createAtirCanonicalizePass()";
  let dependentDialects = [
    "AtirDialect"
  ];
}

/* Select Lowering Strategy */
def AtirSelectLoweringStrategy : Pass<"atir-select-lowering-strategy", "ModuleOp"> {
  let summary = "select lowering strategy";
  let constructor = "createAtirSelectLoweringStrategyPass()";
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


/* matmul add relu Fusion */
def AtirFastCodegen : Pass<"atir-fast-codegen", "ModuleOp"> {
  let summary = "fast codegen";
  let constructor = "createAtirFastCodegenPass()";
  let dependentDialects = [
    "AtirDialect"
  ];
}

/* distribute pass */
def AtirDistribute : Pass<"atir-distribute", "ModuleOp"> {
  let summary = "distribute";
  let constructor = "createAtirDistributePass()";
  let dependentDialects = [
    "AtirDialect"
  ];
}

/* block-fusion pass */
def AtirBlockFusion : Pass<"atir-block-fusion", "ModuleOp"> {
  let summary = "block fusion";
  let constructor = "createAtirBlockFusionPass()";
  let dependentDialects = [
    "AtirDialect"
  ];
}

/* prune-func pass */
def AtirPruneFunc : Pass<"atir-prune-func", "ModuleOp"> {
  let summary = "prune func";
  let constructor = "createAtirPruneFuncPass()";
  let dependentDialects = [
    "AtirDialect"
  ];
}

/* unroll pass */
def AtirUnroll : Pass<"atir-unroll", "ModuleOp"> {
  let summary = "unroll";
  let constructor = "createAtirUnrollPass()";
  let dependentDialects = [
    "AtirDialect"
  ];
}

/* eltwise pass */
def AtirEltwiseFusion : Pass<"atir-Eltwise", "ModuleOp"> {
  let summary = "eltwise fusion";
  let constructor = "createAtirEltwiseFusionPass()";
  let dependentDialects = [
    "AtirDialect"
  ];
}

/* llm pass */
def AtirLLMCodeGen : Pass<"atir-LLM-CodeGen", "ModuleOp"> {
  let summary = "LLM CodeGen";
  let constructor = "createAtirLLMCodeGenPass()";
  let dependentDialects = [
    "AtirDialect"
  ];
}

#endif // ATIR_ALL_PASSES_TD