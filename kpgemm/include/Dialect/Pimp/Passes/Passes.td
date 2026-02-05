#ifndef PIMP_ALL_PASSES_TD
#define PIMP_ALL_PASSES_TD

include "mlir/Pass/PassBase.td"

/* Tiling */
def PimpTiling : Pass<"pimp-tiling", "ModuleOp"> {
  let summary = "tensor tiling";
  let constructor = "createPimpTilingPass()";
  let dependentDialects = [
    "PimpDialect"
  ];
}

/* Canonicalize */
def PimpCanonicalize : Pass<"pimp-canonicalize", "ModuleOp"> {
  let summary = "constant canonicalize";
  let constructor = "createPimpCanonicalizePass()";
  let dependentDialects = [
    "PimpDialect"
  ];
}

/* Operation Fusion */
def PimpOpFusion : Pass<"pimp-op-fusion", "ModuleOp"> {
  let summary = "operator fusion";
  let constructor = "createPimpOpFusionPass()";
  let dependentDialects = [
    "PimpDialect"
  ];
}


/* matmul add relu Fusion */
def PimpFastCodegen : Pass<"pimp-fast-codegen", "ModuleOp"> {
  let summary = "fast codegen";
  let constructor = "createPimpFastCodegenPass()";
  let dependentDialects = [
    "PimpDialect"
  ];
}

/* distribute pass */
def PimpDistribute : Pass<"pimp-distribute", "ModuleOp"> {
  let summary = "distribute";
  let constructor = "createPimpDistributePass()";
  let dependentDialects = [
    "PimpDialect"
  ];
}

/* block-fusion pass */
def PimpBlockFusion : Pass<"pimp-block-fusion", "ModuleOp"> {
  let summary = "block fusion";
  let constructor = "createPimpBlockFusionPass()";
  let dependentDialects = [
    "PimpDialect"
  ];
}

/* prune-func pass */
def PimpPruneFunc : Pass<"pimp-prune-func", "ModuleOp"> {
  let summary = "prune func";
  let constructor = "createPimpPruneFuncPass()";
  let dependentDialects = [
    "PimpDialect"
  ];
}

/* unroll pass */
def PimpUnroll : Pass<"pimp-unroll", "ModuleOp"> {
  let summary = "unroll";
  let constructor = "createPimpUnrollPass()";
  let dependentDialects = [
    "PimpDialect"
  ];
}

/* eltwise pass */
def PimpEltwiseFusion : Pass<"pimp-Eltwise", "ModuleOp"> {
  let summary = "eltwise fusion";
  let constructor = "createPimpEltwiseFusionPass()";
  let dependentDialects = [
    "PimpDialect"
  ];
}

/* llm pass */
def PimpLLMCodeGen : Pass<"pimp-LLM-CodeGen", "ModuleOp"> {
  let summary = "LLM CodeGen";
  let constructor = "createPimpLLMCodeGenPass()";
  let dependentDialects = [
    "PimpDialect"
  ];
}

#endif // PIMP_ALL_PASSES_TD