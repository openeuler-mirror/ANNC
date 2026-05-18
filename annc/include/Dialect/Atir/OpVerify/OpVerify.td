#ifndef ATIR_OP_VERIFY_TD
#define ATIR_OP_VERIFY_TD

include "mlir/Pass/PassBase.td"

/* Operation Verify pass*/
def AtirOpVerify : Pass<"atir-op-verify", "ModuleOp"> {
  let summary = "operator verify";
  let constructor = "createAtirOpVerifyPass()";
  let options = [
      Option<"kpGenLibPath", "kpGenLibPath", "std::string", "", "Comma-separated paths to KP generated libraries (.so files).">,
      Option<"llmGenLibPath", "llmGenLibPath", "std::string", "", "Comma-separated paths to LLM generated libraries (.so files).">
    ];
  let dependentDialects = [
    "AtirDialect"
  ];
}

#endif // ATIR_OP_VERIFY_TD