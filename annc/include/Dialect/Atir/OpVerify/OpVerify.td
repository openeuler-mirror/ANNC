#ifndef ATIR_OP_VERIFY_TD
#define ATIR_OP_VERIFY_TD

include "mlir/Pass/PassBase.td"

/* Operation Verify pass*/
def AtirOpVerify : Pass<"atir-op-verify", "ModuleOp"> {
  let summary = "operator verify";
  let constructor = "createAtirOpVerifyPass()";
  let options = [
      Option<"kpGenLibPath", "kpGenLibPath", "std::string", "", "Path to the external binary library file (e.g., .so or .dll).">, 
      Option<"llmGenLibPath", "llmGenLibPath", "std::string", "", "Path to the external binary library file (e.g., .so or .dll).">
    ];
  let dependentDialects = [
    "AtirDialect"
  ];
}

#endif // ATIR_OP_VERIFY_TD