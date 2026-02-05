#ifndef PIMP_OP_VERIFY_TD
#define PIMP_OP_VERIFY_TD

include "mlir/Pass/PassBase.td"

/* Operation Verify pass*/
def PimpOpVerify : Pass<"pimp-op-verify", "ModuleOp"> {
  let summary = "operator verify";
  let constructor = "createPimpOpVerifyPass()";
  let options = [
      Option<"kpGenLibPath", "kpGenLibPath", "std::string", "", "Path to the external binary library file (e.g., .so or .dll).">, 
      Option<"llmGenLibPath", "llmGenLibPath", "std::string", "", "Path to the external binary library file (e.g., .so or .dll).">
    ];
  let dependentDialects = [
    "PimpDialect"
  ];
}

#endif // PIMP_OP_VERIFY_TD