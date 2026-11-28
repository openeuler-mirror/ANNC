// Private aggregate pass defs for the Stablehlo -> Atir residue sink.
// Frontend-agnostic: the two residue forms (F1 stablehlo.custom_call,
// F2 declaration-carrying annc.residue) serve any frontend. Built only under
// ANNC_ENABLE_TF2ATIR_V2 and kept out of the public Conversion/Passes.td so
// default builds stay stablehlo-free.
#ifndef _STABLEHLO_TO_ATIR_PASSES_TD
#define _STABLEHLO_TO_ATIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def StablehloSink : Pass<"residue-sink", "ModuleOp"> {
  let summary = "aggregate residue sink over both carrier forms (F1 custom_call, F2 annc.residue)";
  let dependentDialects = [
    "atir::AtirDialect",
    "stablehlo::StablehloDialect",
    "arith::ArithDialect",
  ];
  let description = [{
    Consumes a frontend artifact (stablehlo + tf residue + shape/arith
    scaffolding) and emits mixed IR (atir + stablehlo + other dialects):
    sink-table residues in BOTH forms — `stablehlo.custom_call` wrappers
    (F1) and declaration-carrying `annc.residue` carriers (F2) — become atir
    (unmapped residues fall back to the registered `atir.unsupported`
    carrier);
    everything else is kept in place and counted.
    Equivalent to running tf-op-sink, custom-call-sink and custom-op-sink.
  }];
  let constructor = "annc::createStablehloSink()";
}


#endif // _STABLEHLO_TO_ATIR_PASSES_TD
