#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/pb2_common.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace pb_passes {

std::unique_ptr<mlir::Pass> CreateNormalizeRefTypesPass();
std::unique_ptr<mlir::Pass> CreateFlattenExecutorPass();
std::unique_ptr<mlir::Pass> CreateStripFunctionalizeSuffixPass();
// Needs the raw pb graph: derived constants are renamed back to their pb
// origin via the (host, slot) input edge, value-vetoed.
std::unique_ptr<mlir::Pass> CreateRestoreConstSourceLocPass(
    const tensorflow::GraphDef& graph);
// Retypes unranked results from pb _output_shapes annotations (add-info only).
std::unique_ptr<mlir::Pass> CreateAnnotateShapesFromPbPass(
    const tensorflow::GraphDef& graph);
// Converges unranked branch-function args to the callers' operand types.
std::unique_ptr<mlir::Pass> CreateConvergeSignaturesPass();
// Casts index-typed shape operands (shape.* chains) to i32 for stablehlo ODS.
std::unique_ptr<mlir::Pass> CreateNormalizeShapeVectorsPass();
std::unique_ptr<mlir::Pass> CreateAnchorConstantsToUsersPass();
// Post-legalize guard: remaining registered tf.* ops -> tf.residue.
std::unique_ptr<mlir::Pass> CreateTagConstSourceNamePass();
std::unique_ptr<mlir::Pass> CreateWrapRegisteredResiduePass();

// RestoreOps as a pass; failures carry the legacy route exit code (1 / 4).
std::unique_ptr<mlir::Pass> CreateRestoreOpsPass(
    const Options& opt, const std::vector<FeedSpec>& feeds,
    const CustomOpTable& custom_ops, mlir::OpBuilder& b);

// Single-pass runner preserving the legacy RestoreOps contract: prints the
// fatal message and exits(4) on invariant failures, returns 1 on recoverable
// ones (no func / feed count mismatch), 0 on success.
// Legacy contract: may terminate the process (exit 4) on a fatal
// invariant -- unwinding a half-restored module is unsafe past that point.
int RunRestoreOpsPass(mlir::ModuleOp module, const Options& opt,
                      const std::vector<FeedSpec>& feeds,
                      const CustomOpTable& custom_ops, mlir::OpBuilder& b,
                      mlir::MLIRContext& context);

// Renames the residue carrier tf.residue -> annc.residue. Run AFTER the
// odml legalize pipeline (post-legalize wrap included): the tf. prefix is
// load-bearing while the wheel processes the module; past that point the
// rename is purely local and the sink consumes the F3 form name-independently.
std::unique_ptr<mlir::Pass> CreateRenameResidueCarrierPass();
// Tags annc.source_port on expansion-group ops whose value escapes the group
// (evidence chain: final use edge + consumer pb attribution + pb input spec
// port declaration). Undetermined cases are left untagged by design.
std::unique_ptr<mlir::Pass> CreatePortTagPass(
    const tensorflow::GraphDef& graph);

}  // namespace pb_passes
