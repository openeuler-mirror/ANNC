// FusionV2Outline — shared outlining facilities for op-fusion-v2: anchor
// recovery, boundary collection and the move/recreate/parameterize landing.
// Patterns only match and cut; they never create fused ops or reshape members.

#ifndef ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_FUSIONV2OUTLINE_H
#define ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_FUSIONV2OUTLINE_H

#include <string>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace atir {
namespace fusionv2 {

// Strip the '@' suffix (functionalize-branch copy names) and ':port' segment.
std::string stripAnchorSuffixes(llvm::StringRef name);

// Prefer the ':'-free full pb node name among FusedLoc/NameLoc members; empty
// when none.
std::string anchorFromLoc(mlir::Location loc);

// Single-name anchor: only a lone NameLoc is trusted (a CSE FusedLoc names
// an arbitrary use point and must not be wired back to the pb).
std::string singleAnchorName(mlir::Location loc);
mlir::Value peelAssumingYield(mlir::Value v);

// Single-name anchor: only a plain NameLoc ('@'/':' suffixes allowed). A
// CSE-shared constant's FusedLoc would yield an arbitrary use name (the
// adx_1 mis-wiring), so contract naming validates via this; FusedLoc = unnamed.

// Canonical FusionArg.dtype spelling; empty for unknown element types (bail).
std::string dtypeOf(mlir::Type elem);

// The shape dialect is broadcast/shape scaffolding: never a site member nor
// a boundary input.
bool isShapeScaffolding(mlir::Operation *op);

// Sole data user once shape-scaffolding users are ignored; null if not
// unique.
mlir::Operation *singleDataUser(mlir::Value value);

struct ProbeArg {
  mlir::RankedTensorType type;
  mlir::Value value;  // boundary value in main (call argument when outlined)
  std::string tfName;
  std::string role;  // "constant" / "fixed" / "dynamic" (FusionArg spelling)
};

// shape-dialect ops are broadcast/shape scaffolding: never site members or
// boundary inputs.

// The sole data user after ignoring shape scaffolding; null if not unique.

// Entry input-name table: prefer tf.arg_names, fall back to
// tf.entry_function.inputs.
llvm::SmallVector<std::string> entryInputNames(mlir::func::FuncOp main);

// tf_name for ABI boundary values (shared rule): block params take entry
// input names, defined ops their loc anchor, else anchorName+suffix as last
// resort (absent from the pb graph; the converter would drop such a site).
std::string abiNameFor(mlir::func::FuncOp main, mlir::Value value,
                       llvm::StringRef anchorName, llvm::StringRef suffix);

// Rollback guard for operand rebinding: restores originals when outlining
// fails so a half-rewritten site stays matchable for the greedy driver;
// commit() is mandatory on success (members get erased afterwards).
class RewireGuard {
 public:
  explicit RewireGuard(mlir::PatternRewriter &rewriter) : rewriter_(rewriter) {}
  RewireGuard(const RewireGuard &) = delete;
  RewireGuard &operator=(const RewireGuard &) = delete;

  ~RewireGuard() { rollback(); }

  void rewire(mlir::Operation *op, unsigned index, mlir::Value newValue) {
    entries_.push_back({op, index, op->getOperand(index)});
    rewriter_.modifyOpInPlace(op, [&] { op->setOperand(index, newValue); });
  }

  void commit() {
    committed_ = true;
    entries_.clear();
  }

  void rollback() {
    if (committed_) return;
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it)
      rewriter_.modifyOpInPlace(
          it->op, [&] { it->op->setOperand(it->index, it->old); });
    entries_.clear();
  }

 private:
  struct Entry {
    mlir::Operation *op;
    unsigned index;
    mlir::Value old;
  };
  mlir::PatternRewriter &rewriter_;
  llvm::SmallVector<Entry, 4> entries_;
  bool committed_ = false;
};

// Collect site boundary inputs: chain-member operands not produced inside
// the chain. promotedConstants become role-"dynamic" boundaries (V1 stack
// args); fixedValues are pattern-designated fixed boundaries (semantic role).
// Whitelist closure over a legalize-expansion chain (stablehlo/shape/
// arith/tensor plus named atir interior ops); boundary = block args,
// unnamed atir carriers and constants; anything else returns false.
bool collectArithClosure(
    mlir::Value root, llvm::SmallPtrSetImpl<mlir::Operation *> &closure,
    llvm::DenseSet<mlir::Value> &boundary,
    llvm::ArrayRef<llvm::StringRef> atirInteriorOps);

// Members of a data-producing chain in post-order.
void collectDataMembers(mlir::Value root,
                        llvm::SmallPtrSetImpl<mlir::Operation *> &seen,
                        llvm::SmallVectorImpl<mlir::Operation *> &out);

// Cascade port registry owned by the pass: call -> (callee, per-output
// tf_names). Lives exactly one runOnOperation, so stale entries cannot
// outlive the op set they point into.
struct CallPortTable {
  llvm::DenseMap<mlir::Operation *,
                 std::pair<std::string, llvm::SmallVector<std::string>>>
      entries;
};

mlir::LogicalResult collectBoundary(
    mlir::func::FuncOp main, mlir::ArrayRef<mlir::Operation *> members,
    llvm::SmallVectorImpl<ProbeArg> &args,
    llvm::ArrayRef<std::pair<mlir::Value, std::string>> promotedConstants = {},
    llvm::ArrayRef<mlir::Value> fixedValues = {},
    CallPortTable *portTable = nullptr);

// Whitelisted closure from an arith root: stablehlo/shape/arith/tensor plus
// atirInteriorOps-named carrier ops may be traversed; block params, other
// atir results and constants are boundaries; anything else fails (bail).

// Data-member closure: collect defining ops backwards (region internals
// included), stopping at block params, constants (aux clones) and non-Reshape
// atir residues (a Reshape residue is a member). Post-ordered results.

// Land one site (V1 tri-partition: move / recreate / boundary args, all via
// the rewriter). outputs give the site results; tailOps move into the kernel
// in dependency order; contextI64Attrs lands in the fusion.metadata attr.
mlir::LogicalResult outlineSite(mlir::func::FuncOp main,
                                mlir::PatternRewriter &rewriter,
                                llvm::SmallVectorImpl<ProbeArg> &args,
                                mlir::ArrayRef<mlir::Operation *> members,
                                mlir::ArrayRef<mlir::Operation *> tailOps,
                                mlir::ArrayRef<mlir::Value> outputs,
                                const std::string &name,
                                mlir::StringRef patternName,
                                mlir::ArrayRef<std::string> tailAnchors,
                                llvm::ArrayRef<int64_t> contextI64Attrs,
                                CallPortTable &portTable,
                                mlir::ArrayRef<std::string> outputAliases = {});

}  // namespace fusionv2
}  // namespace atir

#endif  // ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_FUSIONV2OUTLINE_H
