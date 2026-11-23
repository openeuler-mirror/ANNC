#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/framework/op.h"

using tensorflow::AttrValue;
using tensorflow::DataType;
using tensorflow::GraphDef;
using tensorflow::NodeDef;
using tensorflow::OpDef;

namespace xla {
void RegisterAllHloDialects(mlir::DialectRegistry& registry);
}  // namespace xla

namespace tensorflow {

class Session;

// Mirror of the .so-internal struct (no pip header); every bool defaulted.
struct GraphdefToMlirOptions {
  std::string debug_info_file;
  std::string xla_compile_device_type;
  bool prune_unused_nodes = false;
  bool convert_legacy_fed_inputs = false;
  bool graph_as_function = false;
  bool upgrade_legacy = false;
  bool enable_shape_inference = false;
  bool unconditionally_use_set_output_shapes = false;
  bool enable_soft_placement = false;
  bool set_original_tf_func_name = false;
};

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GraphdefToMlirTranslateFunction(
    llvm::StringRef input, absl::string_view input_arrays,
    absl::string_view input_dtypes, absl::string_view input_shapes,
    absl::string_view output_arrays, absl::string_view control_output_arrays,
    const GraphdefToMlirOptions& import_options, mlir::MLIRContext* context);

namespace quantization {
absl::Status PreprocessAndFreezeGraph(
    absl::string_view mlir_dump_file_prefix, bool is_inliner_run,
    const absl::flat_hash_set<std::string>& noinline_functions,
    mlir::ModuleOp module_op, mlir::MLIRContext* context,
    std::optional<Session*> session, bool run_tf_to_stablehlo,
    bool deserialize_xla_call_module,
    llvm::ArrayRef<llvm::ArrayRef<int64_t>> input_arg_shapes);
}  // namespace quantization
}  // namespace tensorflow

namespace shlo_pipeline {
void AddTFToStablehloPasses(mlir::OpPassManager& pm);
}  // namespace shlo_pipeline

struct InputRef {
  std::string node;      // referenced node name
  int slot = 0;          // output slot; 0 for control edges
  bool control = false;  // true = "^node" control edge
};

struct NodeInfo {
  const NodeDef* def;  // original NodeDef (read-only source of truth)
  std::vector<DataType> out_dtypes;  // per-slot dtype
};

extern std::map<std::string, NodeInfo> g_nodes;  // node index (deterministic)
extern std::vector<std::string> g_order;       // file order
extern std::map<std::string, OpDef> g_opspecs;  // OpDef registry
extern int g_warn;  // non-fatal warning counter

struct BuildCtx;
struct Options;

// Exit codes shared across the route (single definition; fatal paths use
// these, never bare numbers).
enum ExitCode {
  kExitOk = 0,
  kExitUsage = 1,
  kExitArgs = 2,
  kExitInit = 3,
  kExitGraphStructure = 4,
  kExitInternal = 5,  // importer/functionalize/pipeline stage failure
};

// Map-free attr lookup (Map::find unreliable across the ODR boundary).
const AttrValue* FindAttr(const NodeDef& nd, const std::string& key);
// "node" / "node:slot" / "^node"; fatal on any other encoding (a garbage
// InputRef would silently corrupt the graph downstream).
InputRef ResolveRef(const std::string& s);
// Non-fatal variant for scans over inputs that may carry other encodings
// (e.g. ref edges "a:b:0" inside BuildUnregisteredOpDefs): nullopt when the
// string is not one of the three standard forms. Scans must use this, so
// parsing semantics never fork a fourth time.
std::optional<InputRef> TryResolveRef(const std::string& s);
// Digit string -> slot (strtol with overflow clamp; no digits -> 0).
int ParseSlot(const char* s);

// The one attr->OpDef-default fallback (node attr value first, spec default
// second, nullopt if neither). Hand-rolled copies of this chain have caused
// exactly the drift the ODR notes warn about -- use these.
std::optional<long> AttrIntOrSpecDefault(const NodeDef& nd, const OpDef& spec,
                                         const std::string& attr_name);
std::optional<DataType> AttrTypeOrSpecDefault(const NodeDef& nd,
                                              const OpDef& spec,
                                              const std::string& attr_name);

// ==== Boundary op names (ABI contract; single source of truth) ====
// The four wrap boundaries ride stablehlo.custom_call call_target_name;
// the name is the dispatch ABI (kernel batch later). Constants prevent
// drift; tests intentionally hard-code the same strings as anchors.
inline constexpr char kGetDim[] = "annc.getDim";
inline constexpr char kSetDim[] = "annc.setDim";
inline constexpr char kEncodeString[] = "annc.encodeString";
inline constexpr char kDecodeString[] = "annc.decodeString";

// Attribute names reserved by the tf.residue wrapper / dual attr channel:
// original attrs with these names are skipped on copy and markers written
// last, preventing duplicate DictionaryAttr keys.
inline bool IsReservedAttrName(llvm::StringRef n) {
  return n == "call_target_name" || n == "residue_reason" ||
         n == "custom_op" || n == "custom_attrs" || n == "dtype_assumed";
}
inline void CopyNonReservedAttrs(mlir::OperationState& st, mlir::Operation* op) {
  for (auto& na : op->getAttrs())
    if (!IsReservedAttrName(na.getName()))
      st.addAttribute(na.getName(), na.getValue());
}
// Post-build module verify: tolerates unregistered-op ODS mismatch, prints the
// first few diagnostics. Returns the failure count.
int VerifyBuiltModule(mlir::MLIRContext& context, mlir::ModuleOp module);
void LoadOpSpecs();  // wheel registry only; no external OpList input

std::vector<int> MapInputsToArgs(const NodeDef& nd, const OpDef& spec);

// Output dtype derivation; DT_INVALID = unknown (backfill or fail later).
DataType InferOutDtype(const NodeDef& nd, int slot);

struct EntrySpec {
  DataType dt = tensorflow::DT_INVALID;  // invalid until resolved
  std::vector<int64_t> dims;
  bool has_shape = false;   // false = unranked
  bool from_graph = false;  // dtype inferred from the graph node
};
struct BuildCtx {  // cross-stage state (the five sets route every node)
  std::map<std::string, EntrySpec> entries;      // parsed --input entries
  std::set<std::string> input_set;               // entry node names
  std::set<std::string> argified;                // Consts promoted to args
  std::set<std::string> var_arg;                 // variables promoted to args
  std::set<std::string> var_skip;                // variables + read chain (skip)
};
struct Options {
  std::string graph_path;        // graph pb
  std::string out_path;          // -o <file>: artifact sink; empty = stdout
  std::string crash_repro_path;  // --crash_repro; empty = disabled
                                 // (dump prefix then falls back to /tmp)
  std::vector<std::string> inputs, outputs;  // raw: name[:DT[:dims]] / name
  bool dump_tfmlir = false, dump_out = true,
       skip_preprocess = false;
  bool assume_opaque_i8 = false;
};

// Original NodeDefs of unregistered custom ops (keyed by node name) for the
// post-import verbatim restore.
using CustomOpTable = std::map<std::string, NodeDef>;

struct FeedSpec {
  std::string name;
  DataType dt = tensorflow::DT_INVALID;
  std::vector<int64_t> dims;
  bool has_shape = false;
};

// Complement rewrite at the pb level (after dtype inference/PlanWeights):
// custom ops -> per-slot Placeholders, argified Consts / parametrized
// variables -> same-name Placeholders, read chains removed and consumers
// rewired; dangling control edges dropped. Returns 0 on success.
int RewriteGraphForImport(const Options& opt, BuildCtx& ctx,
                          const GraphDef& graph, GraphDef& gp,
                          CustomOpTable& custom_ops,
                          std::vector<FeedSpec>& feeds);

// GraphDef-level v1 control-flow functionalization (official tf2xla bridge,
// applied to the raw graph): Switch/Merge clusters -> tf.If + branch
// FunctionDefs in out.library(). Only graphs with residual Switches enter.
int FunctionalizeGraphDefControlFlow(const GraphDef& src, GraphDef& out);

// Imports gp into MLIR via the official translator (feeds/fetches from opt).
int ImportOfficial(const Options& opt, const GraphDef& gp,
                   const std::vector<FeedSpec>& feeds,
                   mlir::MLIRContext& context,
                   mlir::OwningOpRef<mlir::ModuleOp>& module);

// Structural flattening of tf_executor graphs into plain blocks (islands
// lifted, control edges dropped, functionalized carriers moved out).
void FlattenExecutor(mlir::ModuleOp module);

// MLIR-side restore: per-scope loc-name scan; custom ops rebuilt verbatim
// from custom_ops (dual attr channel), operands re-linked via in-block
// names / feeds / tf.If capture block args; tf.If wrapped as tf.residue;
// tf.entry_function rewritten to the real entries.
int RestoreOps(mlir::ModuleOp module, const Options& opt,
               const std::vector<FeedSpec>& feeds,
               const CustomOpTable& custom_ops, mlir::OpBuilder& b,
               mlir::MLIRContext& context);

// Official-import route orchestration: GraphPassManager (pb domain) ->
// ImportOfficial -> MLIR PassManager (normalize/flatten/restore/G2) ->
// RunPipelines (legalize + G3 reanchor). Stage implementations live beside
// the passes they drive.
int RunOfficialRoute(const Options& opt, BuildCtx& ctx, const GraphDef& graph,
                     mlir::OpBuilder& b, mlir::MLIRContext& context);

// Official-inference enrichment (best-effort): post-rewrite graph is all
// registered ops + Placeholders, so the official GraphConstructor/ShapeRefiner
// can derive output shapes/ranks; results are written back as _output_shapes
// (never overwriting complete Grappler facts). Failures degrade silently.
int EnrichGraphWithOfficialInference(GraphDef& gp, const Options& opt,
                                     const BuildCtx& ctx);

// Parses --input specs, derives dtypes graph-wide, overrides entries, and
// hard-fails on remaining DT_INVALID (unless --assume-opaque-i8, which marks
// them DT_INT8 plus an explicit dtype_assumed attr for disambiguation).
void InferEntryAndDtypes(const Options& opt, BuildCtx& ctx);

// Slots marked by --assume-opaque-i8 (node -> slots). i8 alone is ambiguous;
// dtype_assumed (MLIR attr + custom_attrs JSON) carries the explicit marker.
extern std::map<std::string, std::vector<int>> g_assumed_dtypes;

// Registry for unregistered custom ops; standard ops fall back to the
// in-process TF registry.
class SynthOpRegistry : public tensorflow::OpRegistryInterface {
 public:
  explicit SynthOpRegistry(std::map<std::string, tensorflow::OpDef> synth)
      : synth_(std::move(synth)) {}

  absl::Status LookUp(
      const std::string& op,
      const tensorflow::OpRegistrationData** out) const override;

 private:
  std::map<std::string, tensorflow::OpDef> synth_;
};

// Synthetic type-attr keys injected by BuildUnregisteredOpDefs (op type ->
// {t0..,o0..}). Exact-key erasure avoids deleting real attrs named t<N>/o<N>.
extern std::map<std::string, std::set<std::string>> g_synth_attr_keys;

// Builds synthetic OpDefs for unregistered custom ops (wildcard input
// type_list + output type attrs) and injects per-slot dtypes into a copy of
// the graph (fg != nullptr) so Graph construction passes. Shape inference is
// left to the official importer (host-side ShapeRefiner is unusable across
// the .so boundary).
void BuildUnregisteredOpDefs(
    const GraphDef& src, std::map<std::string, tensorflow::OpDef>& synth,
    GraphDef* fg);

// Promotes unrepresentable Consts (e.g. DT_STRING vocabs) and, with
// --parametrize, variables + read chains out of the artifact as entry args.
void PlanWeights(const Options& opt, mlir::Builder& b, BuildCtx& ctx);

// Custom-op (unregistered) attr dual channel: MLIR-convertible attrs direct (skip/warn on
// the rest), then custom_op + sorted custom_attrs JSON. Used by the
// official-import restore path.
void AppendCustomOpAttrs(mlir::OperationState& st, const NodeDef& nd,
                        mlir::OpBuilder& b, const std::string& name);

// Official preprocess (optional) + tf->stablehlo pipeline + residue
// report. graph feeds the post-legalize constant reanchor (G3).
int RunPipelines(const Options& opt, const BuildCtx& ctx,
                 const GraphDef& graph, mlir::MLIRContext& context,
                 mlir::ModuleOp& module);

// AttrValue -> MLIR attribute; null if the form is unsupported (caller warns).
mlir::Attribute AttrToMlir(const AttrValue& v, mlir::Builder& b);

// JSON string escape.
std::string JsonEscape(const std::string& s);
// AttrValue -> JSON for custom_attrs. Tensors are encoded losslessly as
// {"__tensor__":"<b64(TensorProto)>"} (lists element-wise); func -> null.
std::string AttrValueToJson(const AttrValue& v);
