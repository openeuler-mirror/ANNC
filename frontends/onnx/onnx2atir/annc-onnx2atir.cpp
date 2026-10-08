#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "Builder/Builder.h"
#include "Helper.h"
#include "onnx_frontend.h"
#include "onnx_mlir_builder.h"
#include "onnx_node_info_adapter.h"
#include "onnx_tensor_resolver.h"

#define LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING 1

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Conversion/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;
using namespace annc::onnx2atir;

namespace {

// Contract version of the produced ATIR module and of the reports beside it.
// MAJOR = shape-incompatible change, MINOR = pure addition, PATCH = wording.
// Frozen in design decision D5 of
// projects/annc/architecture/onnx-frontend-design.md.
constexpr const char* kContractVersion = "1.0.0";

// Opset whose operator shapes this frontend knows. A model declaring another
// version is refused outright unless --relaxed-opset is given.
constexpr int64_t kSupportedOpset = 13;

struct Options {
  std::string model_path;
  std::string stats_path;
  std::string inference_path;
  bool strict = false;
  bool relaxed_opset = false;
  bool quiet = false;
};

void printUsage(const char* argv0) {
  llvm::errs()
      << "Usage: " << argv0 << " <model.onnx> [options] [mlir-opt options]\n"
      << "\n"
      << "Imports an ONNX model into an ATIR (MLIR) module. Nodes that cannot\n"
      << "be represented faithfully are lowered to atir.opaque with a recorded\n"
      << "reason; the tool never drops a node silently.\n"
      << "\n"
      << "Options:\n"
      << "  --dump-stats <path>      Write the coverage / provenance JSON report\n"
      << "  --dump-inference <path>  Write the per-tensor type/shape inference\n"
      << "  --strict                 Treat any fallback as a hard error (exit 1)\n"
      << "  --relaxed-opset          Import a non-" << kSupportedOpset
      << " opset per-op instead of refusing\n"
      << "  --quiet                  Suppress the non-error summary lines\n"
      << "\n"
      << "Exit codes: 0 = every node mapped, 2 = completed with fallbacks,\n"
      << "            1 = hard error.\n"
      << "\n"
      << "Any remaining arguments are forwarded to the MLIR driver, so the\n"
      << "module can be written with the usual flags, e.g. `-o out.bc` or\n"
      << "`--mlir-print-ir-after-all`.\n";
}

bool writeJsonFile(const std::string& path, const llvm::json::Value& value,
                   std::string& error) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    error = "cannot write '" + path + "': " + ec.message();
    return false;
  }
  out << llvm::formatv("{0:2}", value) << "\n";
  return true;
}

llvm::json::Array shapeToJson(const std::vector<int64_t>& shape, int64_t dynamic) {
  llvm::json::Array array;
  for (int64_t dim : shape) {
    if (dim == dynamic)
      array.push_back(nullptr);
    else
      array.push_back(dim);
  }
  return array;
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

llvm::json::Value buildStatsReport(const Options& options,
                                   const LoadedOnnxModel& model,
                                   const OnnxGraph& graph,
                                   const ResolvedOnnxGraph& resolved,
                                   const AdaptedGraph& adapted,
                                   int64_t opaque_ops_in_module,
                                   int64_t constants_in_module) {
  llvm::json::Object root;
  root["contract_version"] = kContractVersion;
  root["tool"] = "annc-onnx2atir";

  llvm::json::Object model_obj;
  model_obj["path"] = options.model_path;
  model_obj["ir_version"] = static_cast<int64_t>(model.model.ir_version());
  model_obj["producer_name"] = model.model.producer_name();
  model_obj["opset_version"] = model.opset_version;
  llvm::json::Array opsets;
  for (const auto& entry : model.opset_imports) {
    llvm::json::Object item;
    item["domain"] = entry.first;
    item["version"] = entry.second;
    opsets.push_back(std::move(item));
  }
  model_obj["opset_import"] = std::move(opsets);
  root["model"] = std::move(model_obj);

  llvm::json::Object size;
  size["nodes"] = adapted.model_nodes;
  size["initializers"] = adapted.initializers;
  size["runtime_inputs"] = adapted.runtime_inputs;
  size["graph_outputs"] = adapted.output_boundaries;
  size["sparse_initializers"] =
      static_cast<int64_t>(graph.sparse_initializers.size());
  root["graph"] = std::move(size);

  // ---- mapping coverage --------------------------------------------------
  struct OpTally {
    int64_t total = 0;
    int64_t mapped = 0;
    int64_t fallback = 0;
    std::string atir_op;
  };
  int64_t mapped = 0;
  int64_t fallback = 0;
  std::map<std::string, OpTally> per_op;
  std::map<std::string, int64_t> reason_totals;
  std::map<std::string, std::map<std::string, int64_t>> reason_totals_by_op;

  for (const NodeProvenance& entry : adapted.provenance) {
    OpTally& tally = per_op[entry.op_type];
    ++tally.total;
    if (entry.mapped) {
      ++mapped;
      ++tally.mapped;
      if (tally.atir_op.empty()) tally.atir_op = entry.atir_op;
    } else {
      ++fallback;
      ++tally.fallback;
      const std::string reason = entry.reason.empty() ? "UNKNOWN" : entry.reason;
      ++reason_totals[reason];
      ++reason_totals_by_op[reason][entry.op_type];
    }
  }

  llvm::json::Object by_op;
  for (const auto& entry : per_op) {
    llvm::json::Object item;
    item["op_type"] = entry.first;
    item["total"] = entry.second.total;
    item["mapped"] = entry.second.mapped;
    item["fallback"] = entry.second.fallback;
    if (!entry.second.atir_op.empty()) item["atir_op"] = entry.second.atir_op;
    by_op[entry.first] = std::move(item);
  }
  llvm::json::Object reasons;
  for (const auto& entry : reason_totals)
    reasons[entry.first] = entry.second;
  llvm::json::Object reason_ops;
  for (const auto& entry : reason_totals_by_op) {
    llvm::json::Object ops;
    for (const auto& op : entry.second) ops[op.first] = op.second;
    reason_ops[entry.first] = std::move(ops);
  }

  llvm::json::Object coverage;
  coverage["mapped_nodes"] = mapped;
  coverage["fallback_nodes"] = fallback;
  coverage["total_nodes"] = mapped + fallback;
  coverage["mapped_percent"] =
      mapped + fallback == 0
          ? 0.0
          : (100.0 * static_cast<double>(mapped) /
             static_cast<double>(mapped + fallback));
  coverage["fallback_reasons"] = std::move(reasons);
  coverage["fallback_by_reason"] = std::move(reason_ops);
  coverage["by_op"] = std::move(by_op);
  root["import_coverage"] = std::move(coverage);

  // ---- G2: type/shape four-state ----------------------------------------
  llvm::json::Object types;
  types["produced_tensors"] = resolved.produced.tensors;
  types["with_type"] = resolved.produced.with_type;
  types["rank_unknown"] = resolved.produced.no_shape;
  types["sym_dims"] = resolved.produced.sym_dims;
  types["unknown_dims"] = resolved.produced.unknown_dims;
  types["dynamic_dims"] = resolved.produced.dynamic_dims;
  types["declared_in_file"] = resolved.produced_declared_in_file;
  types["inferred_only"] = resolved.produced_inferred_only;
  llvm::json::Array unresolved;
  for (const std::string& name : resolved.unresolved)
    unresolved.push_back(name);
  types["unresolved_tensors"] = std::move(unresolved);
  root["tensor_types"] = std::move(types);

  // ---- G3: conservation --------------------------------------------------
  llvm::json::Object conservation;
  // Every model node has exactly one provenance row, and every row marked
  // opaque must be exactly one atir.opaque in the built module. Sparse
  // initializers are rows too (they become atir.opaque as well) but they are
  // not model nodes, so they are counted on their own axis: folding them into
  // either side of the node comparison would make both checks false by
  // construction on any model that has one.
  int64_t node_rows = 0;
  int64_t opaque_rows = 0;
  for (const NodeProvenance& entry : adapted.provenance) {
    if (!entry.sparse_initializer) ++node_rows;
    if (!entry.mapped) ++opaque_rows;
  }
  conservation["model_nodes"] = adapted.model_nodes;
  conservation["initializer_constants"] = adapted.initializers;
  conservation["output_boundaries"] = adapted.output_boundaries;
  conservation["provenance_entries"] =
      static_cast<int64_t>(adapted.provenance.size());
  conservation["sparse_initializers"] =
      static_cast<int64_t>(graph.sparse_initializers.size());
  conservation["opaque_ops_in_module"] = opaque_ops_in_module;
  conservation["constants_in_module"] = constants_in_module;
  conservation["nodes_accounted"] = node_rows == adapted.model_nodes;
  conservation["opaque_nodes_match"] = opaque_ops_in_module == opaque_rows;
  root["conservation"] = std::move(conservation);

  // ---- G3: per-node provenance ------------------------------------------
  llvm::json::Array provenance;
  for (const NodeProvenance& entry : adapted.provenance) {
    llvm::json::Object item;
    item["node"] = entry.node_name;
    item["op_type"] = entry.op_type;
    if (!entry.domain.empty()) item["domain"] = entry.domain;
    item["decision"] = entry.mapped ? "mapped" : "opaque";
    item["kind"] = entry.sparse_initializer ? "sparse_initializer" : "node";
    if (!entry.atir_op.empty()) item["atir_op"] = entry.atir_op;
    if (!entry.reason.empty()) item["reason"] = entry.reason;
    provenance.push_back(std::move(item));
  }
  root["provenance"] = std::move(provenance);
  return llvm::json::Value(std::move(root));
}

llvm::json::Value buildInferenceReport(const OnnxGraph& graph,
                                       const ResolvedOnnxGraph& resolved) {
  llvm::json::Object root;
  root["contract_version"] = kContractVersion;
  root["tool"] = "annc-onnx2atir";
  // Same shape vocabulary as the recorded onnx.shape_inference dump
  // (projects/annc/data/20260924-ort-annc-perf/onnx-static-stats/
  // shape-inference.json) so the two can be diffed tensor by tensor.
  root["dynamic_dim_marker"] = nullptr;

  // Runtime inputs are part of the comparison surface too: the official
  // inference reports their declared types, and leaving them out would read as
  // a dtype disagreement on every graph input.
  llvm::json::Object inputs;
  for (const std::string& name : graph.inputs) {
    const TensorDescriptor* descriptor = resolved.find(name);
    if (descriptor == nullptr) continue;
    llvm::json::Object item;
    item["dtype"] = descriptor->dtype;
    item["shape"] = shapeToJson(descriptor->shape, kDynamicDim);
    item["rank_known"] = descriptor->rank_known;
    item["graph_input"] = true;
    inputs[name] = std::move(item);
  }
  root["graph_inputs"] = std::move(inputs);

  llvm::json::Object initializers;
  for (const auto& entry : graph.initializers) {
    const TensorDescriptor* descriptor = resolved.find(entry.first);
    if (descriptor == nullptr) continue;
    llvm::json::Object item;
    item["dtype"] = descriptor->dtype;
    item["shape"] = shapeToJson(descriptor->shape, kDynamicDim);
    item["rank_known"] = descriptor->rank_known;
    initializers[entry.first] = std::move(item);
  }
  root["initializers"] = std::move(initializers);

  llvm::json::Object tensors;
  for (const OnnxNode& node : graph.nodes) {
    for (const std::string& name : node.outputs) {
      const TensorDescriptor* descriptor = resolved.find(name);
      if (descriptor == nullptr) continue;
      llvm::json::Object item;
      item["dtype"] = descriptor->dtype;
      item["shape"] = shapeToJson(descriptor->shape, kDynamicDim);
      item["rank_known"] = descriptor->rank_known;
      item["sym_dims"] = descriptor->sym_dims;
      item["unknown_dims"] = descriptor->unknown_dims;
      item["declared_in_file"] = descriptor->declared_in_file;
      tensors[name] = std::move(item);
    }
  }
  for (const std::string& name : graph.outputs) {
    const TensorDescriptor* descriptor = resolved.find(name);
    if (descriptor == nullptr) continue;
    llvm::json::Object item;
    item["dtype"] = descriptor->dtype;
    item["shape"] = shapeToJson(descriptor->shape, kDynamicDim);
    item["rank_known"] = descriptor->rank_known;
    item["graph_output"] = true;
    tensors[name] = std::move(item);
  }
  root["tensors"] = std::move(tensors);
  return llvm::json::Value(std::move(root));
}

// Counts how many ATIR ops of a given name the built module contains.
int64_t countOpsInModule(ModuleOp module, llvm::StringRef opName) {
  int64_t count = 0;
  module.walk([&](Operation* op) {
    if (op->getName().getStringRef() == opName) ++count;
  });
  return count;
}

// Verifies that every ATIR op the coverage report claims to target actually
// exists in the dialect. Guards against a table row pointing at an op that was
// never defined (a defect observed on the TF side with atir.Square).
std::vector<std::string> findMissingAtirOps(const AdaptedGraph& adapted,
                                            MLIRContext& context) {
  std::vector<std::string> missing;
  std::set<std::string> checked;
  for (const NodeProvenance& entry : adapted.provenance) {
    if (!entry.mapped || entry.atir_op.empty()) continue;
    if (!checked.insert(entry.atir_op).second) continue;
    if (entry.atir_op == "atir.constant") continue;  // a builder kind, not a row
    if (!OperationName(entry.atir_op, &context).isRegistered())
      missing.push_back(entry.atir_op);
  }
  return missing;
}

}  // namespace

int main(int argc, char** argv) {
  llvm::cl::ResetAllOptionOccurrences();

  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  Options options;
  std::vector<char*> filtered_argv;
  filtered_argv.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto takeValue = [&](std::string& target) -> bool {
      if (i + 1 >= argc) {
        llvm::errs() << "Error: " << arg << " requires a value.\n";
        return false;
      }
      target = argv[++i];
      return true;
    };
    if (arg == "--dump-stats") {
      if (!takeValue(options.stats_path)) return 1;
    } else if (arg == "--dump-inference") {
      if (!takeValue(options.inference_path)) return 1;
    } else if (arg == "--strict") {
      options.strict = true;
    } else if (arg == "--relaxed-opset") {
      options.relaxed_opset = true;
    } else if (arg == "--quiet") {
      options.quiet = true;
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    } else {
      filtered_argv.push_back(argv[i]);
    }
  }
  if (filtered_argv.size() < 2) {
    llvm::errs() << "Error: no model path given.\n";
    printUsage(argv[0]);
    return 1;
  }
  options.model_path = filtered_argv[1];
  const int filtered_argc = static_cast<int>(filtered_argv.size());

  // ---- stage 1: load -----------------------------------------------------
  annc::onnx2atir::LoadedOnnxModel loaded;
  annc::onnx2atir::OnnxModelLoader loader;
  std::string error;
  if (!loader.load(options.model_path, loaded, error)) {
    llvm::errs() << "Error: " << error << "\n";
    return 1;
  }

  // Opset assertion (design decision D7). The whole-model refusal is the
  // default because a table built for one opset silently mistranslates another.
  if (loaded.opset_version != kSupportedOpset && !options.relaxed_opset) {
    llvm::errs() << "Error: OP_SET_MISMATCH: model declares ai.onnx opset "
                 << loaded.opset_version << ", this frontend supports "
                 << kSupportedOpset
                 << ". Re-run with --relaxed-opset to map it per-op and mark "
                    "the result.\n";
    return 1;
  }

  // ---- stage 2: parse ----------------------------------------------------
  annc::onnx2atir::OnnxGraph graph;
  annc::onnx2atir::OnnxGraphParser parser;
  if (!parser.parse(loaded.model.graph(), graph, error)) {
    llvm::errs() << "Error: " << error << "\n";
    return 1;
  }

  // ---- stage 3: resolve types/shapes -------------------------------------
  annc::onnx2atir::ResolvedOnnxGraph resolved;
  annc::onnx2atir::OnnxTensorResolver resolver;
  if (!resolver.resolve(graph, resolved, error)) {
    llvm::errs() << "Error: " << error << "\n";
    return 1;
  }

  // ---- stage 4: adapt ----------------------------------------------------
  annc::onnx2atir::AdaptOptions adapt_options;
  adapt_options.relaxed_opset = options.relaxed_opset;
  adapt_options.opset_version = loaded.opset_version;
  annc::onnx2atir::AdaptedGraph adapted;
  annc::onnx2atir::OnnxNodeInfoAdapter adapter;
  if (!adapter.adapt(graph, resolved, adapt_options, adapted, error)) {
    llvm::errs() << "Error: " << error << "\n";
    return 1;
  }

  // ---- build the module --------------------------------------------------
  DialectRegistry registry;
  registry.insert<func::FuncDialect, atir::AtirDialect, arith::ArithDialect>();
  atir::registerAllAtirPasses();
  atir::registerAtirConversionPasses();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  (void)context.allowsUnregisteredDialects();

  const std::vector<std::string> missing_ops =
      findMissingAtirOps(adapted, context);

  // The MLIR diagnostic engine drops Warning/Note when no handler is
  // registered; the builder reports unmapped attributes through warnings, so
  // surface them the way annc-tf2atir does. Errors keep the default formatting.
  context.getDiagEngine().registerHandler([](Diagnostic& diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error) return failure();
    llvm::errs() << diag.getLocation() << ": "
                 << (diag.getSeverity() == DiagnosticSeverity::Warning
                         ? "warning: "
                         : "note: ")
                 << diag << '\n';
    return success();
  });

  // The module is created here rather than through ANNCBuilder, because that
  // helper instantiates the TF-spec MLIRBuilder directly; the two module-level
  // attributes below are the ones ANNCBuilder::buildModule sets, plus the
  // contract version this frontend freezes (design decision D5).
  ModuleOp module = ModuleOp::create(UnknownLoc::get(&context),
                                     StringRef("main"));
  module->setAttr("module.state", StringAttr::get(&context, "atir"));
  module->setAttr("annc.contract_version",
                  StringAttr::get(&context, kContractVersion));
  {
    annc::onnx2atir::OnnxMLIRBuilder builder(module);
    if (!builder.buildFromNodes(adapted.nodes)) {
      llvm::errs() << "Error: failed to build an MLIR module from the ONNX "
                      "graph.\n";
      return 1;
    }
  }

  const int64_t opaque_ops = countOpsInModule(module, "atir.opaque");
  const int64_t constants = countOpsInModule(module, "atir.constant");

  // ---- reports -----------------------------------------------------------
  // Node fallbacks drive the summary and the exit code; sparse-initializer
  // rows are fallbacks too but they are not nodes, so they are reported on
  // their own line instead of skewing the node counts.
  int64_t fallback_count = 0;
  int64_t sparse_fallback_count = 0;
  for (const NodeProvenance& entry : adapted.provenance) {
    if (entry.mapped) continue;
    if (entry.sparse_initializer)
      ++sparse_fallback_count;
    else
      ++fallback_count;
  }

  if (!options.stats_path.empty()) {
    llvm::json::Value report =
        buildStatsReport(options, loaded, graph, resolved, adapted, opaque_ops,
                         constants);
    if (!writeJsonFile(options.stats_path, report, error)) {
      llvm::errs() << "Error: " << error << "\n";
      return 1;
    }
  }
  if (!options.inference_path.empty()) {
    llvm::json::Value report = buildInferenceReport(graph, resolved);
    if (!writeJsonFile(options.inference_path, report, error)) {
      llvm::errs() << "Error: " << error << "\n";
      return 1;
    }
  }

  if (!options.quiet) {
    llvm::outs() << "model: " << options.model_path << "\n"
                 << "  opset: " << loaded.opset_version << "\n"
                 << "  nodes: " << adapted.model_nodes
                 << " (mapped " << (adapted.model_nodes - fallback_count)
                 << ", fallback " << fallback_count << ")\n"
                 << "  initializers: " << adapted.initializers
                 << ", runtime inputs: " << adapted.runtime_inputs
                 << ", outputs: " << adapted.output_boundaries << "\n"
                 << "  produced tensors: " << resolved.produced.tensors
                 << " (rank-unknown " << resolved.produced.no_shape
                 << ", symbolic dims " << resolved.produced.sym_dims
                 << ", undeclared " << resolved.produced_inferred_only
                 << ")\n"
                 << "  atir.opaque in module: " << opaque_ops
                 << ", atir.constant in module: " << constants << "\n";
    if (sparse_fallback_count > 0)
      llvm::outs() << "  sparse initializers (atir.opaque): "
                   << sparse_fallback_count << "\n";
    if (fallback_count > 0 || sparse_fallback_count > 0) {
      std::map<std::string, int64_t> by_reason;
      for (const NodeProvenance& entry : adapted.provenance)
        if (!entry.mapped) by_reason[entry.reason] += 1;
      llvm::outs() << "  fallback reasons:";
      for (const auto& entry : by_reason)
        llvm::outs() << " " << entry.first << "=" << entry.second;
      llvm::outs() << "\n";
    }
    for (const std::string& name : missing_ops)
      llvm::outs() << "  warning: table targets " << name
                   << ", which is not a registered ATIR op\n";
  }

  if (options.strict && (fallback_count > 0 || sparse_fallback_count > 0 ||
                         !missing_ops.empty())) {
    llvm::errs() << "Error: --strict: " << fallback_count
                 << " node(s)";
    if (sparse_fallback_count > 0)
      llvm::errs() << " and " << sparse_fallback_count
                   << " sparse initializer(s)";
    llvm::errs() << " fell back to atir.opaque.\n";
    return 1;
  }

  // ---- hand the module to the MLIR driver --------------------------------
  // Mirrors annc-tf2atir: the built module is written to a temporary bytecode
  // file which then becomes the driver's input, so passes and `-o` work as
  // they do for any other MLIR tool.
  // The temp file lives in the system temp directory under a name made unique
  // atomically by createUniqueFile, so concurrent invocations never collide on
  // a shared path; make_scope_exit removes it on every exit path.
  llvm::SmallString<128> temp_model;
  llvm::sys::path::system_temp_directory(/*ErasedOnReboot=*/true, temp_model);
  llvm::sys::path::append(temp_model, "annc-onnx2atir-%%%%%%.bin");

  llvm::SmallString<128> temp_path;
  if (std::error_code ec =
          llvm::sys::fs::createUniqueFile(temp_model, temp_path)) {
    llvm::errs() << "Error: cannot create temp file: " << ec.message() << "\n";
    return 1;
  }
  std::string temp_bin = temp_path.str().str();
  auto temp_cleanup =
      llvm::make_scope_exit([&] { (void)llvm::sys::fs::remove(temp_bin); });
  annc::outputBinary(module, temp_bin);
  filtered_argv[1] = const_cast<char*>(temp_bin.c_str());

  const int result = mlir::asMainReturnCode(
      mlir::MlirOptMain(filtered_argc, filtered_argv.data(),
                        "ANNC ONNX-to-ATIR Bridge\n", registry));

  if (result != 0) return result;
  return (fallback_count == 0 && sparse_fallback_count == 0) ? 0 : 2;
}
