#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <string_view>

#include "common/pb2_common.h"

// Tiny Placeholder->Identity graph; importing it registers all TF dialects.
static std::string MakeRegistrationGraphDef() {
  GraphDef g;
  auto* x = g.add_node();
  x->set_name("x");
  x->set_op("Placeholder");
  (*x->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
  auto* y = g.add_node();
  y->set_name("y");
  y->set_op("Identity");
  y->add_input("x");
  (*y->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
  return g.SerializeAsString();
}

// CLI parsing, data-driven: one table row per option; unknown options and
// missing values are fatal (exit 2), the positional arg is the graph path.
static int ParseArgs(int argc, char** argv, Options& opt) {
  struct Opt {
    llvm::StringRef name;
    bool takes_value;
    std::function<void(Options&, const char*)> set;
  };
  static const Opt kOpts[] = {
      {"--input", true, [](Options& o, const char* v) { o.inputs.push_back(v); }},
      {"--output", true, [](Options& o, const char* v) { o.outputs.push_back(v); }},
      {"-o", true, [](Options& o, const char* v) { o.out_path = v; }},
      {"--dump", false, [](Options& o, const char*) { o.dump_tfmlir = true; }},
      {"--no-dump-out", false, [](Options& o, const char*) { o.dump_out = false; }},
      // --parametrize deprecated: variable parametrization is now
      // unconditional (graphs with variables fail official import without
      // it -- resource handle types cannot be Placeholder-expressed).
      {"--parametrize", false,
       [](Options&, const char*) {
         std::cerr << "[warn] --parametrize deprecated: variable "
                      "parametrization is now unconditional\n";
       }},
      {"--skip-preprocess", false, [](Options& o, const char*) { o.skip_preprocess = true; }},
      {"--assume-opaque-i8", false, [](Options& o, const char*) { o.assume_opaque_i8 = true; }},
      {"--crash_repro", true, [](Options& o, const char* v) { o.crash_repro_path = v; }},
  };
  auto usage = [&argv](llvm::StringRef why) {
    std::cerr << why.str() << "\nusage: " << argv[0]
              << " <graph.pb> [--input name:DT:dims]... [--output name]... "
                 "[-o <file>] [--dump] [--no-dump-out] "
                 "[--skip-preprocess] [--assume-opaque-i8] "
                 "[--crash_repro <file>]\n";
    return 2;
  };
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef a = argv[i];
    auto it = std::find_if(std::begin(kOpts), std::end(kOpts),
                           [&](const Opt& o) { return o.name == a; });
    if (it != std::end(kOpts)) {
      if (it->takes_value && ++i >= argc)
        return usage(std::string("[FATAL] ") + a.str() +
                     " requires a value");
      it->set(opt, it->takes_value ? argv[i] : nullptr);
    } else if (a.starts_with("--")) {
      return usage("[FATAL] unknown option: " + a.str());
    } else {
      opt.graph_path = a.str();
    }
  }
  if (opt.assume_opaque_i8)
    std::cerr << "[warn] --assume-opaque-i8 marks dtype-less slots DT_INT8"
                 " (opaque sentinel; downstream numeric conversion refuses them"
                 " -> fallback runtime dispatch). Regression aid; production"
                 " graphs should carry real dtypes\n";
  if (opt.graph_path.empty()) return usage("[FATAL] missing <graph.pb>");
  return 0;
}

// Reads the GraphDef, loads OpDefs, indexes nodes and sizes output slots.
static int LoadGraph(const Options& opt, GraphDef& graph) {
  std::ifstream in(opt.graph_path, std::ios::binary);
  if (!in) {
    std::cerr << "[FATAL] cannot open graph file: " << opt.graph_path << "\n";
    return 1;
  }
  std::string data(std::istreambuf_iterator<char>(in), {});
  if (!graph.ParseFromString(data)) {
    std::cerr << "GraphDef parse failed: " << opt.graph_path << "\n";
    return 1;
  }
  std::cerr << "[graph] nodes = " << graph.node_size() << " producer = "
            << (graph.has_versions() ? graph.versions().producer() : 0) << "\n";
  LoadOpSpecs();
  for (int i = 0; i < graph.node_size(); ++i) {
    const std::string& nm = graph.node(i).name();
    if (!g_nodes.emplace(nm, NodeInfo{&graph.node(i), {}}).second) {
      std::cerr << "[FATAL] duplicate node name: " << nm << "\n";
      return 1;
    }
    g_order.push_back(nm);
  }
  for (auto& kv : g_nodes) {
    auto spec_it = g_opspecs.find(kv.second.def->op());
    int n_out =
        (spec_it != g_opspecs.end()) ? spec_it->second.output_arg_size() : 1;
    kv.second.out_dtypes.assign(n_out, tensorflow::DT_INVALID);
  }
  {
    std::map<std::string, int> max_slot;
    for (auto& name : g_order) {
      for (auto& in : g_nodes.at(name).def->input()) {
        InputRef ref = ResolveRef(in);
        if (ref.control) continue;
        int& m = max_slot[ref.node];
        if (ref.slot > m) m = ref.slot;
      }
    }
    for (auto& [node, m] : max_slot) {
      auto it = g_nodes.find(node);
      if (it != g_nodes.end() && (int)it->second.out_dtypes.size() < m + 1) {
        if (g_opspecs.count(it->second.def->op())) ++g_warn;
        it->second.out_dtypes.resize(m + 1, tensorflow::DT_INVALID);
      }
    }
  }
  return 0;
}

// MLIRContext setup; each step prevents a crash: single-thread (workers
// segfault across the .so), unregistered dialects allowed, and a dummy
// import so the OFFICIAL importer registers the TF dialects (hand-built
// registries would split against the .so).
static int SetupMlirContext(mlir::MLIRContext& context) {
  context.disableMultithreading();
  context.allowUnregisteredDialects();
  {
    tensorflow::GraphdefToMlirOptions opts;
    opts.prune_unused_nodes = true;
    opts.graph_as_function = false;
    opts.enable_shape_inference = false;
    std::string dummy = MakeRegistrationGraphDef();
    auto reg_mod = tensorflow::GraphdefToMlirTranslateFunction(
        llvm::StringRef(dummy), /*input_arrays=*/"x",
        /*input_dtypes=*/"DT_FLOAT", /*input_shapes=*/"",
        /*output_arrays=*/"y", /*control_output_arrays=*/"", opts, &context);
    if (!reg_mod.ok()) {
      std::cerr << "[FATAL] dialect registration import failed: "
                << reg_mod.status().ToString() << "\n";
      return 3;
    }
  }
  {
    mlir::DialectRegistry registry;
    xla::RegisterAllHloDialects(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }
  return 0;
}

// Pipeline: args -> graph -> context -> infer -> official import route
// (pb rewrite -> importer -> flatten -> restore -> pipelines).
// Exit codes: 2 usage, 1 IO/pipeline, 3 dialect, 4 types/invariants,
// 5 internal stage, 6 graph structure (from the route internals).
int main(int argc, char** argv) {
  Options opt;
  if (ParseArgs(argc, argv, opt) != 0) return 2;
  GraphDef graph;
  if (LoadGraph(opt, graph) != 0) return 1;
  mlir::MLIRContext context;
  if (SetupMlirContext(context) != 0) return 3;

  mlir::OpBuilder b(&context);

  BuildCtx ctx;
  InferEntryAndDtypes(opt, ctx);  // --input specs + dtype derivation
  int rc = RunOfficialRoute(opt, ctx, graph, b, context);
  if (g_warn > 0)
    std::cerr << "[warn] total warnings = " << g_warn
              << " (details above; count also visible via stderr grep)\n";
  return rc;
}
