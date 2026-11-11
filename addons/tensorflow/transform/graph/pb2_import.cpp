#include "common/pb2_common.h"

#include <fstream>
#include <memory>

#include "graph/pb2_graph_pass.h"
#include "passes/pb2_mlir_passes.h"

#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/graph.h"

namespace {

// Stand-in Placeholder shape source: _output_shapes[slot] of the custom op.
void CopyShapeAttr(const NodeDef& nd, int slot, int n_slots, NodeDef* ph) {
  const AttrValue* sav = FindAttr(nd, "_output_shapes");
  if (sav == nullptr || sav->value_case() != AttrValue::kList) return;
  if (sav->list().shape_size() != n_slots) return;  // slot mismatch: drop all
  if (sav->list().shape(slot).unknown_rank()) return;
  (*ph->mutable_attr())["shape"].mutable_shape()->CopyFrom(
      sav->list().shape(slot));
}

// True if the node is a Switch whose data side traces back to a feed (fed
// values are always live and take the :1 branch, so the imported graph has no
// such chain; consumers connect to the feed arg directly).
bool FedSwitchRoot(const BuildCtx& ctx, const std::string& node,
                   std::string* root, int depth = 0) {
  if (depth > 64) return false;  // cycle guard: refuse, never overflow
  auto it = g_nodes.find(node);
  if (it == g_nodes.end() || it->second.def->op() != "Switch") return false;
  if (it->second.def->input().empty()) return false;
  InputRef r = ResolveRef(it->second.def->input(0));
  if (ctx.input_set.count(r.node)) {
    *root = r.node;
    return true;
  }
  return FedSwitchRoot(ctx, r.node, root, depth + 1);
}

// Three-step data-ref rewrite (shared by standard ops and functionalization
// carriers): fed-switch elimination -> read-chain rerooting -> custom-op slot
// stand-in names. Nullopt = unresolvable read chain (caller reports fatal).
std::optional<std::string> RewriteDataRef(
    const BuildCtx& ctx, const std::map<std::string, std::string>& var_root,
    const std::map<std::string, std::string>& root_read,
    const std::set<std::string>& custom_names, const std::string& in) {
  InputRef r = ResolveRef(in);
  std::string nn = r.node;
  long slot = r.slot;
  std::string root;
  if (slot == 1 && FedSwitchRoot(ctx, nn, &root)) {
    nn = root;
    slot = 0;
  }
  if (ctx.var_skip.count(nn) && !ctx.var_arg.count(nn)) {
    auto it = var_root.find(nn);
    if (it == var_root.end()) return std::nullopt;
    auto rit = root_read.find(it->second);
    // Consumer edges point at tensor true names (part_0/read).
    nn = rit != root_read.end() ? rit->second : it->second;
    slot = 0;
  }
  if (custom_names.count(nn))
    return slot == 0 ? nn : nn + "__" + std::to_string(slot);
  return slot == 0 ? nn : nn + ":" + std::to_string(slot);
}

// Stage 0: argified consts and variable roots must live in the main-graph
// domain (functionalization lifts function-domain ones into captures).
int VerifyMainDomain(const BuildCtx& ctx,
                     const std::set<std::string>& main_set) {
  for (auto& name : ctx.argified)
    if (!main_set.count(name)) {
      std::cerr << "[FATAL] argified const " << name
                << " landed in a branch function (unsupported)\n";
      return 4;
    }
  for (auto& name : ctx.var_arg)
    if (!main_set.count(name)) {
      std::cerr << "[FATAL] variable " << name
                << " root landed in a branch function (unsupported)\n";
      return 4;
    }
  return 0;
}

// Stage 1a: read-chain nodes -> variable roots (var_skip non-roots).
int ResolveVarRoots(const std::vector<std::string>& main_names,
                    const BuildCtx& ctx,
                    std::map<std::string, std::string>* var_root,
                    std::map<std::string, std::string>* root_read) {
  for (auto& name : main_names) {
    if (!ctx.var_skip.count(name) || ctx.var_arg.count(name)) continue;
    std::string cur = name;
    for (int hop = 0; hop < 64; ++hop) {
      const NodeDef& nd = *g_nodes.at(cur).def;
      if (nd.input().empty()) break;
      InputRef r = ResolveRef(nd.input(0));
      if (ctx.var_arg.count(r.node)) {
        (*var_root)[name] = r.node;
        // Tensor-true-name anchoring: root -> read node (feeds key on
        // tensor names, not resource handles).
        if (!root_read->count(r.node)) (*root_read)[r.node] = cur;
        break;
      }
      if (ctx.var_skip.count(r.node)) {
        cur = r.node;
        continue;
      }
      break;  // chain tail is not a variable: fatal below
    }
    if (!var_root->count(name)) {
      std::cerr << "[FATAL] var_skip node " << name
                << " does not resolve to a variable root\n";
      return 4;
    }
  }
  return 0;
}

// Stage 1b: drop parametrization for variables consumed directly (outside the
// read chain, incl. tf.If captures which use the resource handle type): the
// root placeholder can only express the content type, not the handle.
void TrimParametrizedVars(const std::map<std::string, const NodeDef*>& fmain,
                          const std::vector<std::string>& main_names,
                          const std::map<std::string, std::string>& var_root,
                          BuildCtx* ctx) {
  std::set<std::string> keep_whole;
  for (auto& name : main_names) {
    if (ctx->var_skip.count(name)) continue;
    auto git = fmain.find(name);
    if (git == fmain.end()) continue;
    for (auto& in : git->second->input()) {
      InputRef r = ResolveRef(in);
      if (!r.control && ctx->var_arg.count(r.node)) keep_whole.insert(r.node);
    }
  }
  for (auto& root : keep_whole) {
    ctx->var_arg.erase(root);
    ctx->var_skip.erase(root);
    for (auto& [n, r] : var_root)
      if (r == root) ctx->var_skip.erase(n);  // same-root chain restored too
    std::cerr << "[import-route] var " << root
              << " kept un-parametrized (direct handle consumers)\n";
  }
}

// Stage 3: in-place stand-in replacement for custom ops inside branch
// FunctionDefs (outer values enter as same-name captures).
int RewriteFunctionDomain(GraphDef* gp, const BuildCtx& ctx,
                          CustomOpTable* custom_ops,
                          int* n_func_custom_ops) {
  for (auto& fdef : *gp->mutable_library()->mutable_function()) {
    std::set<std::string> local_custom;
    for (auto& nd : fdef.node_def())
      if (g_opspecs.count(nd.op()) == 0) local_custom.insert(nd.name());
    if (local_custom.empty()) continue;
    // Erase injected synthetic type attrs by the exact key set recorded at
    // injection time (a t<N>/o<N> prefix pattern would hit real custom attrs).
    auto strip_synth_attrs = [](NodeDef* nd) {
      auto kit = g_synth_attr_keys.find(nd->op());
      if (kit == g_synth_attr_keys.end()) return;
      for (auto& k : kit->second) nd->mutable_attr()->erase(k);
    };
    // Function-body refs must be node:output:<slot>; functionalization's
    // cname:out_<N>[:X] form is normalized back here.
    auto rewrite_ref = [&local_custom](std::string in) -> std::string {
      for (const auto& cname : local_custom) {
        if (in == cname) return cname;  // bare slot0 name: normalized anyway
        if (in.rfind(cname + ":out_", 0) != 0) continue;
        std::string rest = in.substr(cname.size() + 5);  // skip ":out_"
        size_t c2 = rest.find(':');
        std::string slot = c2 == std::string::npos ? rest : rest.substr(0, c2);
        if (slot.empty() ||
            slot.find_first_not_of("0123456789") != std::string::npos)
          return in;
        // Slot N lives on the independent carrier cname__N (slot0 keeps the
        // original name); each carrier is single-output, ref is :output:0.
        return (slot == "0" ? cname : cname + "__" + slot) + ":output:0";
      }
      return in;
    };
    for (auto& nd : *fdef.mutable_node_def()) {
      for (int k = 0; k < nd.input_size(); ++k)
        nd.set_input(k, rewrite_ref(nd.input(k)));
    }
    for (auto& [rk, rv] : *fdef.mutable_ret()) rv = rewrite_ref(rv);
    // dtype-only Placeholder carriers: no shape attr = unranked, covering
    // unknown_rank custom ops (CopyShapeAttr keeps the unranked carrier).
    std::vector<NodeDef> extra_slots;
    auto anchor_inputs = [&rewrite_ref, &ctx](const NodeDef& nd) {
      std::vector<std::string> anchors;
      std::set<std::string> seen;
      for (auto& in : nd.input()) {
        if (in.empty() || in[0] == '^') continue;
        std::string prod = rewrite_ref(in);
        size_t c = prod.find(':');
        if (c != std::string::npos) prod = prod.substr(0, c);
        if (prod.rfind("arg", 0) == 0) continue;  // block-arg capture
        if (ctx.var_skip.count(prod) && !ctx.var_arg.count(prod)) continue;
        if (!seen.insert(prod).second) continue;
        anchors.push_back(prod);
      }
      return anchors;
    };
    for (auto& nd : *fdef.mutable_node_def()) {
      if (local_custom.count(nd.name()) == 0) continue;
      strip_synth_attrs(&nd);
      custom_ops->emplace(nd.name(), nd);  // archive for verbatim restore
      // fdef node names always come from the raw graph (GraphDef-level
      // functionalization does not rename; MLIR suffixes are loc-only);
      // find guards against operator[] silently inserting an empty NodeInfo.
      auto nit = g_nodes.find(nd.name());
      if (nit == g_nodes.end()) {
        std::cerr << "[FATAL] func-domain custom op '" << nd.name()
                  << "' not in the node index\n";
        return kExitGraphStructure;
      }
      NodeInfo& ni = nit->second;
      int n = (int)ni.out_dtypes.size();
      if (n < 1) {  // 中-6: empty out_dtypes would index [0] below
        std::cerr << "[FATAL] func-domain custom op '" << nd.name()
                  << "' has no dtype slots\n";
        return kExitGraphStructure;
      }
      auto anchors = anchor_inputs(custom_ops->at(nd.name()));
      for (int i = 1; i < n; ++i) {
        NodeDef ph;
        ph.set_name(nd.name() + "__" + std::to_string(i));
        ph.set_op("Placeholder");
        (*ph.mutable_attr())["dtype"].set_type(ni.out_dtypes[i]);
        CopyShapeAttr(custom_ops->at(nd.name()), i, n, &ph);
        for (auto& a : anchors) ph.add_input("^" + a);
        extra_slots.push_back(std::move(ph));
      }
      // slot0 stand-in: rewrite the original node in place
      std::string name = nd.name();
      DataType dt0 = ni.out_dtypes[0];
      nd.clear_input();
      nd.set_op("Placeholder");
      nd.mutable_attr()->clear();
      (*nd.mutable_attr())["dtype"].set_type(dt0);
      CopyShapeAttr(custom_ops->at(name), 0, n, &nd);
      for (auto& a : anchors) nd.add_input("^" + a);
      ++(*n_func_custom_ops);
    }
    for (auto& ph : extra_slots) *fdef.add_node_def() = std::move(ph);
    std::cerr << "[import-route] func " << fdef.signature().name()
              << " custom-op stand-ins = " << local_custom.size() << "\n";
  }
  return 0;
}

// Stage 4: drop dead Switches (:1 refs already point at feeds, the :0 branch
// is unreachable and has no consumers). The importer would keep them as
// executor SwitchOps and block pure-dataflow flattening. Iterate to a
// fixpoint as upstream Switches go dead in cascade.
void DropDeadSwitches(GraphDef* gp, const std::set<std::string>& main_set) {
  std::set<std::string> dead;
  for (bool grew = true; grew;) {
    grew = false;
    std::set<std::string> consumed;
    for (auto& nd : gp->node()) {
      // A node already dropped consumes nothing; without this the consumed
      // set is recomputed from an unchanged gp and the cascade stalls after
      // the first round.
      if (dead.count(nd.name())) continue;
      for (auto& in : nd.input()) {
        if (!in.empty() && in[0] == '^') continue;
        consumed.insert(ResolveRef(in).node);
      }
    }
    for (auto& name : g_order) {
      if (dead.count(name)) continue;
      auto it = g_nodes.find(name);
      if (it == g_nodes.end() || it->second.def->op() != "Switch") continue;
      if (!main_set.count(name)) continue;  // gone from the main graph
      if (consumed.count(name)) continue;
      dead.insert(name);
      grew = true;
    }
  }
  if (std::getenv("PB_VERBOSE") != nullptr)
    for (auto& nd : gp->node())
      for (auto& in : nd.input())
        if (in.find("Switch") != std::string::npos)
          std::cerr << "[dead-switch-debug] residual ref: " << nd.name()
                    << " <- " << in << "\n";
  if (dead.empty()) return;
  GraphDef kept;
  kept.mutable_versions()->CopyFrom(gp->versions());
  kept.mutable_library()->CopyFrom(gp->library());  // branch FunctionDefs
  for (auto& nd : gp->node()) {
    if (dead.count(nd.name())) continue;
    NodeDef cp = nd;
    cp.clear_input();
    for (auto& in : nd.input())
      if (!(in.size() > 1 && in[0] == '^' && dead.count(in.substr(1))))
        cp.add_input(in);
    *kept.add_node() = cp;
  }
  *gp = std::move(kept);
  std::cerr << "[import-route] dropped dead switch nodes = " << dead.size()
            << "\n";
}

// Stage 5: feed order = signature contract (entries in CLI order, then
// argified and variables in file order).
int AssembleFeeds(const Options& opt, const BuildCtx& ctx,
                  const std::map<std::string, FeedSpec>& feed_info,
                  const std::map<std::string, std::string>& root_read,
                  std::vector<FeedSpec>* feeds) {
  auto push = [&](const char* kind, const std::string& name) {
    auto it = feed_info.find(name);
    if (it == feed_info.end()) {
      // Variable placeholders key on tensor true names (see ResolveVarRoots).
      auto rr = root_read.find(name);
      if (rr != root_read.end()) it = feed_info.find(rr->second);
    }
    if (it == feed_info.end()) {
      std::cerr << "[FATAL] " << kind << " " << name
                << " was not rewritten into a feed placeholder\n";
      return 4;
    }
    feeds->push_back(it->second);
    return 0;
  };
  for (auto& s : opt.inputs)
    if (int rc = push("--input", s.substr(0, s.find(':')))) return rc;
  for (auto& name : g_order)
    if (ctx.argified.count(name))
      if (int rc = push("argified", name)) return rc;
  for (auto& name : g_order)
    if (ctx.var_arg.count(name))
      if (int rc = push("variable", name)) return rc;
  return 0;
}

// Stage view shared by the rewrite stages (the former locals of
// RewriteGraphForImport). Each stage reads/advances it; the sequence
// CollectRewriteView -> VerifyMainDomain -> ResolveVarRoots ->
// TrimParametrizedVars -> RewriteMainDomainStage -> RewriteFunctionDomain ->
// DropDeadSwitches -> AssembleFeeds reproduces the original monolith.
struct RewriteView {
  std::vector<std::string> main_names;
  std::set<std::string> main_set;
  std::map<std::string, const NodeDef*> fmain;   // post-functionalization defs
  std::map<std::string, std::string> var_root;   // read-chain node -> root
  std::map<std::string, std::string> root_read;  // var root -> read tensor
  std::set<std::string> custom_names;
  std::map<std::string, FeedSpec> feed_info;
};

void CollectRewriteView(const GraphDef& graph, RewriteView& view) {
  for (auto& nd : graph.node()) {
    view.main_names.push_back(nd.name());
    view.main_set.insert(nd.name());
    view.fmain[nd.name()] = &nd;
  }
}

}  // namespace

// Stage 2: per-node rewrite of the main-graph domain.
int RewriteMainDomainStage(const Options& opt, BuildCtx& ctx, RewriteView& view,
                           GraphDef& gp, CustomOpTable& custom_ops) {
  auto& main_names = view.main_names;
  auto& var_root = view.var_root;
  auto& root_read = view.root_read;
  auto& custom_names = view.custom_names;
  auto& feed_info = view.feed_info;

  // Entry feeds come from EntrySpec (--input overrides + shape fallback).
  for (auto& s : opt.inputs) {
    std::string name = s.substr(0, s.find(':'));
    const EntrySpec& e = ctx.entries.at(name);
    feed_info.emplace(name, FeedSpec{name, e.dt, e.dims, e.has_shape});
  }
  auto add_placeholder = [&](const std::string& name, DataType dt,
                             const std::vector<int64_t>& dims, bool has_shape) {
    NodeDef* ph = gp.add_node();
    ph->set_name(name);
    ph->set_op("Placeholder");
    (*ph->mutable_attr())["dtype"].set_type(dt);
    if (has_shape)
      for (int64_t d : dims)
        (*ph->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(d);
    feed_info.emplace(name, FeedSpec{name, dt, dims, has_shape});
  };

  for (auto& name : main_names) {
    NodeInfo* ni = nullptr;
    if (auto git = g_nodes.find(name); git != g_nodes.end()) ni = &git->second;
    const NodeDef& nd = *view.fmain.at(name);

    if (ni == nullptr) {
      // functionalization carriers (tf.If/tf.While): copy + ref rewrite
      NodeDef* cp = gp.add_node();
      *cp = nd;
      cp->clear_input();
      for (auto& in : nd.input()) {
        InputRef r = ResolveRef(in);
        if (r.control) {
          if (ctx.var_skip.count(r.node) && !ctx.var_arg.count(r.node))
            continue;
          cp->add_input(in);
          continue;
        }
        auto nn = RewriteDataRef(ctx, var_root, root_read, custom_names, in);
        if (!nn) {
          std::cerr << "[FATAL] " << name << " input " << in
                    << " does not resolve to a variable root\n";
          return 4;
        }
        cp->add_input(*nn);
      }
      continue;
    }

    if (ctx.var_arg.count(name)) {  // variable root -> tensor-name Placeholder
      DataType dt = tensorflow::DT_INVALID;
      if (const AttrValue* dav = FindAttr(nd, "dtype")) dt = dav->type();
      if (dt == tensorflow::DT_INVALID) {
        std::cerr << "[FATAL] variable " << name << " missing dtype attr\n";
        return 4;
      }
      std::vector<int64_t> dims;
      bool has = false;
      if (const AttrValue* sav = FindAttr(nd, "shape")) {
        if (!sav->shape().unknown_rank()) {
          for (const auto& d : sav->shape().dim()) dims.push_back(d.size());
          has = true;
        }
      }
      // Placeholders use tensor true names (part_0/read), not the resource
      // name, matching fusion.metadata anchoring.
      std::string tensor_name = name;
      auto rit = root_read.find(name);
      if (rit != root_read.end()) tensor_name = rit->second;
      add_placeholder(tensor_name, dt, dims, has);
      continue;
    }
    if (ctx.var_skip.count(name)) continue;  // read chain: dropped

    if (ctx.argified.count(name)) {  // unrepresentable const -> arg
      const AttrValue* vav = FindAttr(nd, "value");
      std::vector<int64_t> dims;
      bool has = false;
      if (vav != nullptr && vav->value_case() == AttrValue::kTensor &&
          !vav->tensor().tensor_shape().unknown_rank()) {
        for (const auto& d : vav->tensor().tensor_shape().dim())
          dims.push_back(d.size());
        has = true;
      }
      if (ni->out_dtypes.empty()) {
        std::cerr << "[FATAL] argified const " << name
                  << " has no dtype slot\n";
        return kExitGraphStructure;
      }
      add_placeholder(name, ni->out_dtypes[0], dims, has);
      continue;
    }

    if (g_opspecs.count(nd.op()) == 0) {  // custom op: one stand-in per slot
      if (ctx.input_set.count(name)) {
        std::cerr << "[FATAL] --input " << name
                  << " is an unregistered custom op; the official import "
                     "route needs a registered entry node\n";
        return 4;
      }
      if (ni->out_dtypes.empty()) {
        std::cerr << "[FATAL] custom op " << name
                  << " has no dtype slots\n";
        return kExitGraphStructure;
      }
      int n = (int)ni->out_dtypes.size();
      {  // archive with fed-switch elimination applied, so restore resolves
        NodeDef norm = nd;
        norm.clear_input();
        for (auto& in : nd.input()) {
          InputRef r = ResolveRef(in);
          std::string nn = r.node;
          long slot = r.slot;
          std::string root;
          if (!r.control && slot == 1 && FedSwitchRoot(ctx, nn, &root)) {
            nn = root;
            slot = 0;
          }
          norm.add_input(slot == 0 ? nn : nn + ":" + std::to_string(slot));
        }
        custom_ops.emplace(name, std::move(norm));
      }
      // Stand-ins anchor on data-input producers (dependency-free ones would
      // order early and break restore dominance); read-chain nodes are
      // deleted by the rewrite and must not anchor.
      std::vector<std::string> anchors;
      std::set<std::string> anchor_seen;
      for (auto& in : nd.input()) {
        InputRef r = ResolveRef(in);
        if (r.control) continue;
        if (ctx.var_skip.count(r.node) && !ctx.var_arg.count(r.node)) continue;
        if (!anchor_seen.insert(r.node).second) continue;
        anchors.push_back(r.node);
      }
      for (int i = 0; i < n; ++i) {
        std::string nm = i == 0 ? name : name + "__" + std::to_string(i);
        NodeDef* ph = gp.add_node();
        ph->set_name(nm);
        ph->set_op("Placeholder");
        (*ph->mutable_attr())["dtype"].set_type(ni->out_dtypes[i]);
        CopyShapeAttr(nd, i, n, ph);  // keep ranked result types
        for (auto& a : anchors) ph->add_input("^" + a);
      }
      continue;
    }

    // standard op: copy as-is + ref rewrite
    NodeDef* cp = gp.add_node();
    *cp = nd;
    cp->clear_input();
    for (auto& in : nd.input()) {
      InputRef r = ResolveRef(in);
      if (r.control) {
        // drop control edges into deleted read-chain nodes only; root
        // same-name control edges are kept as-is
        if (ctx.var_skip.count(r.node) && !ctx.var_arg.count(r.node)) continue;
        cp->add_input(in);
        continue;
      }
      auto nn = RewriteDataRef(ctx, var_root, root_read, custom_names, in);
      if (!nn) {
        std::cerr << "[FATAL] " << name << " input " << in
                  << " does not resolve to a variable root\n";
        return 4;
      }
      cp->add_input(*nn);
    }
  }
  return 0;
}

// pb-level complement rewrite: 0 domain check -> 1 read-chain resolve ->
// 2 main-graph rewrite -> 3 function-domain stand-ins -> 4 dead-Switch drop
// -> 5 feed assembly. graph = (optionally functionalized) main + branches.
int RewriteGraphForImport(const Options& opt, BuildCtx& ctx,
                          const GraphDef& graph, GraphDef& gp,
                          CustomOpTable& custom_ops,
                          std::vector<FeedSpec>& feeds) {
  RewriteView view;
  CollectRewriteView(graph, view);
  if (int rc = VerifyMainDomain(ctx, view.main_set)) return rc;
  if (int rc = ResolveVarRoots(view.main_names, ctx, &view.var_root,
                               &view.root_read))
    return rc;
  TrimParametrizedVars(view.fmain, view.main_names, view.var_root, &ctx);

  // Custom-op list collected before the main loop (consumers may precede
  // producers); main domain uses functionalized defs, g_nodes keeps raw.
  for (auto& name : view.main_names)
    if (g_opspecs.count(view.fmain.at(name)->op()) == 0)
      view.custom_names.insert(name);

  gp.mutable_versions()->CopyFrom(graph.versions());
  gp.mutable_library()->CopyFrom(graph.library());

  if (int rc = RewriteMainDomainStage(opt, ctx, view, gp, custom_ops))
    return rc;

  int n_func_custom_ops = 0;
  if (int rc = RewriteFunctionDomain(&gp, ctx, &custom_ops, &n_func_custom_ops))
    return rc;
  DropDeadSwitches(&gp, view.main_set);
  if (int rc = AssembleFeeds(opt, ctx, view.feed_info, view.root_read, &feeds))
    return rc;
  std::cerr << "[import-route] rewritten: custom_ops=" << custom_ops.size()
            << " argified=" << ctx.argified.size()
            << " vars=" << ctx.var_arg.size() << " feeds=" << feeds.size()
            << " func_custom_ops=" << n_func_custom_ops << "\n";
  return 0;
}

// Imports gp via the official translator.
int ImportOfficial(const Options& opt, const GraphDef& gp,
                   const std::vector<FeedSpec>& feeds,
                   mlir::MLIRContext& context,
                   mlir::OwningOpRef<mlir::ModuleOp>& module) {
  std::string in_names, in_dtypes, in_shapes;
  for (size_t i = 0; i < feeds.size(); ++i) {
    const FeedSpec& f = feeds[i];
    in_names += (i ? "," : "") + f.name;
    in_dtypes += (i ? "," : "") + tensorflow::DataType_Name(f.dt);
    // "*": the legal unknown-shape spelling; an empty entry makes the
    // spec parser drop the whole list and silently unrank every entry.
    std::string sh = "*";
    if (f.has_shape) {
      sh.clear();
      for (size_t d = 0; d < f.dims.size(); ++d)
        sh +=
            (d ? "," : "") + (f.dims[d] < 0 ? "?" : std::to_string(f.dims[d]));
    }
    in_shapes += (i ? ":" : "") + sh;
  }
  std::string out_names;
  for (auto& o : opt.outputs) out_names += (out_names.empty() ? "" : ",") + o;

  if (std::getenv("PB_DEBUG_IMPORT_INPUTS")) {
    int n_with = 0;
    size_t p0 = 0;
    for (size_t i = 0; i <= in_shapes.size(); ++i) {
      if (i == in_shapes.size() || in_shapes[i] == ':') {
        if (i > p0) n_with++;
        p0 = i + 1;
      }
    }
    std::cerr << "[import-inputs] feeds=" << feeds.size()
              << " with_shape=" << n_with << " in_names_len=" << in_names.size()
              << " sample_shapes=" << in_shapes.substr(0, 120) << "\n";
  }

  tensorflow::GraphdefToMlirOptions opts;
  // No pruning: stand-in chains are outside the fetch-reachable graph (dead
  // ops are cleaned by DCE); Grappler's _output_shapes wins over inference.
  opts.prune_unused_nodes = false;
  opts.unconditionally_use_set_output_shapes = true;
  opts.graph_as_function = false;
  opts.enable_shape_inference = true;

  std::string gp_str = gp.SerializeAsString();
  auto res = tensorflow::GraphdefToMlirTranslateFunction(
      llvm::StringRef(gp_str), in_names, in_dtypes, in_shapes, out_names, "",
      opts, &context);
  if (!res.ok()) {
    std::cerr << "[FATAL] official import failed: " << res.status().ToString()
              << "\n";
    return kExitInternal;
  }
  module = std::move(res.value());
  std::cerr << "[import-route] official import ok (feeds=" << feeds.size()
            << " outputs=" << opt.outputs.size() << ")\n";
  return 0;
}

// tf2xla-bridge v1 functionalization (Switch/Merge -> tf.If + branch
// FunctionDefs); runs on the raw graph, custom ops join via synth OpDefs.
int FunctionalizeGraphDefControlFlow(const GraphDef& src, GraphDef& out) {
  if (std::getenv("PB_NO_FUNC") != nullptr) {  // early exit: no deep copy
    out = src;
    return 0;
  }
  bool has_ctrl = false;
  for (auto& nd : src.node())
    if (nd.op() == "Switch") {
      has_ctrl = true;
      break;
    }
  if (!has_ctrl) {  // no deep copy needed
    out = src;
    return 0;
  }
  out.Clear();
  std::map<std::string, tensorflow::OpDef> synth;
  GraphDef fg;
  BuildUnregisteredOpDefs(src, synth, &fg);
  SynthOpRegistry reg(std::move(synth));
  tensorflow::Graph tgraph(&reg);
  tensorflow::GraphConstructorOptions copts;
  copts.allow_internal_ops = true;
  auto cst = tensorflow::ConvertGraphDefToGraph(copts, fg, &tgraph);
  if (!cst.ok()) {
    std::cerr << "[FATAL] graph convert for functionalize: " << cst.ToString()
              << "\n";
    return kExitInternal;
  }
  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             src.library());
  auto st = tensorflow::FunctionalizeControlFlow(&tgraph, &flib,
                                                 tensorflow::NodeFilter{},
                                                 /*include_functions=*/false);
  if (!st.ok()) {
    std::cerr << "[FATAL] v1 control-flow functionalization failed: "
              << st.ToString() << "\n";
    return kExitInternal;
  }
  tgraph.ToGraphDef(&out, /*include_flib_def=*/false);
  // Branch FunctionDefs live in flib: merge them into the output library.
  for (const std::string& fn : flib.ListFunctionNames()) {
    const tensorflow::FunctionDef* fdef = flib.Find(fn);
    if (fdef != nullptr) *out.mutable_library()->add_function() = *fdef;
  }
  std::cerr << "[import-route] graph functionalized: nodes " << src.node_size()
            << " -> " << out.node_size()
            << ", functions=" << out.library().function_size() << "\n";
  return 0;
}

// True if the rewritten main graph still has Switches, i.e. a real dynamic
// cond cluster that the rewrite could not eliminate.
static bool HasSwitchNode(const GraphDef& gp) {
  for (auto& nd : gp.node())
    if (nd.op() == "Switch") return true;
  return false;
}

// Official-import route orchestration: GraphPassManager over the pb domain
// (named stages, short-circuit error codes) + MLIR PassManager over the
// MLIR domain.
int RunOfficialRoute(const Options& opt, BuildCtx& ctx, const GraphDef& graph,
                     mlir::OpBuilder& b, mlir::MLIRContext& context) {
  GraphDef gp;
  CustomOpTable custom_ops;
  std::vector<FeedSpec> feeds;

  GraphPassManager pm;
  pm.addPass(makeGraphPass("plan-weights", [&](const GraphDef&, BuildCtx& c) {
    PlanWeights(opt, b, c);
    return 0;
  }));
  pm.addPass(makeGraphPass("pb-rewrite", [&](const GraphDef&, BuildCtx& c) {
    return RewriteGraphForImport(opt, c, graph, gp, custom_ops, feeds);
  }));
  // Conditional functionalization (only if the rewrite left Switches).
  pm.addPass(makeGraphPass(
      "functionalize-if-switch", [&](const GraphDef&, BuildCtx& c) {
        if (!HasSwitchNode(gp)) return 0;
        GraphDef gfunc;
        if (int rc = FunctionalizeGraphDefControlFlow(graph, gfunc)) return rc;
        gp.Clear();
        custom_ops.clear();
        feeds.clear();
        return RewriteGraphForImport(opt, c, gfunc, gp, custom_ops, feeds);
      }));
  if (int rc = pm.run(graph, ctx)) return rc;

  // Best-effort shape enrichment (registered ops + Placeholders only;
  // Grappler facts win).
  EnrichGraphWithOfficialInference(gp, opt, ctx);

  if (const char* dp = std::getenv("PB_DUMP_IMPORT_GRAPH")) {
    std::ofstream out(dp, std::ios::binary);
    std::string ser = gp.SerializeAsString();
    out.write(ser.data(), ser.size());
    std::cerr << "[dump-import-graph] " << dp << " (" << gp.node_size()
              << " nodes)\n";
  }

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int rc = ImportOfficial(opt, gp, feeds, context, module)) return rc;
  if (std::getenv("PB_VERIFY_STAGES") != nullptr) {
    std::cerr << "[verify-stage] after import:\n";
    VerifyBuiltModule(context, *module);
  }

  {
    mlir::PassManager mpm(&context);
    mpm.enableVerifier(false);
    mpm.addPass(pb_passes::CreateNormalizeRefTypesPass());
    if (std::getenv("PB_NO_FLATTEN") == nullptr)
      mpm.addPass(pb_passes::CreateFlattenExecutorPass());
    if (failed(mpm.run(*module))) {
      std::cerr << "[FATAL] normalize/flatten pass pipeline failed\n";
      return kExitInternal;
    }
  }
  if (std::getenv("PB_VERIFY_STAGES") != nullptr) {
    std::cerr << "[verify-stage] after flatten:\n";
    VerifyBuiltModule(context, *module);
  }
  if (int rc = pb_passes::RunRestoreOpsPass(*module, opt, feeds, custom_ops, b,
                                            context))
    return rc;
  if (std::getenv("PB_VERIFY_STAGES") != nullptr)
    std::cerr << "[verify-stage] after restore:\n";
  VerifyBuiltModule(context, *module);
  {
    mlir::PassManager mpm(&context);
    mpm.enableVerifier(false);
    mpm.addPass(pb_passes::CreateStripFunctionalizeSuffixPass());
    mpm.addPass(pb_passes::CreateAnnotateShapesFromPbPass(graph));
    mpm.addPass(pb_passes::CreateConvergeSignaturesPass());
    mpm.addPass(pb_passes::CreateTagConstSourceNamePass());
    if (failed(mpm.run(*module))) {
      std::cerr << "[FATAL] strip pass pipeline failed\n";
      return kExitInternal;
    }
  }

  if (opt.dump_tfmlir) {
    module->print(llvm::outs(), mlir::OpPrintingFlags().enableDebugInfo());
    llvm::outs().flush();
    std::cerr << "[dump] tf.mlir printed\n";
    return 0;
  }

  mlir::ModuleOp m = *module;
  return RunPipelines(opt, ctx, graph, context, m);
}
