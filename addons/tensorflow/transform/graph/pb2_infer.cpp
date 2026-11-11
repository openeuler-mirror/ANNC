#include "common/pb2_common.h"

// Data source of the dtype_assumed meta attr.
std::map<std::string, std::vector<int>> g_assumed_dtypes;
// Synthetic type-attr keys for exact-key erasure (see pb2_common.h).
std::map<std::string, std::set<std::string>> g_synth_attr_keys;

namespace {

// "1,2,-1" -> dims. Empty or illegal segments are fatal (silent atoll("")==0
// would fake a 0-dim), and so are out-of-range values (strtoll would silently
// clamp to LLONG_MAX). '-' is only legal as a leading sign.
std::vector<int64_t> ParseDims(const std::string& ds, const std::string& name) {
  std::vector<int64_t> dims;
  size_t st = 0;
  while (st <= ds.size()) {
    auto c = ds.find(',', st);
    std::string seg =
        (c == std::string::npos) ? ds.substr(st) : ds.substr(st, c - st);
    if (seg.empty() ||
        seg.find_first_not_of("0123456789-") != std::string::npos ||
        seg.find('-', 1) != std::string::npos) {
      std::cerr << "[FATAL] --input " << name << ": bad dims segment '"
                << seg << "' in \"" << ds << "\"\n";
      exit(kExitGraphStructure);
    }
    errno = 0;
    char* end = nullptr;
    long long v = std::strtoll(seg.c_str(), &end, 10);
    if (errno == ERANGE || end != seg.c_str() + seg.size()) {
      std::cerr << "[FATAL] --input " << name
                << ": dims segment '" << seg << "' out of range\n";
      exit(kExitGraphStructure);
    }
    dims.push_back(v);
    if (c == std::string::npos) break;
    st = c + 1;
  }
  return dims;
}

// Parses one --input spec "name[:DT[:dims]]" (explicit dims win over the
// node shape attr; name-only defers the dtype to graph inference).
EntrySpec ParseInputSpec(const std::string& s) {
  auto p1 = s.find(':');
  std::string name = (p1 == std::string::npos) ? s : s.substr(0, p1);
  auto it = g_nodes.find(name);
  if (it == g_nodes.end()) {
    std::cerr << "[FATAL] --input " << name << " not found in graph\n";
    exit(kExitGraphStructure);
  }
  EntrySpec e;
  if (p1 == std::string::npos) {
    e.from_graph = true;  // dtype inferred from the graph node
  } else {
    std::string rest = s.substr(p1 + 1);
    auto p2 = rest.find(':');
    std::string ty = (p2 == std::string::npos) ? rest : rest.substr(0, p2);
    if (!DataType_Parse(ty, &e.dt)) {
      std::cerr << "[FATAL] --input " << name
                << ": unparseable dtype " << ty << "\n";
      exit(kExitGraphStructure);
    }
    if (p2 != std::string::npos && !rest.substr(p2 + 1).empty()) {
      e.dims = ParseDims(rest.substr(p2 + 1), name);
      e.has_shape = true;
    }
  }
  if (!e.has_shape) {
    if (const AttrValue* sav = FindAttr(*it->second.def, "shape");
        sav != nullptr && !sav->shape().unknown_rank()) {
      for (const auto& d : sav->shape().dim()) e.dims.push_back(d.size());
      e.has_shape = true;
    }
  }
  return e;
}

// First dtype pass: Const dtype attr, else OpDef-based InferOutDtype per slot.
void InferInitialDtypes() {
  for (auto& name : g_order) {
    NodeInfo& ni = g_nodes.at(name);
    const NodeDef& nd = *ni.def;
    if (nd.op() != "Const") {
      for (int s = 0; s < (int)ni.out_dtypes.size(); ++s)
        ni.out_dtypes[s] = InferOutDtype(nd, s);
      continue;
    }
    DataType dt = tensorflow::DT_INVALID;
    if (const AttrValue* dav = FindAttr(nd, "dtype");
        dav != nullptr && dav->value_case() == AttrValue::kType)
      dt = dav->type();  // missing dtype attr -> INVALID; final check fails
    for (auto& d : ni.out_dtypes) d = dt;
  }
}

// Input-arg dtype: fixed type > the attr->spec-default fallback chain.
DataType InputArgDtype(const NodeDef& nd, const OpDef& spec,
                       const OpDef::ArgDef& iarg) {
  if (iarg.type_attr().empty()) return iarg.type();
  return AttrTypeOrSpecDefault(nd, spec, iarg.type_attr())
      .value_or(tensorflow::DT_INVALID);
}

// Backfills producer slots from consumer input-arg dtypes to a fixpoint
// (monotone INVALID -> valid; file order is not always topological, so one
// pass can miss chains). Only registered ops backfill.
void BackfillFromConsumers() {
  for (bool changed = true; changed;) {
    changed = false;
    for (auto& name : g_order) {
      NodeInfo& ni = g_nodes.at(name);
      auto spec_it = g_opspecs.find(ni.def->op());
      if (spec_it == g_opspecs.end()) continue;
      const OpDef& spec = spec_it->second;
      std::vector<int> arg_map = MapInputsToArgs(*ni.def, spec);
      int in_k = 0;
      for (auto& in : ni.def->input()) {
        InputRef ref = ResolveRef(in);
        if (ref.control) continue;
        if (in_k >= (int)arg_map.size()) break;
        DataType dt = InputArgDtype(*ni.def, spec,
                                    spec.input_arg(arg_map[in_k]));
        ++in_k;
        if (dt == tensorflow::DT_INVALID) continue;
        auto pit = g_nodes.find(ref.node);
        if (pit == g_nodes.end() ||
            ref.slot >= (int)pit->second.out_dtypes.size())
          continue;
        DataType& cur = pit->second.out_dtypes[ref.slot];
        if (cur == tensorflow::DT_INVALID) {
          cur = dt;
          changed = true;
        }
      }
    }
  }
}

// Resolves from_graph entry dtypes from the inferred graph, then overrides
// every entry slot (explicit --input dtypes are authoritative).
void ResolveEntryDtypes(BuildCtx& ctx) {
  for (auto& [name, e] : ctx.entries) {
    auto it = g_nodes.find(name);
    if (it == g_nodes.end()) continue;
    auto& dtypes = it->second.out_dtypes;
    if (e.from_graph) {
      if (dtypes.empty() || dtypes[0] == tensorflow::DT_INVALID) {
        std::cerr << "[FATAL] entry " << name
                  << " dtype undeterminable (no authoritative source) -- "
                     "specify explicitly --input " << name << ":DT_XXX\n";
        exit(kExitGraphStructure);
      }
      e.dt = dtypes[0];
    }
    std::fill(dtypes.begin(), dtypes.end(), e.dt);
  }
}

// Final gate: remaining DT_INVALID slots are fatal, or marked opaque DT_INT8
// with an explicit dtype_assumed record under --assume-opaque-i8.
void EnforceOrAssumeDtypes(const Options& opt, const BuildCtx& ctx) {
  int bad = 0;
  for (auto& name : g_order) {
    if (ctx.input_set.count(name) || ctx.var_skip.count(name) ||
        ctx.argified.count(name))
      continue;
    auto& dtypes = g_nodes.at(name).out_dtypes;
    for (size_t s = 0; s < dtypes.size(); ++s) {
      if (dtypes[s] != tensorflow::DT_INVALID) continue;
      ++bad;
      if (opt.assume_opaque_i8) {
        // i8 is only the type-carrier placeholder; disambiguation lives in
        // g_assumed_dtypes -> dtype_assumed (see pb2_common.h).
        dtypes[s] = tensorflow::DT_INT8;
        g_assumed_dtypes[name].push_back((int)s);
        std::cerr << "[assume-opaque-i8] " << name << " slot " << s
                  << " (no static dtype source; marked opaque DT_INT8)\n";
      } else if (bad <= 10) {
        std::cerr << "[FATAL] " << name << " slot " << s
                  << " dtype undeterminable (no attr/OpDef/downstream "
                     "constraint)\n";
      }
    }
  }
  if (bad > 0 && !opt.assume_opaque_i8) {
    std::cerr << "[FATAL] " << bad
              << " output slot(s) undeterminable -- refusing to emit "
                 "default-typed artifacts"
              << " (fixes: add node attr / explicit --input / fix graph; "
                 "or pass --assume-opaque-i8 to mark them as opaque "
                 "carriers)\n";
    exit(kExitGraphStructure);
  }
  if (bad > 0)
    std::cerr << "[assume-opaque-i8] " << bad
              << " output slot(s) marked DT_INT8 (--assume-opaque-i8)\n";
}

}  // namespace

// Parses --input specs, derives dtypes graph-wide, backfills from downstream
// type attrs, overrides entries, then hard-fails on remaining DT_INVALID
// (unless --assume-opaque-i8).
void InferEntryAndDtypes(const Options& opt, BuildCtx& ctx) {
  for (auto& s : opt.inputs) {
    std::string name = s.substr(0, s.find(':'));
    EntrySpec e = ParseInputSpec(s);
    ctx.entries[name] = e;
    ctx.input_set.insert(name);
    if (!e.from_graph)
      std::fill(g_nodes.at(name).out_dtypes.begin(),
                g_nodes.at(name).out_dtypes.end(), e.dt);
  }
  InferInitialDtypes();
  BackfillFromConsumers();
  ResolveEntryDtypes(ctx);
  EnforceOrAssumeDtypes(opt, ctx);
}

// Falls back to a synthetic OpDef when the global registry misses the op (lets
// Graph construction pass; ShapeRefiner gives unknown shapes for ops without
// a shape fn, which is non-fatal). Static cache is safe under the R8
// single-thread discipline.
absl::Status SynthOpRegistry::LookUp(
    const std::string& op,
    const tensorflow::OpRegistrationData** out) const {
  absl::Status st = tensorflow::OpRegistry::Global()->LookUp(op, out);
  if (st.ok()) return st;
  auto it = synth_.find(op);
  if (it == synth_.end())
    return absl::NotFoundError("no synth opdef: " + op);
  static std::map<std::string, tensorflow::OpRegistrationData> cache;
  auto& d = cache[op];
  d.op_def = it->second;
  *out = &d;
  return absl::OkStatus();
}

namespace {

// The rewrite domain's single "registered op" notion: the wheel registry
// loaded into g_opspecs (T1 decision: wheel registry is the only OpDef
// source). The TF Graph builder inside FunctionalizeGraphDefControlFlow
// keeps its own OpRegistry::Global lookups -- that is TF's own domain.
bool IsRegisteredOp(const std::string& op) {
  return g_opspecs.count(op) != 0;
}

// Max output-slot count per unregistered op, taken across all uses (dtype
// slots come from the node index, extra slots from "producer:N" references).
// Inputs need no arity: the synthetic arg is variadic (type_list_attr), so
// each node states its own input count via the injected list(type) attr --
// a per-slot scheme keyed on a cross-node max would misalign on nodes with
// fewer inputs than the max.
struct SynthArities {
  std::map<std::string, int> max_out;
};

SynthArities CollectSynthArities(const GraphDef& src) {
  SynthArities ar;
  std::map<std::string, const NodeDef*> by_name;
  for (auto& nd : src.node()) by_name[nd.name()] = &nd;
  for (auto& nd : src.node()) {
    if (IsRegisteredOp(nd.op())) continue;
    auto pit = g_nodes.find(nd.name());
    int dtype_slots =
        pit != g_nodes.end() ? (int)pit->second.out_dtypes.size() : 1;
    ar.max_out[nd.op()] =
        std::max(ar.max_out[nd.op()], std::max(1, dtype_slots));
  }
  for (auto& nd : src.node()) {
    for (auto& in : nd.input()) {
      if (!in.empty() && in[0] == '^') continue;
      auto ref = TryResolveRef(in);
      if (!ref.has_value() || ref->control) continue;
      auto pit = by_name.find(ref->node);
      if (pit == by_name.end() || IsRegisteredOp(pit->second->op())) continue;
      ar.max_out[pit->second->op()] =
          std::max(ar.max_out[pit->second->op()], ref->slot + 1);
    }
  }
  return ar;
}

// Variadic wildcard input ("tl" as a list(type)) and one wildcard-typed
// output arg ("o<i>") per slot.
void AppendSynthInput(OpDef* d) {
  auto* arg = d->add_input_arg();
  arg->set_name("in");
  arg->set_type_list_attr("tl");
  auto* attr = d->add_attr();
  attr->set_name("tl");
  attr->set_type("list(type)");
}

void AppendSynthOutput(OpDef* d, int index) {
  std::string key = "o" + std::to_string(index);
  auto* arg = d->add_output_arg();
  arg->set_name("out_" + std::to_string(index));
  arg->set_type_attr(key);
  auto* attr = d->add_attr();
  attr->set_name(key);
  attr->set_type("type");
}

// Injects per-slot type attrs into the graph copy (dtype = producer
// inference result; DT_INT8 as the unambiguous opaque sentinel when
// missing) and records the injected keys for exact-key erasure later.
void InjectSynthTypeAttrs(GraphDef* fg, const SynthArities& ar) {
  auto slot_dtype = [](const std::string& node, int slot) -> DataType {
    auto it = g_nodes.find(node);
    if (it == g_nodes.end() || slot >= (int)it->second.out_dtypes.size() ||
        it->second.out_dtypes[slot] == tensorflow::DT_INVALID)
      return tensorflow::DT_INT8;
    return it->second.out_dtypes[slot];
  };
  for (auto& nd : *fg->mutable_node()) {
    int n_out = ar.max_out.count(nd.op()) ? ar.max_out.at(nd.op()) : 0;
    if (n_out == 0) continue;  // registered op: untouched
    std::set<std::string>& injected = g_synth_attr_keys[nd.op()];
    // Input side: one list(type) attr carrying exactly this node's data
    // inputs (both the collector and this loop gate on TryResolveRef, so
    // the data-input notion cannot diverge between the two passes).
    auto* tl = (*nd.mutable_attr())["tl"].mutable_list();
    for (auto& in : nd.input()) {
      auto ref = TryResolveRef(in);
      if (!ref.has_value() || ref->control) continue;
      tl->add_type(slot_dtype(ref->node, ref->slot));
    }
    injected.insert("tl");
    for (int s = 0; s < n_out; ++s) {
      std::string key = "o" + std::to_string(s);
      (*nd.mutable_attr())[key].set_type(slot_dtype(nd.name(), s));
      injected.insert(key);
    }
  }
}

}  // namespace

// Builds synthetic OpDefs for unregistered custom ops and (optionally)
// injects per-slot type attrs into a copy of the graph, so the official
// Graph construction passes. Registered ops are untouched.
void BuildUnregisteredOpDefs(
    const GraphDef& src, std::map<std::string, OpDef>& synth, GraphDef* fg) {
  synth.clear();
  SynthArities ar = CollectSynthArities(src);
  for (auto& [op, max_out] : ar.max_out) {
    OpDef& d = synth[op];
    d.set_name(op);
    AppendSynthInput(&d);
    for (int i = 0; i < max_out; ++i) AppendSynthOutput(&d, i);
  }
  if (fg == nullptr) return;
  *fg = src;
  InjectSynthTypeAttrs(fg, ar);
}
