#include <algorithm>
#include <utility>
#include <vector>

#include "common/pb2_common.h"
#include "passes/pb2_mlir_passes.h"

// Custom-op (unregistered) attr dual channel: MLIR-convertible attrs direct,
// then custom_op + sorted custom_attrs JSON (round-trip channel for attrs the
// MLIR conversion cannot represent).
void AppendCustomOpAttrs(mlir::OperationState& st, const NodeDef& nd,
                         mlir::OpBuilder& b, const std::string& name) {
  for (auto& [aname, av] : nd.attr()) {
    if (!aname.empty() && aname[0] == '_') continue;
    if (IsReservedAttrName(aname)) continue;
    if (mlir::Attribute a = AttrToMlir(av, b)) {
      st.addAttribute(aname, a);
    } else {
      ++g_warn;
      std::cerr << "[warn] " << name << " attr '" << aname
                << "' not representable as MLIR attr, skipped (carried by JSON "
                   "fallback)\n";
    }
  }
  st.addAttribute("custom_op", b.getStringAttr(nd.op()));
  {
    std::ostringstream j;
    j << "{";
    bool first = true;
    std::vector<std::pair<std::string, const AttrValue*>> sorted;
    for (auto& [aname, av] : nd.attr())
      if (!(aname.empty() || aname[0] == '_' || IsReservedAttrName(aname)))
        sorted.emplace_back(aname, &av);
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& c) { return a.first < c.first; });
    for (auto& [aname, av] : sorted) {
      if (!first) j << ",";
      first = false;
      j << '"' << JsonEscape(aname) << "\":" << AttrValueToJson(*av);
    }
    // JSON side of the dtype-assumed marker (i8 alone would be ambiguous).
    if (auto ait = g_assumed_dtypes.find(name); ait != g_assumed_dtypes.end()) {
      if (!first) j << ",";
      j << "\"_dtype_assumed\":[";
      for (size_t k = 0; k < ait->second.size(); ++k)
        j << (k ? "," : "") << ait->second[k];
      j << "]";
    }
    j << "}";
    st.addAttribute("custom_attrs", b.getStringAttr(j.str()));
    if (auto ait = g_assumed_dtypes.find(name); ait != g_assumed_dtypes.end()) {
      llvm::SmallVector<int64_t> slots(ait->second.begin(), ait->second.end());
      st.addAttribute("dtype_assumed", b.getI64ArrayAttr(slots));
    }
  }
}

// What becomes entry args instead of staying in the artifact: Consts whose
// value ConvertTensorProto rejects (oversize tables). DT_STRING consts stay
// in the graph on purpose (the importer carries them, the sink emits them as
// atir.constant_v2; argifying would add N string feeds for no gain).
void PlanWeights(const Options& opt, mlir::Builder& b, BuildCtx& ctx) {
  for (auto& name : g_order) {
    NodeInfo& ni = g_nodes.at(name);
    if (ni.def->op() != "Const") continue;
    const AttrValue* ait = FindAttr(*ni.def, "value");
    if (ait != nullptr && ait->value_case() == AttrValue::kTensor &&
        tensorflow::ConvertTensorProto(ait->tensor(), &b, false).ok())
      continue;  // representable: official importer carries it as tf.Const
    ctx.argified.insert(name);
  }
  std::cerr << "[argify] " << ctx.argified.size() << " consts -> entry args\n";

  for (auto& name : g_order) {
    const std::string& op = g_nodes.at(name).def->op();
    if (op == "VariableV2" || op == "VarHandleOp" || op == "Variable") {
      ctx.var_arg.insert(name);
      ctx.var_skip.insert(name);
    }
  }
  bool grew = true;
  while (grew) {
    grew = false;
    for (auto& name : g_order) {
      if (ctx.var_skip.count(name)) continue;
      const NodeDef& nd = *g_nodes.at(name).def;
      if (nd.op() != "Identity" && nd.op() != "ReadVariableOp") continue;
      if (nd.input().empty()) continue;
      InputRef ref = ResolveRef(nd.input(0));
      if (ref.control) continue;  // control edges are not read chains
      if (ctx.var_skip.count(ref.node)) {
        ctx.var_skip.insert(name);
        grew = true;
      }
    }
  }
  std::cerr << "[parametrize] vars=" << ctx.var_arg.size()
            << " skips=" << ctx.var_skip.size() << "\n";
}
