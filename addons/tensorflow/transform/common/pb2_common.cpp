#include "common/pb2_common.h"

// Exit codes (stage-localizable):
// 0 ok; 1 usage; 2 arg parsing; 3 OpDef/MLIR-context init failure
// (LoadOpSpecs and SetupMlirContext share 3 -- historical;
// disambiguate via logs); 4 GraphDef structure failure (bad input ref).

// Global state: written only during single-threaded init (LoadOpSpecs/
// LoadGraph); read-only afterwards. Never mutate these tables from parallel walks.
std::map<std::string, NodeInfo> g_nodes;
std::vector<std::string> g_order;
std::map<std::string, OpDef> g_opspecs;
int g_warn = 0;

struct TF_Buffer {
  const void* data;
  size_t length;
  void (*data_deallocator)(void* data, size_t length);
};
extern "C" TF_Buffer* TF_GetAllOpList();

// Loads OpDefs from the in-process (wheel) TF registry.
void LoadOpSpecs() {
  TF_Buffer* buf = TF_GetAllOpList();
  if (buf == nullptr || buf->data == nullptr) {
    std::cerr << "[FATAL] TF_GetAllOpList returned empty (libtensorflow_cc too old?)\n";
    exit(kExitInit);
  }
  tensorflow::OpList ops;
  bool ok = ops.ParseFromArray(buf->data, (int)buf->length);
  if (buf->data_deallocator != nullptr)
    buf->data_deallocator(const_cast<void*>(buf->data), buf->length);
  (void)buf;
  if (!ok) {
    std::cerr << "[FATAL] OpList proto parse failed\n";
    exit(kExitInit);
  }
  for (const OpDef& def : ops.op()) g_opspecs[def.name()] = def;
  std::cerr << "[opspec] loaded " << g_opspecs.size()
            << " op defs (in-process wheel TF registry)\n";
}

// strtol with overflow clamp (atoi on overflowing digit strings is UB).
int ParseSlot(const char* s) {
  errno = 0;
  long v = std::strtol(s, nullptr, 10);
  if (errno == ERANGE)
    return v > 0 ? std::numeric_limits<int>::max() : 0;
  if (v < 0) return 0;
  return (int)std::min<long>(v, std::numeric_limits<int>::max());
}

// Only the three standard GraphDef encodings are accepted; anything else is
// fatal (e.g. FunctionDef-style "a:b:0" would parse into a garbage InputRef
// and silently corrupt structure downstream); TryResolveRef is the
// non-fatal scan variant declared in pb2_common.h.
InputRef ResolveRef(const std::string& s) {
  auto r = TryResolveRef(s);
  if (!r.has_value()) {
    std::cerr << "[FATAL] invalid input encoding " << s
              << ": expected node / node:slot / ^node\n";
    exit(kExitGraphStructure);
  }
  return *r;
}

std::optional<InputRef> TryResolveRef(const std::string& s) {
  InputRef r;
  if (!s.empty() && s[0] == '^') {  // "^node" control edge
    if (s.size() == 1 || s.find(':', 1) != std::string::npos)
      return std::nullopt;
    r.control = true;
    r.node = s.substr(1);
    return r;
  }
  size_t colon = s.find(':');
  if (colon == std::string::npos) {  // "node"
    if (s.empty()) return std::nullopt;
    r.node = s;
    return r;
  }
  std::string slot = s.substr(colon + 1);  // "node:digits"
  if (colon == 0 || slot.empty() ||
      slot.find_first_not_of("0123456789") != std::string::npos)
    return std::nullopt;
  r.node = s.substr(0, colon);
  r.slot = ParseSlot(slot.c_str());
  return r;
}

// Linear scan: protobuf Map::find is unreliable across the ODR boundary.
const AttrValue* FindAttr(const NodeDef& nd, const std::string& key) {
  for (const auto& [k, v] : nd.attr()) {
    if (k == key) return &v;
  }
  return nullptr;
}

int VerifyBuiltModule(mlir::MLIRContext& context, mlir::ModuleOp module) {
  int verify_fails = 0;
  mlir::ScopedDiagnosticHandler diag(&context, [&](mlir::Diagnostic& d) {
    std::string loc;
    llvm::raw_string_ostream os(loc);
    d.getLocation().print(os);
    if (verify_fails < 3)
      std::cerr << "[diag] " << loc << ": " << d.str() << "\n";
    ++verify_fails;
    return mlir::success();
  });
  if (failed(mlir::verify(module))) {
    std::cerr << "[build] verify failed, " << verify_fails
              << " diagnostics (mostly unregistered-op ODS mismatch; continuing)\n";
  } else {
    std::cerr << "[build] verify ok\n";
  }
  return verify_fails;
}

std::optional<DataType> AttrTypeOrSpecDefault(const NodeDef& nd,
                                              const OpDef& spec,
                                              const std::string& attr_name) {
  if (const AttrValue* av = FindAttr(nd, attr_name))
    if (av->value_case() == AttrValue::kType) return av->type();
  for (const auto& ad : spec.attr())
    if (ad.name() == attr_name && ad.has_default_value() &&
        ad.default_value().value_case() == AttrValue::kType)
      return ad.default_value().type();
  return std::nullopt;
}

std::optional<long> AttrIntOrSpecDefault(const NodeDef& nd, const OpDef& spec,
                                         const std::string& attr_name) {
  if (const AttrValue* av = FindAttr(nd, attr_name))
    if (av->value_case() == AttrValue::kI) return av->i();
  for (const auto& ad : spec.attr())
    if (ad.name() == attr_name && ad.has_default_value() &&
        ad.default_value().value_case() == AttrValue::kI)
      return ad.default_value().i();
  return std::nullopt;
}

// Maps data inputs to OpDef input-arg indices (inverse OpPortIdToArgId);
// control deps excluded, count mismatches warn.
std::vector<int> MapInputsToArgs(const NodeDef& nd, const OpDef& spec) {
  int num_data = 0;  // data inputs (control deps excluded)
  for (const auto& in : nd.input())
    if (in.empty() || in[0] != '^') ++num_data;

  std::vector<int> map;
  const int nargs = spec.input_arg_size();
  for (int argi = 0; argi < nargs; ++argi) {
    const auto& a = spec.input_arg(argi);
    long n = 1;
    if (!a.number_attr().empty()) {
      auto nattr = AttrIntOrSpecDefault(nd, spec, a.number_attr());
      if (!nattr.has_value()) {
        ++g_warn;
        std::cerr << "[warn] " << nd.name() << " (" << nd.op()
                  << ") missing number_attr for data input " << a.number_attr()
                  << " and no OpDef default -> counting 1\n";
      }
      n = nattr.value_or(1);
    } else if (!a.type_list_attr().empty()) {
      const AttrValue* av = FindAttr(nd, a.type_list_attr());
      bool got = av != nullptr && av->value_case() == AttrValue::kList;
      if (!got) {
        ++g_warn;
        std::cerr << "[warn] " << nd.name() << " (" << nd.op()
                  << ") missing type_list_attr for data input " << a.type_list_attr()
                  << " -> counting 1\n";
      }
      n = got ? av->list().type_size() : 1;
    }
    for (long k = 0; k < n; ++k) map.push_back(argi);
  }
  if (nargs == 0) return map;  // no args: skip padding to avoid -1 indexing
  if ((int)map.size() < num_data) {
    ++g_warn;
    std::cerr << "[warn] " << nd.name() << " (" << nd.op() << ") data inputs "
              << num_data << " > OpDef expansion " << map.size()
              << " -> extras bound to the last arg\n";
    while ((int)map.size() < num_data) map.push_back(nargs - 1);
  } else if ((int)map.size() > num_data) {
    ++g_warn;
    std::cerr << "[warn] " << nd.name() << " (" << nd.op() << ") OpDef expansion "
              << map.size() << " > data inputs " << num_data << " -> truncating to data inputs\n";
  }
  return map;
}

// Dtype inference priority, first hit wins: static type > explicit type_attr
// > shared-input propagation > OpDef default (only when a shared input exists);
// DT_INVALID = unknown (backfill or fail later).
DataType InferOutDtype(const NodeDef& nd, int slot) {
  auto spec_it = g_opspecs.find(nd.op());
  if (spec_it == g_opspecs.end()) return tensorflow::DT_INVALID;
  const OpDef& spec = spec_it->second;
  if (slot >= spec.output_arg_size()) return tensorflow::DT_INVALID;
  const auto& oarg = spec.output_arg(slot);

  if (oarg.type_attr().empty()) return oarg.type();
  const std::string& a = oarg.type_attr();

  if (const AttrValue* av = FindAttr(nd, a);
      av != nullptr && av->value_case() == AttrValue::kType)
    return av->type();

  bool foundShared = false;
  for (int k = 0; k < spec.input_arg_size(); ++k) {
    if (spec.input_arg(k).type_attr() != a) continue;
    foundShared = true;
    std::vector<int> arg_map = MapInputsToArgs(nd, spec);
    for (int inp = 0; inp < (int)arg_map.size(); ++inp) {
      if (arg_map[inp] != k) continue;
      InputRef ref = ResolveRef(nd.input(inp));
      auto it = g_nodes.find(ref.node);
      if (it != g_nodes.end() &&
          ref.slot < (int)it->second.out_dtypes.size())
        return it->second.out_dtypes[ref.slot];
      break;  // first input of this arg only (historical)
    }
    break;
  }

  if (foundShared) {
    for (const auto& ad : spec.attr())
      if (ad.name() == a && ad.has_default_value() &&
          ad.default_value().value_case() == AttrValue::kType)
        return ad.default_value().type();
  }

  return tensorflow::DT_INVALID;
}
