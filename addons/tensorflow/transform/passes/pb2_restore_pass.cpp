#include "common/pb2_common.h"

#include <functional>
#include <limits>
#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "passes/pb2_mlir_passes.h"

namespace pb_passes {
namespace {

struct BodyScope {
  mlir::Block* block = nullptr;
  std::string sym;  // func symbol name (branch capture mapping)
  std::map<std::string, mlir::Operation*> by_name;
  std::map<std::string, mlir::Value> arg_val;
};

// Slot-stand-in value for a producer custom op not restored yet: a later
// RAUW re-points the reference to the restored op. Null if unavailable.
mlir::Value SlotStandinValue(const BodyScope& sc, const std::string& node,
                             int slot) {
  if (slot <= 0) return nullptr;
  auto pi = sc.by_name.find(node + "__" + std::to_string(slot));
  if (pi == sc.by_name.end() || pi->second->getNumResults() != 1)
    return nullptr;
  return pi->second->getResult(0);
}

// Resolves one archived input ref to a value in the scope: capture/feed arg >
// in-block producer result > slot stand-in of a not-yet-restored producer >
// cname__N carrier fallback. Null on failure (near-miss hints printed).
mlir::Value ResolveOperand(const BodyScope& sc, const InputRef& r) {
  if (auto av = sc.arg_val.find(r.node); av != sc.arg_val.end())
    return av->second;  // feed or functionalization capture
  auto oi = sc.by_name.find(r.node);
  if (oi != sc.by_name.end() && r.slot < (int)oi->second->getNumResults())
    return oi->second->getResult(r.slot);
  if (oi != sc.by_name.end())  // producer not restored yet
    return SlotStandinValue(sc, r.node, r.slot);
  auto us = r.node.rfind("__");
  if (us != std::string::npos && us + 2 < r.node.size() &&
      r.node.find_first_not_of("0123456789", us + 2) == std::string::npos) {
    auto bi = sc.by_name.find(r.node.substr(0, us));
    int idx = ParseSlot(r.node.c_str() + us + 2);
    if (bi != sc.by_name.end() && idx < (int)bi->second->getNumResults())
      return bi->second->getResult(idx);
  }
  for (auto& [k, op] : sc.by_name)
    if (k.rfind(r.node, 0) == 0 || r.node.rfind(k, 0) == 0)
      std::cerr << "[restore-hint] near-miss '" << k
                << "' results=" << op->getNumResults() << "\n";
  return nullptr;
}

// Binds a branch scope's block args to capture names: the operand-aligned
// producer name plus the functionalizer's argN alias (archived NodeDefs use
// the fdef input-arg name -- dual key).
void BindBranchCaptures(
    mlir::Operation* op, int off, BodyScope* sc,
    std::map<std::pair<mlir::Block*, unsigned>, std::string>* arg_name,
    const std::function<std::string(mlir::Value)>& node_name_of) {
  for (mlir::BlockArgument ba : sc->block->getArguments()) {
    int j = (int)ba.getArgNumber();
    if (j + off >= (int)op->getNumOperands()) continue;
    std::string nm = node_name_of(op->getOperand(j + off));
    if (!nm.empty()) {
      sc->arg_val.emplace(nm, ba);
      arg_name->emplace(std::make_pair(sc->block, ba.getArgNumber()), nm);
    }
    sc->arg_val.emplace("arg" + std::to_string(j), ba);
  }
}

// Wraps every tf.If as tf.residue (tf-dialect passthrough to the sink; TF
// types are legal on the carrier, branch bodies stay, dispatch by attr).
void WrapIfAsFallbackCall(mlir::ModuleOp module) {
  int n_if = 0;
  llvm::SmallVector<mlir::Operation*> ifs;
  module.walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "tf.If") ifs.push_back(op);
  });
  for (auto* op : ifs) {
    mlir::OpBuilder b(op);
    mlir::OperationState st(op->getLoc(), "tf.residue");
    st.operands.append(op->operand_begin(), op->operand_end());
    st.addTypes(op->getResultTypes());
    st.addAttribute("call_target_name", b.getStringAttr("tf.If"));
    st.addAttribute("residue_reason", b.getStringAttr("control_flow_carrier"));
    CopyNonReservedAttrs(st, op);  // then/else/Tin/Tout for sink dispatch
    auto* call = b.create(st);
    op->replaceAllUsesWith(call);
    op->erase();
    ++n_if;
  }
  if (n_if != 0)
    std::cerr << "[import-route] wrapped " << n_if
              << " tf.If as tf.residue (TF-op passthrough)\n";
}

// ---- restore session state + stage helpers ----

// State shared by the restore stages below.
struct RestoreCtx {
  std::vector<std::unique_ptr<BodyScope>> scopes;
  std::map<mlir::Operation*, std::string> op_name;  // op -> node name
  // BlockArgument has no operator<; key on (block ptr, arg number).
  std::map<std::pair<mlir::Block*, unsigned>, std::string> arg_name;
  BodyScope* main_sc = nullptr;
  mlir::Operation* func_op = nullptr;
  BodyScope* makeScope() {
    scopes.emplace_back(std::make_unique<BodyScope>());
    return scopes.back().get();
  }
};

// All names in a loc tree (FusedLocs recurse).
void CollectLocNames(mlir::Location loc,
                     llvm::SmallVectorImpl<std::string>* out) {
  if (auto nl = llvm::dyn_cast<mlir::NameLoc>(loc)) {
    out->push_back(nl.getName().str());
    return;
  }
  if (auto fl = llvm::dyn_cast<mlir::FusedLoc>(loc))
    for (mlir::Location l : fl.getLocations()) CollectLocNames(l, out);
}

// X@_functionalize_if_then_branch_N[__i] -> X[__i] (table keys use pb
// names, locs carry suffixes; both spellings must resolve).
std::string StripFuncSuffix(const std::string& n) {
  auto at = n.find("@_functionalize");
  if (at == std::string::npos) return n;
  auto after = n.find("__", at);
  return n.substr(0, at) +
         (after == std::string::npos ? "" : n.substr(after));
}

// Fatal invariant: prints the Switch-name diagnostics (the historical
// failure signature) and returns the legacy code 4.
int RestoreFatal(const RestoreCtx& rc, const std::string& msg) {
  std::cerr << "[FATAL] " << msg << "\n";
  int hits = 0;
  for (mlir::Operation& op : rc.main_sc->block->getOperations()) {
    llvm::SmallVector<std::string, 4> names;
    CollectLocNames(op.getLoc(), &names);
    for (auto& n : names)
      if (n.find("Switch") != std::string::npos && hits++ < 10)
        std::cerr << "[diag] op=" << op.getName().getStringRef().str()
                  << " loc_name=" << n << "\n";
  }
  std::cerr << "[diag] switch-name hits = " << hits << "\n";
  return 4;
}

// Name of the node that defines v (op loc name, else bound block arg).
std::string NodeNameOf(const RestoreCtx& rc, mlir::Value v) {
  if (auto* d = v.getDefiningOp()) {
    auto it = rc.op_name.find(d);
    return it != rc.op_name.end() ? it->second : std::string();
  }
  auto ba = llvm::dyn_cast<mlir::BlockArgument>(v);
  if (!ba) return "";
  auto it = rc.arg_name.find({ba.getParentBlock(), ba.getArgNumber()});
  return it != rc.arg_name.end() ? it->second : std::string();
}

// One scope per flattenable block; recurses into funcs / executor graphs.
void ScanBlock(mlir::Block& blk, BodyScope* sc, RestoreCtx& rc) {
  static const std::set<std::string> kRecurseInto = {"func.func",
                                                     "tf_executor.GraphOp"};
  sc->block = &blk;
  for (mlir::Operation& op : blk.getOperations()) {
    llvm::SmallVector<std::string> names;
    CollectLocNames(op.getLoc(), &names);
    for (auto& n : names) {
      sc->by_name.emplace(n, &op);  // first wins (importer: unique)
      if (auto colon = n.find(':'); colon != std::string::npos)
        sc->by_name.emplace(n.substr(0, colon), &op);
      std::string sn = StripFuncSuffix(n);
      if (sn != n) sc->by_name.emplace(sn, &op);
      if (!rc.op_name.count(&op) && !n.empty() && n.back() != ':')
        rc.op_name.emplace(&op, sn);  // prefer pb name (capture align)
    }
    if (!kRecurseInto.count(op.getName().getStringRef().str())) continue;
    for (mlir::Region& reg : op.getRegions()) {
      if (reg.empty()) continue;
      ScanBlock(reg.front(), rc.makeScope(), rc);
    }
  }
}

// Walks func.func ops into scopes; the first one is @main.
bool CollectScopes(mlir::ModuleOp module, RestoreCtx& rc) {
  module.walk([&](mlir::Operation* f) {
    if (f->getName().getStringRef() != "func.func") return;
    BodyScope* sc = rc.makeScope();
    if (rc.main_sc == nullptr) {
      rc.main_sc = sc;
      rc.func_op = f;
    }
    if (auto sym = f->getAttrOfType<mlir::StringAttr>("sym_name"))
      sc->sym = sym.str();
    ScanBlock(f->getRegion(0).front(), sc, rc);
  });
  return rc.main_sc != nullptr && rc.func_op != nullptr;
}

// Cross-block captures: branch-body args aligned to carrier operands via
// the branch symbol attrs. If/While carry one SymbolRefAttr, Case an
// ArrayAttr of them.
void BindBranchCarriers(mlir::ModuleOp module, RestoreCtx& rc) {
  auto name_of = [&rc](mlir::Value v) { return NodeNameOf(rc, v); };
  module.walk([&](mlir::Operation* op) {
    llvm::StringRef n = op->getName().getStringRef();
    llvm::SmallVector<std::pair<llvm::StringRef, int>, 2> branches;
    if (n == "tf.If") {
      branches = {{"then_branch", 1}, {"else_branch", 1}};
    } else if (n == "tf.While") {
      branches = {{"cond", 0}, {"body", 0}};
    } else if (n == "tf.Case") {
      branches = {{"branches", 1}};
    } else {
      return;
    }
    for (auto& [attr_name, off] : branches) {
      llvm::SmallVector<mlir::Attribute, 4> refs;
      if (auto attr = op->getAttrOfType<mlir::SymbolRefAttr>(attr_name))
        refs.push_back(attr);
      else if (auto arr = op->getAttrOfType<mlir::ArrayAttr>(attr_name))
        refs.append(arr.begin(), arr.end());
      for (mlir::Attribute a : refs) {
        auto sym = llvm::dyn_cast<mlir::SymbolRefAttr>(a);
        if (!sym) continue;
        std::string target = sym.getLeafReference().str();
        for (auto& sc : rc.scopes) {
          if (sc->sym != target) continue;
          BindBranchCaptures(op, off, sc.get(), &rc.arg_name, name_of);
        }
      }
    }
  });
}

// Rebuilds one archived custom op: locate scope and slot stand-ins,
// resolve operands, insert at a dominance-safe point, swap the stand-ins
// for the rebuilt carrier (aliases re-keyed). 0 ok / 1 skip / 4 fatal.
int RestoreOneCustomOp(const std::string& name, const tensorflow::NodeDef& nd,
                       RestoreCtx& rc, mlir::OpBuilder& b,
                       mlir::MLIRContext& context) {
  std::string pb_name = name.substr(0, name.find("@_functionalize"));
  BodyScope* host = nullptr;
  for (auto& scp : rc.scopes) {
    if (scp->by_name.count(name)) {
      host = scp.get();
      break;
    }
  }
  if (host == nullptr) return 1;  // stand-in lives in no scanned block
  BodyScope& sc = *host;
  auto git = g_nodes.find(pb_name);
  int n = git != g_nodes.end() ? (int)git->second.out_dtypes.size() : 1;
  if (n < 1)  // side-effect-only op: no slot to hang the carrier on
    return RestoreFatal(rc, "custom-op '" + name + "' has no dtype slots");
  llvm::SmallVector<mlir::Operation*> slot_ops(n, nullptr);
  for (int i = 0; i < n; ++i) {
    std::string nm = i == 0 ? name : name + "__" + std::to_string(i);
    auto sit = sc.by_name.find(nm);
    if (sit == sc.by_name.end())
      return RestoreFatal(
          rc, "custom-op stand-in placeholder '" + name + "' not found");
    slot_ops[i] = sit->second;
  }
  mlir::OperationState st(slot_ops[0]->getLoc(),
                          mlir::OperationName("tf.residue", &context));
  st.addAttribute("call_target_name", b.getStringAttr("tf." + nd.op()));
  for (auto& in : nd.input()) {
    InputRef r;
    {
      // three-segment node:<argname>:<slot> refs: normalize (ResolveRef
      // only knows the two-segment form).
      auto c1 = in.find(':');
      auto c2 =
          c1 == std::string::npos ? std::string::npos : in.find(':', c1 + 1);
      if (c2 != std::string::npos) {
        std::string base = in.substr(0, c1);
        r.control = !base.empty() && base[0] == '^';
        r.node = r.control ? base.substr(1) : base;
        r.slot = ParseSlot(in.c_str() + c2 + 1);
      } else {
        r = ResolveRef(in);
      }
    }
    if (r.control) continue;  // control edges are not restored
    mlir::Value v = ResolveOperand(sc, r);
    if (!v)
      return RestoreFatal(rc, name + " input " + in +
                                  " unresolved in imported IR");
    st.operands.push_back(v);
  }
  for (int i = 0; i < n; ++i)
    st.addTypes(slot_ops[i]->getResult(0).getType());
  AppendCustomOpAttrs(st, nd, b, name);
  st.addAttribute("residue_reason",
                  b.getStringAttr("unregistered_custom_op"));
  // Insertion point = first slot stand-in position (anchor-pinned), never
  // before any data dependency (latest+1, defensive).
  {
    std::map<mlir::Operation*, int> order;
    int idx = 0;
    for (mlir::Operation& op : sc.block->getOperations()) order[&op] = idx++;
    int latest = -1;
    for (mlir::Value v : st.operands)
      if (auto* d = v.getDefiningOp()) {
        auto oit = order.find(d);
        if (oit != order.end()) latest = std::max(latest, oit->second);
      }
    int first_slot = std::numeric_limits<int>::max();
    for (mlir::Operation* sop : slot_ops) {
      auto oit = order.find(sop);
      if (oit != order.end()) first_slot = std::min(first_slot, oit->second);
    }
    int target = std::max(
        latest + 1,
        first_slot == std::numeric_limits<int>::max() ? 0 : first_slot);
    if (target <= 0) {
      b.setInsertionPointToStart(sc.block);
    } else {
      auto bit = sc.block->begin();
      std::advance(bit, target - 1);
      b.setInsertionPointAfter(&*bit);
    }
  }
  mlir::Operation* new_op = b.create(st);
  for (int i = 0; i < n; ++i) {
    slot_ops[i]->getResult(0).replaceAllUsesWith(new_op->getResult(i));
    // Drop EVERY alias pointing at this stand-in (loc variants: bare name,
    // ":arg" base, functionalize-stripped pb name) -- a stale alias would
    // leave a dangling Operation* for later lookups (use-after-free).
    for (auto it = sc.by_name.begin(); it != sc.by_name.end();) {
      if (it->second == slot_ops[i])
        it = sc.by_name.erase(it);
      else
        ++it;
    }
    slot_ops[i]->erase();
  }
  // Re-key both spellings so chained custom ops resolve by either the
  // archived (functionalized) name or the plain pb name.
  sc.by_name[name] = new_op;
  if (pb_name != name) sc.by_name[pb_name] = new_op;
  rc.op_name.emplace(new_op, name);
  return 0;
}

// Non-numeric registered ops (string/variant/resource results) -> residue
// guard; returns the wrapped count.
int WrapNonNumericOps(mlir::ModuleOp module, mlir::OpBuilder& b,
                      mlir::MLIRContext& context) {
  llvm::SmallVector<mlir::Operation*> wrap_targets;
  module.walk([&](mlir::Operation* op) {
    llvm::StringRef nm = op->getName().getStringRef();
    if (!nm.starts_with("tf.") || nm == "tf.residue") return;
    if (g_opspecs.count(nm.drop_front(3).str()) == 0) return;  // custom op
    if (op->getNumResults() == 0) return;
    bool numeric = true;
    for (mlir::Type t : op->getResultTypes()) {
      auto tt = llvm::dyn_cast<mlir::TensorType>(t);
      if (tt == nullptr ||
          !llvm::isa_and_nonnull<mlir::FloatType, mlir::IntegerType,
                                 mlir::ComplexType>(tt.getElementType())) {
        numeric = false;
        break;
      }
    }
    if (!numeric) wrap_targets.push_back(op);
  });
  for (auto it = wrap_targets.rbegin(); it != wrap_targets.rend(); ++it) {
    mlir::Operation& op = **it;
    llvm::StringRef nm = op.getName().getStringRef();
    mlir::OperationState st(op.getLoc(),
                            mlir::OperationName("tf.residue", &context));
    st.operands.append(op.operand_begin(), op.operand_end());
    st.addTypes(op.getResultTypes());
    st.addAttribute("call_target_name", b.getStringAttr(nm));
    st.addAttribute("residue_reason", b.getStringAttr("non_numeric_type"));
    CopyNonReservedAttrs(st, &op);
    b.setInsertionPoint(&op);
    mlir::Operation* call = b.create(st);
    op.replaceAllUsesWith(call);
    op.erase();
  }
  return (int)wrap_targets.size();
}

// Cast/Identity passthrough: result shape follows the operand (operand
// wins over conflicting _output_shapes).
void AlignCastIdentity(mlir::ModuleOp module) {
  module.walk([&](mlir::Operation* op) {
    llvm::StringRef nm = op->getName().getStringRef();
    if (nm != "tf.Cast" && nm != "tf.Identity") return;
    if (op->getNumOperands() == 0 || op->getNumResults() == 0) return;
    auto ot = llvm::dyn_cast<mlir::TensorType>(op->getOperand(0).getType());
    auto rt = llvm::dyn_cast<mlir::TensorType>(op->getResult(0).getType());
    if (ot == nullptr || rt == nullptr) return;
    bool differs =
        ot.hasRank() != rt.hasRank() ||
        (ot.hasRank() && rt.hasRank() && ot.getShape() != rt.getShape());
    if (!differs) return;
    op->getResult(0).setType(ot.clone(rt.getElementType()));
  });
}

// tf.entry_function + tf.arg_names rewritten to the real contract.
void RewriteEntryContract(mlir::Operation* func_op, const Options& opt,
                          const std::vector<FeedSpec>& feeds,
                          mlir::OpBuilder& b, mlir::MLIRContext& context) {
  std::string in_str, out_str;
  for (auto& s : opt.inputs)
    in_str += (in_str.empty() ? "" : ",") + s.substr(0, s.find(':')) + ":0";
  for (auto& name : opt.outputs)
    out_str += (out_str.empty() ? "" : ",") + name + ":0";
  mlir::NamedAttrList el;
  el.set("inputs", b.getStringAttr(in_str));
  el.set("outputs", b.getStringAttr(out_str));
  el.set("control_outputs", b.getStringAttr(""));
  func_op->setAttr("tf.entry_function", el.getDictionary(&context));
  mlir::SmallVector<mlir::Attribute, 16> arg_names;
  for (auto& s : feeds) arg_names.push_back(b.getStringAttr(s.name));
  func_op->setAttr("tf.arg_names", b.getArrayAttr(arg_names));
}

// MLIR-side restore: rebuilds custom ops from custom_ops, wraps non-numeric
// registered ops and tf.If as tf.residue, aligns Cast/Identity shapes,
// rewrites tf.entry_function. 0 ok / 1 recoverable / 4 fatal invariant.
int RestoreOpsImpl(mlir::ModuleOp module, const Options& opt,
                   const std::vector<FeedSpec>& feeds,
                   const CustomOpTable& custom_ops, mlir::OpBuilder& b,
                   mlir::MLIRContext& context) {
  RestoreCtx rc;
  if (!CollectScopes(module, rc)) {
    std::cerr << "[FATAL] imported module has no func\n";
    return 1;
  }
  if ((int)rc.main_sc->block->getNumArguments() != (int)feeds.size()) {
    std::cerr << "[FATAL] imported func has "
              << rc.main_sc->block->getNumArguments() << " args but "
              << feeds.size() << " feeds were declared\n";
    return 1;
  }
  for (size_t i = 0; i < feeds.size(); ++i) {
    rc.main_sc->arg_val.emplace(feeds[i].name,
                                rc.main_sc->block->getArgument((int)i));
    rc.arg_name.emplace(std::make_pair(rc.main_sc->block, (unsigned)i),
                        feeds[i].name);
  }
  BindBranchCarriers(module, rc);

  int restored = 0;
  for (auto& kv : custom_ops) {
    int r = RestoreOneCustomOp(kv.first, kv.second, rc, b, context);
    if (r == 4) return 4;
    if (r == 0) ++restored;
  }
  int n_wrapped = WrapNonNumericOps(module, b, context);
  AlignCastIdentity(module);
  RewriteEntryContract(rc.func_op, opt, feeds, b, context);
  WrapIfAsFallbackCall(module);
  std::cerr << "[import-route] restored: custom_ops=" << custom_ops.size()
            << " (" << restored << " ops, " << rc.scopes.size() << " scopes)"
            << " fallback_wrapped=" << n_wrapped << "\n";
  return 0;
}

struct RestoreOpsPass
    : public mlir::PassWrapper<RestoreOpsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RestoreOpsPass)
  RestoreOpsPass(const Options& opt, const std::vector<FeedSpec>& feeds,
                 const CustomOpTable& custom_ops, mlir::OpBuilder& b)
      : opt_(opt), feeds_(feeds), custom_ops_(custom_ops), b_(b) {}
  llvm::StringRef getArgument() const override { return "pb-restore-ops"; }
  void runOnOperation() override {
    code_ = RestoreOpsImpl(getOperation(), opt_, feeds_, custom_ops_, b_,
                           *getOperation().getContext());
    if (code_ != 0) signalPassFailure();
  }
  int errorCode() const { return code_; }

  const Options& opt_;
  const std::vector<FeedSpec>& feeds_;
  const CustomOpTable& custom_ops_;
  mlir::OpBuilder& b_;
  int code_ = 0;
};

}  // namespace

int RunRestoreOpsPass(mlir::ModuleOp module, const Options& opt,
                      const std::vector<FeedSpec>& feeds,
                      const CustomOpTable& custom_ops, mlir::OpBuilder& b,
                      mlir::MLIRContext& context) {
  mlir::PassManager pm(&context);
  pm.enableVerifier(false);
  auto pass = std::make_unique<RestoreOpsPass>(opt, feeds, custom_ops, b);
  RestoreOpsPass* raw = pass.get();
  pm.addPass(std::move(pass));
  if (failed(pm.run(module))) {
    if (raw->errorCode() == 4)
      exit(4);  // legacy contract: fatal unwinding is unsafe past this point
    return raw->errorCode() != 0 ? raw->errorCode() : 1;
  }
  return 0;
}

}  // namespace pb_passes
