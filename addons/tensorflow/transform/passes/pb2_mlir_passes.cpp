#include "common/pb2_common.h"
// MLIR-domain passes of the official-import route; RestoreOps lives in
// pb2_restore_pass.cpp (failure-code plumbing).
#include <map>
#include <set>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "passes/pb2_mlir_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace pb_passes {
namespace {

// ---- shared pb-consultation helpers ----

// All names in a loc tree (mixed FusedLocs carry both odml's "OpType:"
// tag and the real pb name).
void CollectLocNamesRec(mlir::Location loc,
                        llvm::SmallVectorImpl<std::string>* out) {
  if (auto nl = llvm::dyn_cast<mlir::NameLoc>(loc)) {
    out->push_back(nl.getName().str());
    return;
  }
  if (auto fl = llvm::dyn_cast<mlir::FusedLoc>(loc))
    for (mlir::Location l : fl.getLocations()) CollectLocNamesRec(l, out);
}

// pb name index for the passes that consult the original GraphDef.
using PbIndex = llvm::DenseMap<llvm::StringRef, const tensorflow::NodeDef*>;

inline PbIndex BuildPbIndex(const tensorflow::GraphDef& graph) {
  PbIndex pb;
  for (const tensorflow::NodeDef& n : graph.node())
    pb.try_emplace(n.name(), &n);
  return pb;
}

// First loc-tree name present in the pb index ("OpType:" tag members never
// match, so this is a pure data decision). Empty = no pb host.
inline std::string PbHostOf(mlir::Location loc, const PbIndex& pb) {
  llvm::SmallVector<std::string, 4> names;
  CollectLocNamesRec(loc, &names);
  for (const std::string& n : names)
    if (pb.count(llvm::StringRef(n))) return n;
  return {};
}

// Pure pb data-input name: rejects control edges (^node) and strips the
// ":N" output-slot suffix (TF node names never contain ':').
bool PlainInputName(const std::string& in, std::string& out) {
  if (in.empty() || in[0] == '^') return false;
  size_t colon = in.rfind(':');
  if (colon == std::string::npos) {
    out = in;
  } else {
    for (size_t i = colon + 1; i < in.size(); ++i)
      if (!isdigit((unsigned char)in[i])) return false;
    out = in.substr(0, colon);
  }
  return !out.empty();
}

// Element-wise veto: only int/float/bool dtypes with identical flattened
// values pass; anything else (string carriers, packed i8 sandwiches,
// dtype-converted forms) rejects so no wrong rename can land.
bool ValueMatches(const tensorflow::TensorProto& t,
                  mlir::DenseElementsAttr dense) {
  mlir::Type elem = dense.getElementType();
  unsigned bits = elem.getIntOrFloatBitWidth();
  tensorflow::DataType dt = t.dtype();
  bool ok = (dt == tensorflow::DT_INT32 && bits == 32) ||
            (dt == tensorflow::DT_INT64 && bits == 64) ||
            (dt == tensorflow::DT_INT8 && bits == 8) ||
            (dt == tensorflow::DT_BOOL && bits == 1) ||
            (dt == tensorflow::DT_FLOAT && elem.isF32()) ||
            (dt == tensorflow::DT_DOUBLE && elem.isF64());
  if (!ok) return false;
  bool intSide = elem.isIntOrIndex();
  llvm::StringRef raw(t.tensor_content());
  int64_t esz = (bits + 7) / 8;
  int64_t n = dense.getNumElements();
  int64_t pn = !raw.empty()                  ? (int64_t)raw.size() / esz
               : dt == tensorflow::DT_INT64  ? t.int64_val_size()
               : dt == tensorflow::DT_FLOAT  ? t.float_val_size()
               : dt == tensorflow::DT_DOUBLE ? t.double_val_size()
               : dt == tensorflow::DT_BOOL   ? t.bool_val_size()
                                             : t.int_val_size();
  if (pn != n) return false;
  if (intSide) {
    auto vi = dense.getValues<llvm::APInt>().begin();
    for (int64_t i = 0; i < n; ++i, ++vi) {
      int64_t pv = 0;
      if (!raw.empty()) {
        if (bits == 1) {
          pv = (unsigned char)raw[i] != 0;
        } else {
          for (int64_t b = esz - 1; b >= 0; --b)
            pv = (pv << 8) | (unsigned char)raw[i * esz + b];
        }
      } else if (dt == tensorflow::DT_INT64) {
        pv = t.int64_val(i);
      } else if (dt == tensorflow::DT_BOOL) {
        pv = t.bool_val(i) ? 1 : 0;
      } else {
        pv = t.int_val(i);
      }
      if ((*vi).getSExtValue() != pv) return false;
    }
  } else {
    auto vf = dense.getValues<llvm::APFloat>().begin();
    for (int64_t i = 0; i < n; ++i, ++vf) {
      double pv;
      if (!raw.empty()) {
        if (bits == 32) {
          float f;
          std::memcpy(&f, raw.data() + i * esz, 4);
          pv = f;
        } else {
          std::memcpy(&pv, raw.data() + i * esz, 8);
        }
      } else if (dt == tensorflow::DT_FLOAT) {
        pv = t.float_val(i);
      } else {
        pv = t.double_val(i);
      }
      if ((*vf).convertToDouble() != pv) return false;
    }
  }
  return true;
}

// The original TF op name lives in the "OpType:"-suffixed member of odml's
// fused loc trees (single token before the trailing ':').
std::string SrcOpTag(mlir::Operation* op) {
  std::string tag;
  llvm::SmallVector<mlir::Location, 8> locStack{op->getLoc()};
  while (!locStack.empty() && tag.empty()) {
    mlir::Location l = locStack.pop_back_val();
    if (auto nl = llvm::dyn_cast<mlir::NameLoc>(l)) {
      llvm::StringRef n = nl.getName();
      if (n.ends_with(":") && n.size() > 1 &&
          n.drop_back().find(' ') == llvm::StringRef::npos)
        tag = n.drop_back().str();
    } else if (auto fl = llvm::dyn_cast<mlir::FusedLoc>(l)) {
      for (mlir::Location in : fl.getLocations()) locStack.push_back(in);
    }
  }
  return tag;
}

// Generic tf.residue carrier at the builder's insertion point. Only the
// two identity attrs: compare/Fill fallbacks start from odml conversion
// products (original TF attrs gone); the wrap path copies attrs itself.
mlir::Operation* BuildResidueCarrier(mlir::OpBuilder& b, mlir::Location loc,
                                     mlir::ValueRange operands,
                                     mlir::TypeRange types,
                                     llvm::StringRef target,
                                     llvm::StringRef reason) {
  mlir::OperationState st(
      loc, mlir::OperationName("tf.residue", b.getContext()));
  st.addOperands(operands);
  st.addTypes(types);
  st.addAttribute("call_target_name", b.getStringAttr(target));
  st.addAttribute("residue_reason", b.getStringAttr(reason));
  return b.create(st);
}

// ---- NormalizeRefTypes (was pb2_import.cpp) -------------------------------

// Forwarding refs are equivalent to plain tensors in inference graphs; ref
// semantics only exist in session variable forwarding. Stablehlo cannot
// spell ref types, so deref after import.
mlir::Type DeRef(mlir::Type t) {
  if (auto tt = llvm::dyn_cast<mlir::TensorType>(t)) {
    if (llvm::isa<mlir::tf_type::TensorFlowRefType>(tt.getElementType()))
      return tt.clone(DeRef(tt.getElementType()));
    return t;
  }
  if (auto rt = llvm::dyn_cast<mlir::tf_type::TensorFlowRefType>(t))
    return rt.RemoveRef();
  return t;
}

void NormalizeRefTypesImpl(mlir::ModuleOp module) {
  module.walk([&](mlir::Operation* op) {
    for (mlir::OpResult r : op->getOpResults()) {
      mlir::Type nt = DeRef(r.getType());
      if (nt != r.getType()) r.setType(nt);
    }
    if (op->getName().getStringRef() == "func.func") {
      auto fta = op->getAttrOfType<mlir::TypeAttr>("function_type");
      auto fty =
          fta ? llvm::dyn_cast<mlir::FunctionType>(fta.getValue()) : nullptr;
      if (fty != nullptr) {
        llvm::SmallVector<mlir::Type> in(fty.getInputs().begin(),
                                         fty.getInputs().end());
        llvm::SmallVector<mlir::Type> out(fty.getResults().begin(),
                                          fty.getResults().end());
        bool changed = false;
        // Results too: body returns were already de-ref'd above, so the
        // signature must follow or the func would end up type-mismatched.
        for (auto* types : {&in, &out}) {
          for (auto& t : *types) {
            mlir::Type nt = DeRef(t);
            if (nt != t) {
              t = nt;
              changed = true;
            }
          }
        }
        if (changed)
          op->setAttr("function_type",
                      mlir::TypeAttr::get(mlir::FunctionType::get(
                          op->getContext(), in, out)));
      }
      if (op->getNumRegions() > 0 && !op->getRegion(0).empty())
        for (mlir::BlockArgument a : op->getRegion(0).front().getArguments()) {
          mlir::Type nt = DeRef(a.getType());
          if (nt != a.getType()) a.setType(nt);
        }
    }
    for (auto named : op->getAttrs()) {
      auto ta = llvm::dyn_cast<mlir::TypeAttr>(named.getValue());
      if (ta == nullptr) continue;
      mlir::Type nt = DeRef(ta.getValue());
      if (nt != ta.getValue())
        op->setAttr(named.getName(), mlir::TypeAttr::get(nt));
    }
  });
}

struct NormalizeRefTypesPass
    : public mlir::PassWrapper<NormalizeRefTypesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeRefTypesPass)
  llvm::StringRef getArgument() const override { return "pb-normalize-ref"; }
  void runOnOperation() override { NormalizeRefTypesImpl(getOperation()); }
};

// ---- FlattenExecutor (was pb2_import.cpp) ---------------------------------

void FlattenExecutorImpl(mlir::ModuleOp module) {
  struct GraphFlattener {
    int n = 0;
    void operator()(mlir::tf_executor::GraphOp graph) {
      mlir::Region& gr = graph.getRegion();
      if (!gr.hasOneBlock()) return;  // irregular shape: pipeline handles it
      mlir::Block& gb = gr.front();
      auto fetch =
          llvm::dyn_cast<mlir::tf_executor::FetchOp>(gb.getTerminator());
      if (fetch == nullptr) return;
      llvm::SmallVector<mlir::Operation*> region_ops;
      for (mlir::Operation& op : gb.getOperations()) {
        if (llvm::isa<mlir::tf_executor::IslandOp>(op) ||
            llvm::isa<mlir::tf_executor::FetchOp>(op))
          continue;
        llvm::StringRef n = op.getName().getStringRef();
        if (n == "tf.If" || n == "tf.While" || n == "tf.Case") {
          region_ops.push_back(&op);  // self-contained carrier, moved whole
          continue;
        }
        if (std::getenv("PB_VERBOSE") != nullptr)
          std::cerr << "[flatten-skip] non-island op: " << n.str() << "\n";
        return;
      }

      // 1. Collect islands, then preflight before any mutation: zip()
      //    would silently truncate on mismatches and erase() would dangle
      //    results -- bail out whole (same policy as hasOneBlock).
      llvm::SmallVector<mlir::tf_executor::IslandOp> islands;
      for (auto island : gb.getOps<mlir::tf_executor::IslandOp>())
        islands.push_back(island);
      if (graph->getNumResults() != fetch.getNumOperands()) return;
      for (auto island : islands) {
        mlir::Region& ibody = island.getBody();
        if (!ibody.hasOneBlock()) return;
        auto y = llvm::dyn_cast<mlir::tf_executor::YieldOp>(
            ibody.front().getTerminator());
        if (y == nullptr) return;
        unsigned nData = island.getNumResults();
        while (nData > 0 &&
               llvm::isa<mlir::tf_executor::ControlType>(
                   island.getResult(nData - 1).getType()))
          --nData;
        if (y.getNumOperands() != nData) return;
        for (unsigned i = nData; i < island.getNumResults(); ++i)
          if (!island.getResult(i).use_empty()) return;
      }

      // 2. strip all control dependencies of the graph.
      auto strip_ctrl = [&](mlir::Operation* op) {
        llvm::SmallVector<unsigned> keep;
        for (unsigned i = 0; i < op->getNumOperands(); ++i)
          if (!llvm::isa<mlir::tf_executor::ControlType>(
                  op->getOperand(i).getType()))
            keep.push_back(i);
        llvm::SmallVector<mlir::Value> new_ops;
        for (unsigned i : keep) new_ops.push_back(op->getOperand(i));
        op->setOperands(new_ops);
      };
      for (auto island : islands) strip_ctrl(island);
      strip_ctrl(fetch);

      // 3. graph results <- fetch operands.
      for (unsigned i = 0; i < graph->getNumResults(); ++i)
        graph->getResult(i).replaceAllUsesWith(fetch.getOperand(i));

      // 4. per island: lift inner ops before the graph, RAUW results.
      for (auto island : islands) {
        mlir::Block& ib = island.getBody().front();
        auto yield = llvm::cast<mlir::tf_executor::YieldOp>(ib.getTerminator());
        for (mlir::Operation& op :
             llvm::make_early_inc_range(ib.getOperations())) {
          if (llvm::isa<mlir::tf_executor::YieldOp>(op)) continue;
          op.moveBefore(graph);
        }
        for (unsigned i = 0; i < yield.getNumOperands(); ++i)
          island.getResult(i).replaceAllUsesWith(yield.getOperand(i));
        island.erase();
      }
      // 5. region carriers move out last (operands already RAUW'd to real
      //    in-block values, so dominance holds).
      for (mlir::Operation* rop : region_ops) rop->moveBefore(graph);
      fetch.erase();
      graph.erase();
      ++n;
    }
  };
  // Collect before processing: erase inside walk is iterator-invalidating.
  llvm::SmallVector<mlir::tf_executor::GraphOp> graphs;
  module.walk([&](mlir::tf_executor::GraphOp g) { graphs.push_back(g); });
  GraphFlattener fl;
  for (auto g : graphs) fl(g);
  std::cerr << "[import-route] flattened executor graphs = " << fl.n << "\n";
}

struct FlattenExecutorPass
    : public mlir::PassWrapper<FlattenExecutorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenExecutorPass)
  llvm::StringRef getArgument() const override { return "pb-flatten-executor"; }
  void runOnOperation() override { FlattenExecutorImpl(getOperation()); }
};

// ---- G2: functionalize-suffix strip (was pb2_import.cpp) ------------------

std::string StripFunctionalizeSuffix(const std::string& n) {
  auto at = n.find("@_functionalize");
  if (at == std::string::npos) return n;
  auto after = n.find("__", at);
  return n.substr(0, at) + (after == std::string::npos ? "" : n.substr(after));
}

mlir::Location StripLocSuffixRec(mlir::Location loc, mlir::MLIRContext& ctx) {
  if (auto nl = llvm::dyn_cast<mlir::NameLoc>(loc)) {
    std::string s = StripFunctionalizeSuffix(nl.getName().str());
    if (s == nl.getName().str()) return loc;
    return mlir::NameLoc::get(mlir::StringAttr::get(&ctx, s));
  }
  if (auto fl = llvm::dyn_cast<mlir::FusedLoc>(loc)) {
    llvm::SmallVector<mlir::Location> inner;
    bool changed = false;
    for (mlir::Location l : fl.getLocations()) {
      mlir::Location r = StripLocSuffixRec(l, ctx);
      inner.push_back(r);
      changed |= (r != l);
    }
    return changed ? mlir::FusedLoc::get(&ctx, inner, fl.getMetadata()) : loc;
  }
  return loc;
}

void StripFunctionalizeSuffixesImpl(mlir::ModuleOp module) {
  module->walk([&](mlir::Operation* op) {
    mlir::Location r = StripLocSuffixRec(op->getLoc(), *op->getContext());
    if (r != op->getLoc()) op->setLoc(r);
  });
}

struct StripFunctionalizeSuffixPass
    : public mlir::PassWrapper<StripFunctionalizeSuffixPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripFunctionalizeSuffixPass)
  llvm::StringRef getArgument() const override {
    return "pb-strip-func-suffix";
  }
  void runOnOperation() override {
    StripFunctionalizeSuffixesImpl(getOperation());
  }
};

// ---- G3: constant host anchoring ----

struct AnchorConstantsToUsersPass
    : public mlir::PassWrapper<AnchorConstantsToUsersPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnchorConstantsToUsersPass)
  llvm::StringRef getArgument() const override {
    return "pb-anchor-constants-to-users";
  }
  llvm::StringRef getDescription() const override {
    return "give unknown-loc constants the FusedLoc of their named users";
  }
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* ctx = module.getContext();
    int anchored = 0;
    module->walk([&](mlir::Operation* op) {
      llvm::StringRef nm = op->getName().getStringRef();
      if (nm != "stablehlo.constant" && nm != "arith.constant") return;
      if (!llvm::isa<mlir::UnknownLoc>(op->getLoc())) return;
      llvm::SmallVector<std::string, 4> names;
      for (mlir::OpOperand& use : op->getUses()) {
        llvm::SmallVector<std::string, 2> un;
        CollectLocNamesRec(use.getOwner()->getLoc(), &un);
        for (auto& n : un)
          if (!n.empty() && n.back() != ':') names.push_back(n);
      }
      if (names.empty()) return;
      llvm::sort(names);
      names.erase(std::unique(names.begin(), names.end()), names.end());
      llvm::SmallVector<mlir::Location, 4> locs;
      for (const std::string& n : names)
        locs.push_back(mlir::NameLoc::get(mlir::StringAttr::get(ctx, n)));
      op->setLoc(mlir::FusedLoc::get(ctx, locs));
      ++anchored;
    });
    if (anchored)
      std::cerr << "[anchor-users] constants anchored to users: " << anchored
                << "\n";
  }
};

// ---- Constant source-name tagging ----

namespace {
struct TagConstSourceNamePass
    : public mlir::PassWrapper<TagConstSourceNamePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TagConstSourceNamePass)
  llvm::StringRef getArgument() const override {
    return "pb-tag-const-source-name";
  }
  static std::string firstName(mlir::Location loc) {
    if (auto nl = llvm::dyn_cast<mlir::NameLoc>(loc)) return nl.getName().str();
    if (auto fl = llvm::dyn_cast<mlir::FusedLoc>(loc)) {
      for (mlir::Location l : fl.getLocations())
        if (auto n = firstName(l); !n.empty()) return n;
    }
    return "";
  }
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    int tagged = 0;
    module->walk([&](mlir::Operation* op) {
      if (op->getName().getStringRef() != "tf.Const") return;
      if (op->hasAttr("annc.source_node")) return;
      std::string n = firstName(op->getLoc());
      if (n.empty()) return;
      op->setAttr("annc.source_node", mlir::StringAttr::get(&getContext(), n));
      ++tagged;
    });
    if (tagged)
      std::cerr << "[tag-source] consts tagged with annc.source_node: "
                << tagged << "\n";
  }
};
}  // namespace

// ---- Constant source-loc restore ----

namespace {

// Constants: trace each unknown-loc constant through its carrier-user
// seats to the pb Const feeding that seat, veto by element-wise value
// equality, then restore the pb name as loc + annc.const_source.
int RestoreConstantLocs(mlir::ModuleOp module, const PbIndex& pb) {
  int restored = 0;
  module->walk([&](mlir::Operation* op) {
    llvm::StringRef nm = op->getName().getStringRef();
    if (nm != "stablehlo.constant" && nm != "arith.constant") return;
    if (!llvm::isa<mlir::UnknownLoc>(op->getLoc())) {
      return;
    }
    if (op->getNumResults() != 1) {
      return;
    }
    auto dense = op->getAttrOfType<mlir::DenseElementsAttr>("value");
    if (!dense) {
      return;
    }

    llvm::SmallVector<std::pair<mlir::Operation*, unsigned>, 4> seats;
    for (mlir::Operation* user : op->getResult(0).getUsers())
      for (unsigned i = 0; i < user->getNumOperands(); ++i)
        if (user->getOperand(i) == op->getResult(0))
          seats.emplace_back(user, i);

    // Both TF carriers (tf.residue and the odml custom_call @tf.X residue)
    // keep TF input order, so any named carrier host qualifies; helper ops
    // skip -- those constants have no pb input edge to consult.
    std::string origin;
    for (auto& seat : seats) {
      auto ct =
          seat.first->getAttrOfType<mlir::StringAttr>("call_target_name");
      bool tfCarrier =
          seat.first->getName().getStringRef() == "tf.residue" ||
          (seat.first->getName().getStringRef() == "stablehlo.custom_call" &&
           ct && ct.getValue().starts_with("tf."));
      if (!tfCarrier) {
        return;
      }
      if (!ct || !ct.getValue().starts_with("tf.")) {
        return;
      }
      std::string host = PbHostOf(seat.first->getLoc(), pb);
      if (host.empty()) {
        return;
      }
      auto hn = pb.find(llvm::StringRef(host));
      if (hn == pb.end()) {
        return;
      }
      if ((unsigned)hn->second->input_size() <= seat.second) {
        return;
      }
      std::string in;
      if (!PlainInputName(hn->second->input(seat.second), in)) {
        return;
      }
      if (origin.empty()) {
        origin = in;
      } else if (origin != in) {
        return;
      }
    }
    if (seats.empty()) {
      return;
    }
    if (origin.empty()) return;

    auto sn = pb.find(llvm::StringRef(origin));
    if (sn == pb.end()) {
      return;
    }
    const tensorflow::AttrValue* val = FindAttr(*sn->second, "value");
    if (val == nullptr) {
      return;
    }
    if (!ValueMatches(val->tensor(), dense)) {
      return;
    }

    op->setLoc(
        mlir::NameLoc::get(mlir::StringAttr::get(op->getContext(), origin)));
    op->setAttr("annc.const_source",
                mlir::StringAttr::get(op->getContext(), origin));
    ++restored;
  });
  return restored;
}

// StridedSlice sites: tag slice/real_dynamic_slice with the pb name of the
// begin vector Const (annc.begin_source, CSE-immune).
int TagSliceBegins(mlir::ModuleOp module, const PbIndex& pb) {
  int tagged = 0;
  module->walk([&](mlir::Operation* op) {
    llvm::StringRef nm = op->getName().getStringRef();
    if (nm != "stablehlo.slice" && nm != "stablehlo.real_dynamic_slice")
      return;
    if (op->hasAttr("annc.begin_source")) return;
    std::string host = PbHostOf(op->getLoc(), pb);
    if (host.empty()) return;
    auto hn = pb.find(llvm::StringRef(host));
    if (hn->second->op() != "StridedSlice" || hn->second->input_size() < 2)
      return;
    std::string begin;
    if (!PlainInputName(hn->second->input(1), begin)) return;
    auto bn = pb.find(llvm::StringRef(begin));
    if (bn == pb.end()) return;
    if (FindAttr(*bn->second, "value") == nullptr) return;
    op->setAttr("annc.begin_source",
                mlir::StringAttr::get(op->getContext(), begin));
    ++tagged;
  });
  return tagged;
}

// Per-site pack constants: CSE merges identical constants (loc carries
// only Pack names), so tag each site's OWN concatenate op instead -- BFS
// (depth 4) over pb input edges to the nearest Const, no op-type guessing.
int TagConcatConstSources(mlir::ModuleOp module, const PbIndex& pb) {
  int tagged = 0;
  module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() != "stablehlo.concatenate") return;
    if (op->hasAttr("annc.const_source")) return;
    std::string host = PbHostOf(op->getLoc(), pb);
    if (host.empty()) return;
    auto hn = pb.find(llvm::StringRef(host));
    if (hn == pb.end()) return;
    std::string packSrc;
    std::set<std::string> visited{host};
    llvm::SmallVector<std::string, 8> frontier{host};
    for (int depth = 0; depth < 4 && !frontier.empty(); ++depth) {
      llvm::SmallVector<std::string, 8> next;
      for (const auto& cur : frontier) {
        auto cit = pb.find(cur);
        if (cit == pb.end()) continue;
        if (FindAttr(*cit->second, "value") != nullptr) {
          packSrc = cur;
          break;
        }
        for (int slot = 0; slot < cit->second->input_size(); ++slot) {
          std::string cand;
          if (!PlainInputName(cit->second->input(slot), cand)) continue;
          if (visited.insert(cand).second) next.push_back(cand);
        }
      }
      if (!packSrc.empty()) break;
      frontier = std::move(next);
    }
    if (packSrc.empty()) return;
    op->setAttr("annc.const_source",
                mlir::StringAttr::get(op->getContext(), packSrc));
    ++tagged;
  });
  return tagged;
}

// Shape constants: the setDim loc resolves to the pb Reshape, whose
// input(1) is the shape Const; tag the stablehlo.constant operand
// (annc.const_source, same CSE-immune channel as begin_source).
int TagShapeConsts(mlir::ModuleOp module, const PbIndex& pb) {
  int tagged = 0;
  module->walk([&](mlir::Operation* op) {
    auto ct = op->getAttrOfType<mlir::StringAttr>("call_target_name");
    if (!ct || ct.getValue() != "annc.setDim") return;
    if (op->getNumOperands() < 2) return;
    auto shapeOp = op->getOperand(1).getDefiningOp();
    if (!shapeOp ||
        shapeOp->getName().getStringRef() != "stablehlo.constant")
      return;
    if (shapeOp->hasAttr("annc.const_source")) return;
    llvm::SmallVector<std::string, 4> names;
    CollectLocNamesRec(op->getLoc(), &names);
    std::string reshapePb;
    for (const std::string& n : names) {
      auto it = pb.find(llvm::StringRef(n));
      if (it != pb.end() && it->second->op() == "Reshape") {
        reshapePb = n;
        break;
      }
    }
    if (reshapePb.empty()) return;
    auto rn = pb.find(llvm::StringRef(reshapePb));
    if (rn == pb.end() || rn->second->input_size() < 2) return;
    std::string shapeConstName;
    if (!PlainInputName(rn->second->input(1), shapeConstName)) return;
    auto cn = pb.find(llvm::StringRef(shapeConstName));
    if (cn == pb.end()) return;
    if (FindAttr(*cn->second, "value") == nullptr) return;
    shapeOp->setAttr(
        "annc.const_source",
        mlir::StringAttr::get(shapeOp->getContext(), shapeConstName));
    ++tagged;
  });
  return tagged;
}

// Slot provenance: annc.prov on subtract ops, one "srcNode:srcPort" entry
// per operand from the pb data-input edges (FusionV2 binding contract; no
// attr -> the matcher drops the site, no type guessing).
int TagSubtractProvenance(mlir::ModuleOp module, const PbIndex& pb) {
  int tagged = 0;
  module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() != "stablehlo.subtract") return;
    if (op->hasAttr("annc.prov")) return;
    std::string host = PbHostOf(op->getLoc(), pb);
    if (host.empty()) return;
    auto hn = pb.find(llvm::StringRef(host));
    unsigned nOperands = op->getNumOperands();
    if (hn->second->input_size() < static_cast<int>(nOperands)) return;
    auto* ctx = op->getContext();
    llvm::SmallVector<mlir::Attribute, 4> items;
    for (unsigned i = 0; i < nOperands; ++i) {
      const std::string& edge = hn->second->input(static_cast<int>(i));
      if (edge.empty() || edge[0] == '^') return;
      std::string src = edge;
      int port = 0;
      size_t colon = edge.rfind(':');
      if (colon != std::string::npos) {
        for (size_t k = colon + 1; k < edge.size(); ++k)
          if (!isdigit(static_cast<unsigned char>(edge[k]))) return;
        port = std::atoi(edge.c_str() + colon + 1);
        src = edge.substr(0, colon);
      }
      if (src.empty() || !pb.count(llvm::StringRef(src))) return;
      items.push_back(mlir::StringAttr::get(
          ctx, src + ":" + std::to_string(port)));
    }
    op->setAttr("annc.prov", mlir::ArrayAttr::get(ctx, items));
    ++tagged;
  });
  return tagged;
}

// Universal fallback: every op with a pb-recoverable loc name gets
// annc.source_node + annc.source_op so downstream can trace any op back
// to its pb origin. Role-specific tags stay untouched.
int TagUniversalSource(mlir::ModuleOp module, const PbIndex& pb) {
  int tagged = 0;
  module->walk([&](mlir::Operation* op) {
    if (op->hasAttr("annc.source_node")) return;
    std::string n = PbHostOf(op->getLoc(), pb);
    if (n.empty()) return;
    op->setAttr("annc.source_node",
                mlir::StringAttr::get(op->getContext(), n));
    op->setAttr(
        "annc.source_op",
        mlir::StringAttr::get(
            op->getContext(),
            "tf." + pb.lookup(llvm::StringRef(n))->op()));
    ++tagged;
  });
  return tagged;
}

struct RestoreConstSourceLocPass
    : public mlir::PassWrapper<RestoreConstSourceLocPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RestoreConstSourceLocPass)
  RestoreConstSourceLocPass(const tensorflow::GraphDef& graph)
      : graph_(graph) {}
  llvm::StringRef getArgument() const override {
    return "pb-restore-const-source-loc";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    PbIndex pb = BuildPbIndex(graph_);
    int restored = RestoreConstantLocs(module, pb);
    int slices = TagSliceBegins(module, pb);

    int concats = TagConcatConstSources(module, pb);
    if (concats)
      std::cerr << "[restore-source-loc] concats tagged with const source: "
                << concats << "\n";
    int shapes = TagShapeConsts(module, pb);
    if (shapes)
      std::cerr << "[restore-source-loc] shape consts tagged with const "
                   "source: "
                << shapes << "\n";
    int provs = TagSubtractProvenance(module, pb);
    if (provs)
      std::cerr << "[restore-source-loc] subtract ops tagged with prov: "
                << provs << "\n";
    if (restored)
      std::cerr << "[restore-source-loc] constants restored to pb origin: "
                << restored << "\n";
    if (slices)
      std::cerr << "[restore-source-loc] slices tagged with begin source: "
                << slices << "\n";

    int sources = TagUniversalSource(module, pb);
    if (sources)
      std::cerr << "[restore-source-loc] universal source_node tagged: "
                << sources << "\n";
  }
  const tensorflow::GraphDef& graph_;
};
}  // namespace

// ---- Function signature convergence ----
// Branch signatures are dtype-only, so args stay unranked and miss odml
// legalize. Converge to the agreeing caller/body type; skip on mismatch.

namespace {
struct ConvergeSignaturesPass
    : public mlir::PassWrapper<ConvergeSignaturesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvergeSignaturesPass)
  llvm::StringRef getArgument() const override {
    return "pb-converge-signatures";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::SymbolTable symtab(module);

    // Call sites: ops carrying a callee symbol (tf.If and its tf.residue
    // wrapper keep then_branch/else_branch; calls carry f).
    llvm::DenseMap<llvm::StringRef, llvm::SmallVector<mlir::Operation*, 4>>
        sites;
    module->walk([&](mlir::Operation* op) {
      if (op->getNumOperands() == 0) return;
      for (mlir::NamedAttribute na : op->getAttrs()) {
        llvm::StringRef an = na.getName();
        if (an != "then_branch" && an != "else_branch" && an != "f") continue;
        if (auto sym = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(na.getValue()))
          sites[sym.getValue()].push_back(op);
      }
    });

    int converged = 0;
    // Nested conditionals need fixpoint iteration: an inner call site's
    // operand is the outer branch function's argument, which only becomes
    // ranked after the outer signature converges in an earlier round.
    for (int round = 0; round < 8; ++round) {
      int roundConverged = 0;
      for (auto& kv : sites) {
        mlir::func::FuncOp fn = symtab.lookup<mlir::func::FuncOp>(kv.first);
        if (!fn || fn.isExternal()) continue;
        bool changed = false;

        // Args: caller operand (i + 1 -- index/predicate occupies slot 0)
        // must agree across sites, be ranked, and match the callee dtype.
        for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
          mlir::Type cur = fn.getArgument(i).getType();
          auto curT = llvm::dyn_cast<mlir::TensorType>(cur);
          if (!curT || curT.hasRank()) continue;
          mlir::Type agreed;
          bool ok = true;
          for (mlir::Operation* c : kv.second) {
            if (i + 1 >= c->getNumOperands()) {
              ok = false;
              break;
            }
            mlir::Type t = c->getOperand(i + 1).getType();
            auto tt = llvm::dyn_cast<mlir::TensorType>(t);
            if (!tt || !tt.hasRank()) {
              ok = false;
              break;
            }
            if (tt.getElementType() != curT.getElementType()) {
              ok = false;
              break;
            }
            if (!agreed) {
              agreed = t;
            } else if (agreed != t) {
              ok = false;
              break;
            }
          }
          if (ok && agreed) {
            fn.getArgument(i).setType(agreed);
            ++converged;
            ++roundConverged;
            changed = true;
          }
        }

        if (changed) {
          auto newTy = mlir::FunctionType::get(
              &getContext(), fn.getArgumentTypes(), fn.getResultTypes());
          fn.setType(newTy);
        }
      }
      if (roundConverged == 0) break;
    }
    if (converged)
      std::cerr << "[converge-signatures] args converged to caller types: "
                << converged << "\n";
  }
};
}  // namespace

// ---- pb shape-annotation retyping ----
// The importer ignores grappler's _output_shapes, so custom ops infect
// chains unranked. Retype from annotations (rank only, never overwrite).

namespace {
struct AnnotateShapesFromPbPass
    : public mlir::PassWrapper<AnnotateShapesFromPbPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotateShapesFromPbPass)
  AnnotateShapesFromPbPass(const tensorflow::GraphDef& graph) : graph_(graph) {}
  llvm::StringRef getArgument() const override { return "pb-annotate-shapes"; }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    PbIndex pb = BuildPbIndex(graph_);

    int retyped = 0;
    llvm::StringMap<int> why;
    module->walk([&](mlir::Operation* op) {
      if (op->getNumResults() == 0) return;
      bool anyUnranked = false;
      for (mlir::Type t : op->getResultTypes())
        if (auto tt = llvm::dyn_cast<mlir::TensorType>(t); tt && !tt.hasRank())
          anyUnranked = true;
      if (!anyUnranked) return;
      llvm::SmallVector<std::string, 4> names;
      CollectLocNamesRec(op->getLoc(), &names);
      if (names.empty()) {
        ++why["no-loc-name"];
        return;
      }
      const tensorflow::NodeDef* node = nullptr;
      for (const std::string& n : names) {
        auto it = pb.find(llvm::StringRef(n));
        if (it != pb.end()) {
          node = it->second;
          break;
        }
      }
      if (node == nullptr) {
        ++why["not-in-pb"];
        return;
      }

      // Proto-map access by iteration only (wheel-protobuf Map::find
      // pitfall).
      const tensorflow::AttrValue* av = nullptr;
      for (const auto& kv : node->attr())
        if (kv.first == "_output_shapes") av = &kv.second;
      if (av == nullptr) {
        ++why["no-annotation"];
        return;
      }
      const auto& shapes = av->list().shape();
      if ((int)shapes.size() != (int)op->getNumResults()) {
        ++why["count-mismatch"];
        return;
      }

      bool changed = false;
      llvm::SmallVector<mlir::Type, 4> newTypes(op->result_type_begin(),
                                                op->result_type_end());
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        auto tt = llvm::dyn_cast<mlir::TensorType>(newTypes[i]);
        if (!tt || tt.hasRank()) continue;
        const auto& shp = shapes[i];
        if (shp.unknown_rank()) continue;
        llvm::SmallVector<int64_t, 6> dims;
        for (const auto& d : shp.dim())
          dims.push_back(d.size() < 0 ? mlir::ShapedType::kDynamic : d.size());
        newTypes[i] = mlir::RankedTensorType::get(dims, tt.getElementType());
        changed = true;
      }
      if (!changed) return;
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        if (newTypes[i] != op->getResult(i).getType())
          op->getResult(i).setType(newTypes[i]);
      ++retyped;
    });
    if (retyped)
      std::cerr << "[pb-annotate-shapes] ops retyped from _output_shapes: "
                << retyped << "\n";
    for (auto& w : why)
      std::cerr << "[annotate-skip] " << w.getKey().str() << ": "
                << w.getValue() << "\n";
  }
  const tensorflow::GraphDef& graph_;
};
}  // namespace

// ---- Shape-vector index normalization ----
// odml's dynamic fallbacks leave index-typed shape operands; the stablehlo
// ODS wants integer ones. Cast to i32 in place (arith.index_cast).

namespace {

// stablehlo ops taking a shape-vector operand (op name -> operand slot).
const std::pair<llvm::StringRef, unsigned> kShapeSlots[] = {
    {"stablehlo.dynamic_broadcast_in_dim", 1},
    {"stablehlo.dynamic_reshape", 1},
    {"stablehlo.dynamic_pad", 1},
    {"stablehlo.dynamic_iota", 0},
    {"stablehlo.real_dynamic_slice", 2},
    {"stablehlo.dynamic_update_slice", 2},
};

// odml's compare pattern skips broadcasting on mixed shapes, which the
// ODS rejects. Restore the TF comparison as a residue carrier (operand
// order unchanged). Returns 0 fine / 1 restored / 2 unrepresentable.
int RestoreCompareResidue(mlir::Operation* op,
                          llvm::SmallVectorImpl<mlir::Operation*>& toErase) {
  if (op->getNumResults() != 1) return 0;  // rewrite below touches result 0
  llvm::SmallVector<mlir::ShapedType, 4> shapes;
  bool hasUnranked = false;
  for (mlir::Value v : op->getOperands()) {
    auto t = llvm::dyn_cast<mlir::ShapedType>(v.getType());
    if (!t) continue;
    if (!t.hasRank()) hasUnranked = true;
    shapes.push_back(t);
  }
  if (hasUnranked || shapes.size() < 2) return 0;
  bool mismatch = false;
  for (size_t i = 1; i < shapes.size(); ++i)
    if (shapes[i].getShape() != shapes[0].getShape()) mismatch = true;
  for (mlir::Value v : op->getResults()) {
    auto t = llvm::dyn_cast<mlir::ShapedType>(v.getType());
    if (t && t.hasRank() && t.getShape() != shapes[0].getShape())
      mismatch = true;
  }
  if (!mismatch) return 0;
  std::string tag = SrcOpTag(op);
  if (tag.empty()) return 2;
  mlir::OpBuilder b(op);
  auto carrier = BuildResidueCarrier(b, op->getLoc(), op->getOperands(),
                                     op->getResultTypes(), "tf." + tag,
                                     "unranked");
  op->getResult(0).replaceAllUsesWith(carrier->getResult(0));
  toErase.push_back(op);
  return 1;
}

struct NormalizeShapeVectorsPass
    : public mlir::PassWrapper<NormalizeShapeVectorsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeShapeVectorsPass)
  llvm::StringRef getArgument() const override {
    return "pb-normalize-shape-vectors";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    int fixed = 0;
    llvm::SmallVector<mlir::Operation*, 4> toErase;
    module->walk([&](mlir::Operation* op) {
      llvm::StringRef nm = op->getName().getStringRef();
      if (nm == "stablehlo.compare") {
        int r = RestoreCompareResidue(op, toErase);
        if (r == 1) ++fixed;
        return;
      }
      for (const auto& slot : kShapeSlots) {
        if (nm != slot.first || op->getNumOperands() <= slot.second) continue;
        mlir::Value v = op->getOperand(slot.second);
        auto t = llvm::dyn_cast<mlir::RankedTensorType>(v.getType());
        if (t && t.getElementType().isIndex()) {
          mlir::OpBuilder b(op);
          auto i32Ty =
              mlir::RankedTensorType::get(t.getShape(), b.getI32Type());
          auto cast =
              b.create<mlir::arith::IndexCastOp>(op->getLoc(), i32Ty, v);
          op->setOperand(slot.second, cast.getResult());
          v = cast.getResult();
        }
        auto vt = llvm::dyn_cast<mlir::RankedTensorType>(v.getType());
        if (vt && vt.hasStaticShape() && vt.getRank() == 1) {
          ++fixed;
          continue;
        }
        // ODS requires a statically shaped 1-D shape vector; odml's string
        // fallbacks can leave a dynamically-sized one. Dead ones just go
        // away.
        if (op->getNumResults() == 1 && op->getResult(0).use_empty()) {
          toErase.push_back(op);
          ++fixed;
          continue;
        }
        // Live but unrepresentable: odml's dynamic fallback for tf.Fill.
        // Restore it as a tf.residue carrier with operands reordered back
        // to Fill(dims, value); the pb node exists, so back-write works.
        std::string srcTag = SrcOpTag(op);
        if (srcTag == "Fill" && op->getNumOperands() == 2 &&
            op->getNumResults() == 1) {
          mlir::OpBuilder b(op);
          auto carrier = BuildResidueCarrier(
              b, op->getLoc(), {op->getOperand(1), op->getOperand(0)},
              op->getResult(0).getType(), "tf.Fill", "unranked");
          op->getResult(0).replaceAllUsesWith(carrier->getResult(0));
          toErase.push_back(op);
          ++fixed;
          continue;
        }
      }
    });
    // Erase after the walk (erasing during walk invalidates iterators).
    for (mlir::Operation* op : toErase) op->erase();
    if (fixed)
      std::cerr << "[normalize-shape-vectors] index shape operands cast to "
                   "i32/erased: "
                << fixed << "\n";
  }
};
}  // namespace

// ---- Post-legalize residue wrap (was pb2_emit.cpp RunPipelines) -----------

// Post-legalize unification: remaining registered tf.* ops wrapped as
// tf.residue. Reshape/StridedSlice are static-gated; all others report
// "no_legalize_pattern" (probe-calibrated 2026-11-08).
static bool isStaticGatedPattern(llvm::StringRef cls) {
  static const llvm::SmallDenseSet<llvm::StringRef, 4> kStaticGated = {
      "Reshape", "StridedSlice"};
  return kStaticGated.contains(cls);
}

static bool hasNonStaticShape(mlir::Operation* op) {
  for (mlir::Type t : op->getOperandTypes()) {
    auto tt = llvm::dyn_cast<mlir::ShapedType>(t);
    if (tt && !tt.hasStaticShape()) return true;
  }
  for (mlir::Type t : op->getResultTypes()) {
    auto tt = llvm::dyn_cast<mlir::ShapedType>(t);
    if (tt && !tt.hasStaticShape()) return true;
  }
  return false;
}

void WrapRegisteredResidueImpl(mlir::ModuleOp module,
                               mlir::MLIRContext& context) {
  std::map<std::string, std::pair<int, std::string>> reg_res;
  llvm::SmallVector<mlir::Operation*> wrap_targets;
  module.walk([&](mlir::Operation* op) {
    llvm::StringRef nm = op->getName().getStringRef();
    if (!nm.starts_with("tf.") || nm == "tf.residue") return;
    std::string cls(nm.drop_front(3));
    if (!g_opspecs.count(cls)) return;
    wrap_targets.push_back(op);
    auto& e = reg_res[cls];
    if (++e.first == 1) {
      llvm::raw_string_ostream os(e.second);
      op->getLoc().print(os);
    }
  });
  mlir::OpBuilder b(&context);
  for (auto it = wrap_targets.rbegin(); it != wrap_targets.rend(); ++it) {
    mlir::Operation* op = *it;
    mlir::OperationState st(op->getLoc(),
                            mlir::OperationName("tf.residue", &context));
    st.operands.append(op->operand_begin(), op->operand_end());
    st.addTypes(op->getResultTypes());
    st.addAttribute("call_target_name",
                    b.getStringAttr(op->getName().getStringRef()));
    std::string cls(op->getName().getStringRef().drop_front(3));
    st.addAttribute(
        "residue_reason",
        b.getStringAttr(isStaticGatedPattern(cls) && hasNonStaticShape(op)
                            ? "pattern_requires_static_shape"
                            : "no_legalize_pattern"));
    CopyNonReservedAttrs(st, op);
    b.setInsertionPoint(op);
    mlir::Operation* res = b.create(st);
    op->replaceAllUsesWith(res);
    op->erase();
  }
  if (!reg_res.empty()) {
    int total = 0;
    for (auto& kv : reg_res) total += kv.second.first;
    std::cerr << "[residue] " << reg_res.size()
              << " registered op classes wrapped as tf.residue (" << total
              << " instances):\n";
    for (auto& kv : reg_res)
      std::cerr << "  tf." << kv.first << " x" << kv.second.first
                << "  first: " << kv.second.second << "\n";
  }
}

struct WrapRegisteredResiduePass
    : public mlir::PassWrapper<WrapRegisteredResiduePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WrapRegisteredResiduePass)
  llvm::StringRef getArgument() const override {
    return "pb-wrap-registered-residue";
  }
  void runOnOperation() override {
    WrapRegisteredResidueImpl(getOperation(), *getOperation().getContext());
  }
};

// ---- Post-legalize carrier rename (stablehlo conversion is complete) ----

// tf.residue -> annc.residue once the wheel is done: the tf. prefix is
// load-bearing while the wheel lives (its unknown-op update path faults on
// string carriers). The sink matches by call_target_name (F3), not name.
void RenameResidueCarrierImpl(mlir::ModuleOp module,
                              mlir::MLIRContext& context) {
  llvm::SmallVector<mlir::Operation*> targets;
  module.walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "tf.residue") targets.push_back(op);
  });
  mlir::OpBuilder b(&context);
  for (auto it = targets.rbegin(); it != targets.rend(); ++it) {
    mlir::Operation* op = *it;
    mlir::OperationState st(op->getLoc(),
                            mlir::OperationName("annc.residue", &context));
    st.operands.append(op->operand_begin(), op->operand_end());
    st.addTypes(op->getResultTypes());
    for (const auto& attr : op->getAttrs())
      st.addAttribute(attr.getName(), attr.getValue());
    b.setInsertionPoint(op);
    mlir::Operation* res = b.create(st);
    op->replaceAllUsesWith(res);
    op->erase();
  }
  if (!targets.empty())
    std::cerr << "[residue] carrier renamed: tf.residue -> annc.residue ("
              << targets.size() << " instances)\n";
}

struct RenameResidueCarrierPass
    : public mlir::PassWrapper<RenameResidueCarrierPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RenameResidueCarrierPass)
  llvm::StringRef getArgument() const override {
    return "pb-rename-residue-carrier";
  }
  void runOnOperation() override {
    RenameResidueCarrierImpl(getOperation(), *getOperation().getContext());
  }
};

// ---- Post-legalize port tagging (consumption graph is final) ----

// Tags annc.source_ports (TF output port per result) via a three-part
// evidence chain, all required, else untagged (a wrong tag fuses wrongly):
// cross-group use edge -> consumer source_node -> "<node>[:k]" pb spec.
void PortTagImpl(mlir::ModuleOp module, mlir::MLIRContext& context,
                 const llvm::DenseMap<llvm::StringRef,
                                      const tensorflow::NodeDef*>& pb) {
  // Escape edges keyed by the producing op.
  llvm::DenseMap<mlir::Operation*,
                 llvm::SmallVector<std::pair<mlir::Operation*, int64_t>>>
      escapes;
  module.walk([&](mlir::Operation* op) {
    auto src = op->getAttrOfType<mlir::StringAttr>("annc.source_node");
    if (!src) return;
    for (mlir::OpResult r : op->getResults()) {
      for (mlir::OpOperand& use : r.getUses()) {
        auto ownerSrc =
            use.getOwner()->getAttrOfType<mlir::StringAttr>("annc.source_node");
        if (ownerSrc && ownerSrc.getValue() == src.getValue()) continue;  // in-group
        escapes[op].push_back({use.getOwner(), r.getResultNumber()});
      }
    }
  });
  int tagged = 0, ambiguous = 0, unresolved = 0;
  mlir::OpBuilder b(&context);
  for (auto& [producer, uses] : escapes) {
    auto src = producer->getAttrOfType<mlir::StringAttr>("annc.source_node");
    if (!src) continue;
    llvm::StringRef node = src.getValue();
    // Per-RESULT resolution: one group's results map to distinct TF ports
    // by design, so evidence coexists across results; only disagreement
    // about the SAME result is a conflict.
    llvm::DenseMap<unsigned, int64_t> resultPort;
    bool conflict = false;
    for (auto& [owner, resNo] : uses) {
      auto ownerSrc =
          owner->getAttrOfType<mlir::StringAttr>("annc.source_node");
      if (!ownerSrc) { ++unresolved; continue; }  // e.g. func.return: no pb consumer
      auto dn = pb.lookup(ownerSrc.getValue());
      if (dn == nullptr) { ++unresolved; continue; }
      // Name-based port lookup in the consumer's pb input spec.
      llvm::SmallVector<int64_t, 2> ports;
      for (const auto& input : dn->input()) {
        llvm::StringRef ref(input);
        if (ref.starts_with("^")) continue;       // control dependency
        llvm::StringRef suffix = ref;
        if (!suffix.consume_front(node)) continue;  // references another node
        int64_t p = 0;
        if (suffix.empty()) {                     // bare name = port 0
          ports.push_back(p);
          continue;
        }
        if (suffix[0] != ':') continue;  // e.g. "<node>Foo" — another node
        if (suffix.substr(1).getAsInteger(10, p)) continue;
        ports.push_back(p);
      }
      if (ports.empty()) { ++unresolved; continue; }
      bool same = llvm::all_of(ports, [&](int64_t p) { return p == ports[0]; });
      if (!same) { conflict = true; break; }
      auto it = resultPort.find(resNo);
      if (it == resultPort.end()) resultPort[resNo] = ports[0];
      else if (it->second != ports[0]) { conflict = true; break; }
    }
    if (conflict || resultPort.empty()) {
      if (conflict) ++ambiguous;
      continue;
    }
    unsigned maxRes = 0;
    for (auto& [resNo, port] : resultPort) maxRes = std::max(maxRes, resNo);
    llvm::SmallVector<int32_t, 4> portVec(maxRes + 1, -1);  // -1 = no evidence
    for (auto& [resNo, port] : resultPort)
      portVec[resNo] = static_cast<int32_t>(port);
    producer->setAttr("annc.source_ports",
                      b.getI32TensorAttr(portVec));
    ++tagged;
  }
  if (tagged || ambiguous || unresolved)
    std::cerr << "[source-port] tagged: " << tagged << " ambiguous: "
              << ambiguous << " unresolved: " << unresolved << "\n";
}

struct PortTagPass
    : public mlir::PassWrapper<PortTagPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PortTagPass)
  PortTagPass(const tensorflow::GraphDef& graph) : graph_(graph) {}
  llvm::StringRef getArgument() const override { return "pb-source-port-tag"; }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    PortTagImpl(module, *module.getContext(), BuildPbIndex(graph_));
  }
  const tensorflow::GraphDef& graph_;
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateRenameResidueCarrierPass() {
  return std::make_unique<RenameResidueCarrierPass>();
}
std::unique_ptr<mlir::Pass> CreatePortTagPass(
    const tensorflow::GraphDef& graph) {
  return std::make_unique<PortTagPass>(graph);
}
std::unique_ptr<mlir::Pass> CreateNormalizeRefTypesPass() {
  return std::make_unique<NormalizeRefTypesPass>();
}
std::unique_ptr<mlir::Pass> CreateFlattenExecutorPass() {
  return std::make_unique<FlattenExecutorPass>();
}
std::unique_ptr<mlir::Pass> CreateStripFunctionalizeSuffixPass() {
  return std::make_unique<StripFunctionalizeSuffixPass>();
}
std::unique_ptr<mlir::Pass> CreateAnchorConstantsToUsersPass() {
  return std::make_unique<AnchorConstantsToUsersPass>();
}
std::unique_ptr<mlir::Pass> CreateTagConstSourceNamePass() {
  return std::make_unique<TagConstSourceNamePass>();
}

std::unique_ptr<mlir::Pass> CreateRestoreConstSourceLocPass(
    const tensorflow::GraphDef& graph) {
  return std::make_unique<RestoreConstSourceLocPass>(graph);
}

std::unique_ptr<mlir::Pass> CreateAnnotateShapesFromPbPass(
    const tensorflow::GraphDef& graph) {
  return std::make_unique<AnnotateShapesFromPbPass>(graph);
}

std::unique_ptr<mlir::Pass> CreateConvergeSignaturesPass() {
  return std::make_unique<ConvergeSignaturesPass>();
}

std::unique_ptr<mlir::Pass> CreateNormalizeShapeVectorsPass() {
  return std::make_unique<NormalizeShapeVectorsPass>();
}

std::unique_ptr<mlir::Pass> CreateWrapRegisteredResiduePass() {
  return std::make_unique<WrapRegisteredResiduePass>();
}

}  // namespace pb_passes
