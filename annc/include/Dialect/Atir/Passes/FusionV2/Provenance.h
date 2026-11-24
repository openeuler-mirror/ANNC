#ifndef ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_PROVENANCE_H
#define ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_PROVENANCE_H

#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

namespace atir {
namespace fusionv2 {

// Pb provenance record: source node name + output port (TF value identity
// "node:port").
struct ProvRecord {
  std::string node;
  int port = 0;

  // metadata tf_name spelling (converter wires by node, slots by port).
  std::string str() const { return node + ":" + std::to_string(port); }
};

// "node:port" parse: split the port suffix from the right (TF node names
// contain no ':'; the port must be all digits); a missing suffix means 0.
// Empty node or non-numeric port returns false.
inline bool parseProv(llvm::StringRef s, ProvRecord &out) {
  if (s.empty()) return false;
  size_t colon = s.rfind(':');
  if (colon == llvm::StringRef::npos) {
    out.node = s.str();
    out.port = 0;
    return true;
  }
  llvm::StringRef tail = s.substr(colon + 1);
  // The port must be all digits: getAsInteger alone would accept "-1".
  if (tail.empty() ||
      !llvm::all_of(tail, [](char c) { return c >= '0' && c <= '9'; }))
    return false;
  int port = 0;
  // Overflow (digits beyond INT32) must fail too: a silent 0 would rewire to
  // a real output slot.
  if (tail.getAsInteger(10, port)) return false;
  out.node = s.substr(0, colon).str();
  if (out.node.empty()) return false;  // ":3" carries no node name
  out.port = port;
  return true;
}

// Consumer provenance: item idx of the consumer's annc.prov array (one
// "node:port" per operand, fixed by the FE); missing/out-of-range/invalid
// entries all return false.
inline bool operandProvOf(mlir::Operation *consumer, unsigned idx,
                          ProvRecord &out) {
  if (!consumer) return false;
  auto arr = consumer->getAttrOfType<mlir::ArrayAttr>("annc.prov");
  if (!arr || idx >= arr.size()) return false;
  auto item = llvm::dyn_cast_or_null<mlir::StringAttr>(arr[idx]);
  if (!item) return false;
  return parseProv(item.getValue(), out);
}

// Producer provenance: source node via annc.source_node (fold/CSE-proof);
// the port prefers annc.source_ports[resultNo] (-1/missing slot bails out);
// a missing tag keeps the IR result number (residue forms are natural ports).
// A source_node tag may already carry a port suffix ("node:1"); the port
// must be appended exactly once or the metadata endpoint becomes
// "node:1:1" and downstream validators reject it.
inline std::string stripPortSuffix(llvm::StringRef name) {
  size_t colon = name.rfind(':');
  if (colon == llvm::StringRef::npos || colon + 1 >= name.size())
    return name.str();
  for (char c : name.drop_front(colon + 1))
    if (c < '0' || c > '9') return name.str();
  return name.take_front(colon).str();
}

inline std::string sourceNameOf(mlir::Value v) {
  auto res = mlir::dyn_cast<mlir::OpResult>(v);
  mlir::Operation *def = res ? res.getOwner() : nullptr;
  auto sn = def ? def->getAttrOfType<mlir::StringAttr>("annc.source_node")
                : nullptr;
  if (!sn || sn.getValue().empty()) return "";
  int64_t port = static_cast<int64_t>(res.getResultNumber());
  if (auto ports = def->getAttrOfType<mlir::ArrayAttr>("annc.source_ports")) {
    if (res.getResultNumber() >= ports.size()) return "";
    auto item = llvm::dyn_cast_or_null<mlir::IntegerAttr>(
        ports[res.getResultNumber()]);
    if (!item || item.getInt() < 0) return "";
    port = item.getInt();
  }
  return stripPortSuffix(sn.getValue()) + ":" + std::to_string(port);
}

}  // namespace fusionv2
}  // namespace atir

#endif  // ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_PROVENANCE_H
