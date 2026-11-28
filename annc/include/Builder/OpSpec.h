#ifndef ANNC_OPSPEC_H
#define ANNC_OPSPEC_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace annc {

struct NodeInfo;

enum class AttrKind {
  Direct,          // same-named attribute; value converted by its MLIR type
  Transform,       // converted via an explicit function
  DocumentedDrop,  // TF-only attribute: kept in metadata, never applied
};

enum class OpKind {
  Generic,   // uniform pattern (or a transformer when transformer != nullptr)
  Constant,  // Const/Constant: raw tensor bytes decoded by buildConstantNode
};

using AttrTransform = mlir::Attribute (*)(mlir::OpBuilder&, const NodeInfo&,
                                          llvm::StringRef atirAttr);

struct AttrMapping {
  llvm::StringRef tfAttr;
  llvm::StringRef atirAttr;
  AttrKind kind = AttrKind::Direct;
  AttrTransform transform = nullptr;
};

struct AttrDefault {
  llvm::StringRef name;
  mlir::Attribute (*build)(mlir::OpBuilder&);
};

class OpContext;
using Transformer = mlir::LogicalResult (*)(const NodeInfo&,
                                            llvm::ArrayRef<mlir::Type>,
                                            llvm::ArrayRef<mlir::Value>,
                                            OpContext&);

inline constexpr unsigned kUnbounded = ~0u;
inline constexpr unsigned kVariableOutputs = ~0u;

// One declarative row: how a TF op maps to an ATIR op. 1:1 ops (output type +
// optional BufferOp + variadic inputs + defaulted attrs) need no transformer.
struct OpSpec {
  // Owns alias names (StringRefs point at string literals; a plain ArrayRef
  // would dangle on a temporary initializer_list).
  llvm::SmallVector<llvm::StringRef, 4> tfOps;  // TF op names (aliases)
  llvm::StringRef atirOp;                       // target ATIR op name
  unsigned minInputs = 0;
  unsigned maxInputs = kUnbounded;
  unsigned numOutputs = 1;  // kVariableOutputs = variable
  bool needsBuffer = true;  // emit a BufferOp as first operand
  llvm::ArrayRef<AttrDefault>
      attrDefaults;  // defaults for generic construction
  llvm::ArrayRef<AttrMapping> attrMappings;  // TF attr -> ATIR attr translation
  Transformer transformer = nullptr;         // non-null for non-1:1 ops
  OpKind kind = OpKind::Generic;
};

// C++17-safe factory for table rows (avoids designated initializers).
inline OpSpec makeSpec(std::initializer_list<llvm::StringRef> tfOps,
                       llvm::StringRef atirOp, unsigned minInputs,
                       unsigned maxInputs, unsigned numOutputs,
                       bool needsBuffer,
                       llvm::ArrayRef<AttrDefault> attrDefaults = {},
                       llvm::ArrayRef<AttrMapping> attrMappings = {},
                       Transformer transformer = nullptr,
                       OpKind kind = OpKind::Generic) {
  return OpSpec{tfOps,       atirOp,       minInputs,    maxInputs,  numOutputs,
                needsBuffer, attrDefaults, attrMappings, transformer, kind};
}

// Sandbox handed to transformers: op construction, result binding and typed
// TF attribute lookup, so transformers never touch framework state.
class OpContext {
 public:
  OpContext(mlir::OpBuilder& builder,
            std::unordered_map<std::string, mlir::Value>& tensorValues,
            const std::unordered_map<std::string, const NodeInfo*>& nodesByName)
      : builder_(builder),
        tensorValues_(tensorValues),
        nodesByName_(nodesByName) {}

  mlir::OpBuilder& builder() { return builder_; }
  mlir::Location loc(llvm::StringRef name) const;
  // Typed TF attr lookup on the current node; true only for an exact type T
  // match. Definition lives in MLIROpBuilder.h (needs the full NodeInfo).
  template <typename T>
  bool getAttr(const NodeInfo& node, llvm::StringRef name, T& out) const;
  // Resolves a graph node (typically a constant) by name.
  const NodeInfo* findNode(llvm::StringRef name) const {
    auto it = nodesByName_.find(name.str());
    return it == nodesByName_.end() ? nullptr : it->second;
  }
  // Registers a produced value under its output name.
  void bindResult(llvm::StringRef name, mlir::Value value) {
    tensorValues_[name.str()] = value;
  }
  mlir::Value getValue(llvm::StringRef name) const {
    auto it = tensorValues_.find(name.str());
    return it == tensorValues_.end() ? mlir::Value() : it->second;
  }
  // Prints a diagnostic for the current build and returns failure().
  mlir::LogicalResult emitError(const NodeInfo& node, llvm::StringRef message);

 private:
  mlir::OpBuilder& builder_;
  std::unordered_map<std::string, mlir::Value>& tensorValues_;
  const std::unordered_map<std::string, const NodeInfo*>& nodesByName_;
};

}  // namespace annc

#endif  // ANNC_OPSPEC_H
