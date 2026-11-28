#ifndef ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_NAMING_H
#define ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_NAMING_H

#include <string>

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace atir {
namespace fusionv2 {

// Replaces non-alphanumerics with '_' for use in kernel/symbol names.
inline std::string sanitizeName(std::string name) {
  for (char &c : name)
    if (!isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
  return name;
}

// Kernel-name scheme shared with V1: 4-byte hex of a SHA256 over the anchor
// name (and the pattern name for epilogue sites) - stable across runs and
// distinct per site.
inline std::string hash4(llvm::StringRef a, llvm::StringRef b = {}) {
  llvm::SHA256 sha;
  sha.update(a);
  sha.update(b);
  auto hash = sha.final();
  std::string hex;
  for (size_t i = 0; i < 4; ++i) hex += llvm::toHex(hash[i]);
  return hex;
}

// Appends _N until the symbol is unique in the module.
inline std::string uniquifySymbolName(mlir::ModuleOp module,
                                      llvm::StringRef baseName) {
  std::string unique = baseName.str();
  unsigned suffix = 1;
  while (module.lookupSymbol(unique))
    unique = (llvm::Twine(baseName) + "_" + llvm::Twine(suffix++)).str();
  return unique;
}

// fusion.metadata tf.name uniquification (the converter names the GraphDef
// fusion node after it); identical to the V1 helper. A module walk per probe
// is required because tf.name is metadata, not an IR symbol: lookupSymbol
// cannot see it.
inline std::string uniquifyFusionName(mlir::ModuleOp module,
                                      llvm::StringRef baseName) {
  std::string unique = baseName.str();
  unsigned suffix = 1;
  bool changed = true;
  while (changed) {
    changed = false;
    module->walk([&](mlir::func::FuncOp fn) {
      if (changed) return;
      auto metadata =
          fn->getAttrOfType<mlir::DictionaryAttr>("fusion.metadata");
      if (!metadata) return;
      auto name =
          llvm::dyn_cast_or_null<mlir::StringAttr>(metadata.get("tf.name"));
      if (name && name.getValue() == unique) changed = true;
    });
    if (changed)
      unique = (llvm::Twine(baseName) + "_" + llvm::Twine(suffix++)).str();
  }
  return unique;
}

}  // namespace fusionv2
}  // namespace atir

#endif  // ANNC_INCLUDE_DIALECT_ATIR_PASSES_FUSIONV2_NAMING_H
