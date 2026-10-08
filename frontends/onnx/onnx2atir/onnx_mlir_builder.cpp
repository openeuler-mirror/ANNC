#include "onnx_mlir_builder.h"

#include "Builder/Transformers.h"
#include "llvm/ADT/StringMap.h"
#include "onnx_transformers.h"

namespace annc::onnx2atir {

// Defines populateOnnxOpSpecTable(); must precede the storage below. Like the
// TF table it is included from exactly one translation unit.
#include "onnx_op_spec_table.inc"

namespace {

// Storage for the ONNX table, mirroring the TF table's storage in
// MLIROpBuilder.cpp: the name map points into a stable vector so that OpSpec
// addresses stay valid for the process lifetime.
const llvm::SmallVector<OpSpec, 0>& onnxOpSpecStorage() {
  static const llvm::SmallVector<OpSpec, 0> kSpecs = [] {
    llvm::SmallVector<OpSpec, 0> specs;
    populateOnnxOpSpecTable(specs);
    return specs;
  }();
  return kSpecs;
}

const llvm::StringMap<const OpSpec*>& onnxOpSpecTable() {
  static llvm::StringMap<const OpSpec*> table = [] {
    llvm::StringMap<const OpSpec*> map;
    for (const OpSpec& spec : onnxOpSpecStorage())
      for (llvm::StringRef op : spec.tfOps) map[op] = &spec;
    return map;
  }();
  return table;
}

}  // namespace

const OpSpec* lookupOnnxOpSpec(llvm::StringRef opType) {
  const auto& table = onnxOpSpecTable();
  auto it = table.find(opType);
  return it == table.end() ? nullptr : it->second;
}

const OpSpec* OnnxMLIRBuilder::lookupSpec(llvm::StringRef opType) const {
  return lookupOnnxOpSpec(opType);
}

bool OnnxMLIRBuilder::shouldLowerToOpaque(const NodeInfo& node) const {
  int64_t marker = 0;
  return node.getAttr(kFallbackAttr.str(), marker);
}

}  // namespace annc::onnx2atir
