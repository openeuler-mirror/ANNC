#ifndef ANNC_FUSION_METADATA_FUSION_METADATA_H
#define ANNC_FUSION_METADATA_FUSION_METADATA_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "llvm/Support/Error.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace annc::fusion {

// Describes one logical tensor at a fusion boundary. FusionInfo::args and
// FusionInfo::outputs are the canonical source for tensor names, roles, shapes,
// ranks, and data types; consumers must derive runtime mirrors from them.
struct FusionArg {
  // Boundary role: "constant", "fixed", or "dynamic" for an input, and
  // "output" for an output.
  std::string role;

  // TensorFlow tensor name used to reconnect the fused node to the GraphDef.
  // A data input may include an output suffix such as "node:1".
  std::string tfName;

  // Compile-time ATIR shape. Dynamic dimensions use MLIR's dynamic sentinel
  // until the metadata format normalizes them at a serialization boundary.
  std::vector<int64_t> shape;

  // Tensor rank, stored explicitly so malformed metadata can be rejected when
  // rank does not match shape.size().
  int64_t rank = -1;

  // Canonical ATIR dtype spelling, for example "f32", "i64", "si64",
  // "ui64", "index", or "string". Encoded TensorFlow types such as string
  // take their dtype from the ATIR tensor encoding rather than from the
  // underlying complex storage type.
  std::string dtype;
};

// Canonical metadata contract for one generated fusion kernel. This structure
// intentionally excludes derived TensorFlow runtime attributes such as input
// counts, input/output ranks, and output shape strings.
struct FusionInfo {
  // Name of the ANNCFused node that will be added to the TensorFlow GraphDef.
  std::string name;

  // Semantic fusion label, for example "matmul_add_relu". Lowering must still
  // validate the function body and ABI before selecting a specialized kernel.
  std::string pattern;

  // Exported kernel symbol. The runtime resolves _mlir_ciface_<kernelName>
  // from the generated shared library for the mlir_ciface ABI.
  std::string kernelName;

  // Ordered external inputs. This order becomes the ANNCFused input order and
  // is also the base index space used by kernelArgOrder.
  std::vector<FusionArg> args;

  // Ordered logical outputs. Their TensorFlow names identify the original
  // graph edges replaced by the generated ANNCFused node.
  std::vector<FusionArg> outputs;

  // Pattern-specific, non-generic values such as embedding num_buckets.
  std::map<std::string, std::string> patternAttrs;

  // Runtime calling convention. Only "mlir_ciface" is currently supported.
  std::string abi = "mlir_ciface";

  // Output-axis indices whose sizes are resolved from the dynamic input at
  // runtime. For example {0} marks the batch dimension as dynamic.
  std::vector<int64_t> dynamicDims;

  // Permutation applied before the C-interface call. Indices address the
  // concatenated memref list [args..., outputs...].
  std::vector<int64_t> kernelArgOrder;

  // Reserved symbolic-shape description. It is currently persisted and copied
  // to ANNCFused for compatibility but is not parsed or consumed by runtime.
  std::string symbolicSignature;

  // Reserved fallback function name. ANNCFused currently does not execute a
  // fallback function; the value is retained for GraphDef compatibility.
  std::string fallbackFunction;
};

llvm::Error validateFusionInfo(const FusionInfo &info);

llvm::Expected<FusionInfo> readFusionInfo(mlir::func::FuncOp func);

llvm::Expected<std::vector<FusionInfo>> extractFusionInfos(
    mlir::ModuleOp module);

llvm::Expected<std::vector<FusionInfo>> extractFusionInfosFromMlirText(
    const std::string &path);

}  // namespace annc::fusion

#endif  // ANNC_FUSION_METADATA_FUSION_METADATA_H
