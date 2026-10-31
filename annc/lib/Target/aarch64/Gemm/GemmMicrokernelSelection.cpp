#include "GemmMicrokernelSelection.h"

#include <limits>

#include "GemmPlan.h"
#include "llvm/ADT/Twine.h"

namespace annc::aarch64::gemm {
namespace {
using namespace mlir;

FailureOr<int64_t> getDimension(func::CallOp call, StringRef name) {
  auto value = call->getAttrOfType<IntegerAttr>(name);
  if (!value || value.getInt() <= 0 ||
      value.getInt() > std::numeric_limits<uint32_t>::max()) {
    call.emitOpError() << "requires positive static attribute " << name
                       << " within the codegen ABI range";
    return failure();
  }
  return value.getInt();
}

}  // namespace

FailureOr<FinalMicrokernelSelection> selectFinalMicrokernel(func::CallOp call) {
  FailureOr<int64_t> m = getDimension(call, kMicrokernelMAttrName);
  FailureOr<int64_t> n = getDimension(call, kMicrokernelNAttrName);
  FailureOr<int64_t> k = getDimension(call, kMicrokernelKAttrName);
  if (failed(m) || failed(n) || failed(k)) return failure();
  FailureOr<GemmPlan> plan = readPlan(call);
  if (failed(plan)) return failure();
  if (plan->vectorLengthBytes <= 0 ||
      plan->vectorLengthBytes > std::numeric_limits<uint32_t>::max())
    return call.emitOpError("has an invalid codegen vector length");
  const bool rowMajor =
      usesLdbAbi(getGemmLeafKind(plan->executionKind, plan->rhsPacking));
  if (rowMajor != isRowMajorMicrokernelLeaf(call.getCallee()))
    return call.emitOpError("microkernel leaf does not match the GEMM plan");
  FailureOr<int64_t> nr = getGemmNr(plan->kernelTile, plan->vectorLengthBytes,
                                    plan->dataType, plan->executionKind);
  if (failed(nr) || *m > plan->kernelTile.mr || *n > *nr)
    return call.emitOpError("has a microtile outside the static kernel family");
  auto mode = call->getAttrOfType<StringAttr>(kKcModeAttrName);
  if (!mode ||
      (mode.getValue() != "overwrite" && mode.getValue() != "accumulate"))
    return call.emitOpError("requires a valid static KC mode");

  FinalMicrokernelSelection result;
  codegen::KernelDesc &kernel = result.codegenKernel;
  kernel.m = *m;
  kernel.n = *n;
  kernel.k = *k;
  kernel.vectorLengthBytes = plan->vectorLengthBytes;
  kernel.accumulate = mode.getValue() == "accumulate";
  auto role = call->getAttrOfType<StringAttr>(kKcRoleAttrName);
  if (role && role.getValue() != "single" && role.getValue() != "final" &&
      role.getValue() != "first" && role.getValue() != "middle")
    return call.emitOpError("has an invalid KC role");

  StringRef prefix =
      rowMajor ? "annc_aarch64_sve_kernel_rm_mr" : "annc_aarch64_sve_kernel_mr";
  if (plan->isa == GemmIsa::kSve) {
    const int64_t lanes = plan->vectorLengthBytes / sizeof(float);
    const int64_t groups = (*n + lanes - 1) / lanes;
    if (groups < 1 || groups > plan->kernelTile.panelLanes)
      return call.emitOpError("has an invalid SVE N-vector group count");
    result.symbol =
        (llvm::Twine(prefix) + llvm::Twine(*m) + "_n" + llvm::Twine(groups) +
         "vl" + (kernel.accumulate ? "_acc_f32" : "_f32"))
            .str();
    kernel.family =
        rowMajor ? codegen::Family::SveDirect : codegen::Family::SvePacked;
    return result;
  }

  const GemmKernelABI &abi = getGemmKernelABI(
      plan->target, plan->isa, plan->dataType, plan->executionKind);
  FailureOr<int64_t> unroll = getGemmKScalarUnroll(abi, plan->dataType);
  if (failed(unroll))
    return call.emitOpError("requires a valid NEON kernel ABI vector K unroll");
  const int64_t groups = *k / *unroll;
  const int64_t residue = *k % *unroll;
  const std::string kVariant =
      groups == 0 ? (llvm::Twine("k") + llvm::Twine(residue)).str()
                  : (llvm::Twine("kg_r") + llvm::Twine(residue)).str();
  if (plan->executionKind == GemmExecutionKind::kMatrixVector) {
    if (*n != 1 || *m > 4 || abi.kVectorUnroll != 4)
      return call.emitOpError("has an invalid matrix-vector microtile");
    result.symbol =
        (llvm::Twine("annc_aarch64_neon_matvec_mr") + llvm::Twine(*m) + "_" +
         kVariant + (kernel.accumulate ? "_acc_f32" : "_f32"))
            .str();
    kernel.family = codegen::Family::NeonMatvec;
    return result;
  }
  if (plan->executionKind == GemmExecutionKind::kVectorMatrix) {
    if (!rowMajor || *m != 1 || *n > 16 || abi.kVectorUnroll != 2)
      return call.emitOpError("has an invalid vector-matrix microtile");
    result.symbol =
        (llvm::Twine("annc_aarch64_neon_vecmat_n") + llvm::Twine(*n) + "_" +
         kVariant + (kernel.accumulate ? "_acc_f32" : "_f32"))
            .str();
    result.supportsEpilogueCodegen = false;
    return result;
  }
  prefix = rowMajor ? "annc_aarch64_neon_kernel_rm_mr"
                    : "annc_aarch64_neon_kernel_mr";
  result.symbol =
      (llvm::Twine(prefix) + llvm::Twine(*m) + "_n" + llvm::Twine(*n) + "_" +
       kVariant + (kernel.accumulate ? "_acc_f32" : "_f32"))
          .str();
  kernel.family =
      rowMajor ? codegen::Family::NeonDirect : codegen::Family::NeonPacked;
  return result;
}

LogicalResult attachFinalMicrokernelSelection(
    func::CallOp call, const FinalMicrokernelSelection &selection) {
  Builder builder(call.getContext());
  const codegen::KernelDesc &kernel = selection.codegenKernel;
  NamedAttrList attrs;
  attrs.append("version", builder.getI32IntegerAttr(1));
  attrs.append("symbol", builder.getStringAttr(selection.symbol));
  attrs.append("family",
               builder.getI32IntegerAttr(static_cast<int32_t>(kernel.family)));
  attrs.append("vector_length_bytes",
               builder.getI32IntegerAttr(kernel.vectorLengthBytes));
  attrs.append("accumulate", builder.getBoolAttr(kernel.accumulate));
  attrs.append("epilogue_codegen_supported",
               builder.getBoolAttr(selection.supportsEpilogueCodegen));
  call->setDiscardableAttr(kFinalMicrokernelSelectionAttrName,
                           attrs.getDictionary(call.getContext()));
  return success();
}

FailureOr<FinalMicrokernelSelection> readFinalMicrokernelSelection(
    func::CallOp call) {
  auto attrs =
      call->getAttrOfType<DictionaryAttr>(kFinalMicrokernelSelectionAttrName);
  if (!attrs || attrs.size() != 6)
    return call.emitOpError("has no valid final microkernel selection");
  auto version = attrs.getAs<IntegerAttr>("version");
  auto symbol = attrs.getAs<StringAttr>("symbol");
  auto family = attrs.getAs<IntegerAttr>("family");
  FailureOr<int64_t> m = getDimension(call, kMicrokernelMAttrName);
  FailureOr<int64_t> n = getDimension(call, kMicrokernelNAttrName);
  FailureOr<int64_t> k = getDimension(call, kMicrokernelKAttrName);
  if (failed(m) || failed(n) || failed(k)) return failure();
  auto vl = attrs.getAs<IntegerAttr>("vector_length_bytes");
  auto accumulate = attrs.getAs<BoolAttr>("accumulate");
  auto supported = attrs.getAs<BoolAttr>("epilogue_codegen_supported");
  if (!version || version.getInt() != 1 || !symbol || !family || !vl ||
      vl.getInt() <= 0 ||
      vl.getInt() > std::numeric_limits<uint32_t>::max() || !accumulate ||
      !supported)
    return call.emitOpError("has malformed final microkernel selection");
  if (family.getInt() < static_cast<int32_t>(codegen::Family::NeonPacked) ||
      family.getInt() > static_cast<int32_t>(codegen::Family::NeonMatvec))
    return call.emitOpError("has an unknown selected microkernel enum");
  FinalMicrokernelSelection result;
  result.symbol = symbol.getValue().str();
  result.codegenKernel.family = static_cast<codegen::Family>(family.getInt());
  result.codegenKernel.m = *m;
  result.codegenKernel.n = *n;
  result.codegenKernel.k = *k;
  result.codegenKernel.vectorLengthBytes = vl.getInt();
  result.codegenKernel.accumulate = accumulate.getValue();
  result.supportsEpilogueCodegen = supported.getValue();
  return result;
}
}  // namespace annc::aarch64::gemm
