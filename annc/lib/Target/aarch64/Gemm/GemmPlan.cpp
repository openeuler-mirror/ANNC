#include "GemmPlan.h"

#include <algorithm>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Diagnostics.h"

namespace annc::aarch64::gemm {

const GemmKernelABI &getGemmKernelABI(GemmTarget target, GemmIsa isa,
                                      GemmDataType dataType,
                                      GemmExecutionKind executionKind) {
  (void)target;
  (void)dataType;
  static const GemmKernelABI neon{"annc-neon-f32-v1", 16, 6, 4, 1, 1};
  static const GemmKernelABI sve{"annc-sve-f32-v1", 32, 6, 4, 0, 4};
  static const GemmKernelABI matrixVector{"annc-neon-matvec-f32-v1", 16, 4,
                                           1, 4, 1};
  static const GemmKernelABI vectorMatrix{"annc-neon-vecmat-f32-v1", 16, 1,
                                          4, 2, 1};
  if (executionKind == GemmExecutionKind::kMatrixVector &&
      isa == GemmIsa::kNeon)
    return matrixVector;
  if (executionKind == GemmExecutionKind::kVectorMatrix &&
      isa == GemmIsa::kNeon)
    return vectorMatrix;
  return isa == GemmIsa::kSve ? sve : neon;
}

llvm::FailureOr<int64_t> getGemmKScalarUnroll(const GemmKernelABI &abi,
                                              GemmDataType dataType) {
  const int64_t elementBytes = dataType == GemmDataType::kF32 ? 4 : 0;
  if (elementBytes == 0 || abi.vectorLengthBytes <= 0 ||
      abi.vectorLengthBytes % elementBytes != 0 || abi.kVectorUnroll <= 0)
    return mlir::failure();
  return abi.kVectorUnroll * (abi.vectorLengthBytes / elementBytes);
}

llvm::StringRef getGemmDataTypeName(GemmDataType dataType) {
  (void)dataType;
  return "f32";
}

const GemmDataTypeInfo &getGemmDataTypeInfo(GemmDataType dataType) {
  (void)dataType;
  static const GemmDataTypeInfo f32{4, 4, 4};
  return f32;
}

llvm::FailureOr<int64_t> getGemmNr(int64_t panelLanes,
                                   int64_t vectorLengthBytes,
                                   GemmDataType dataType,
                                   GemmExecutionKind executionKind) {
  const int64_t elementBytes = dataType == GemmDataType::kF32 ? 4 : 0;
  if (elementBytes == 0 || vectorLengthBytes <= 0 ||
      vectorLengthBytes % elementBytes != 0 || panelLanes <= 0)
    return mlir::failure();
  if (executionKind == GemmExecutionKind::kMatrixVector) {
    if (panelLanes != 1) return mlir::failure();
    return int64_t{1};
  }
  return panelLanes * (vectorLengthBytes / elementBytes);
}

bool isGemmAnchor(mlir::Operation *op) {
  return llvm::isa<mlir::linalg::MatmulOp>(op) ||
         (llvm::isa<mlir::linalg::GenericOp>(op) && op->hasAttr(kGemmAttrName));
}

mlir::Value getGemmInput(mlir::Operation *op, unsigned index) {
  if (auto matmul = llvm::dyn_cast<mlir::linalg::MatmulOp>(op))
    return matmul.getInputs()[index];
  return llvm::cast<mlir::linalg::GenericOp>(op).getInputs()[index];
}

mlir::Value getGemmOutput(mlir::Operation *op, unsigned index) {
  if (auto matmul = llvm::dyn_cast<mlir::linalg::MatmulOp>(op))
    return matmul.getOutputs()[index];
  return llvm::cast<mlir::linalg::GenericOp>(op).getOutputs()[index];
}

int64_t resolveEntryInputSlot(mlir::Value value, mlir::func::FuncOp func,
                              bool *exclusive) {
  mlir::Value root = value;
  if (exclusive) *exclusive = root.hasOneUse();
  while (mlir::Operation *def = root.getDefiningOp()) {
    mlir::Value next;
    if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(def)) {
      next = cast.getSource();
    } else if (auto cast = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(
                   def)) {
      if (cast->getNumOperands() != 1) break;
      next = cast->getOperand(0);
    } else if (auto toMemref =
                   llvm::dyn_cast<mlir::bufferization::ToMemrefOp>(def)) {
      next = toMemref.getTensor();
    } else if (auto toTensor =
                   llvm::dyn_cast<mlir::bufferization::ToTensorOp>(def)) {
      next = toTensor.getMemref();
    } else if (auto load = llvm::dyn_cast<mlir::LLVM::LoadOp>(def)) {
      next = load->getOperand(0);
    } else {
      break;
    }
    root = next;
    if (exclusive) *exclusive &= root.hasOneUse();
  }
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(root))
    return argument.getOwner() == &func.front() ? argument.getArgNumber() : -1;
  if (auto call = root.getDefiningOp<mlir::func::CallOp>()) {
    if (call.getCallee() == "annc_context_get_input_memref" &&
        call.getNumOperands() == 2 && func.getNumArguments() > 0 &&
        call.getOperand(0) == func.front().getArgument(0)) {
      mlir::Operation *constant = call->getOperand(1).getDefiningOp();
      if (const auto attr = constant ? constant->getAttrOfType<mlir::IntegerAttr>(
                                           "value")
                                     : mlir::IntegerAttr{})
        return attr.getValue().getSExtValue();
    }
  }
  return -1;
}

bool entrySlotRoleImmutable(mlir::func::FuncOp func, int64_t slot) {
  auto metadata = func->getAttrOfType<mlir::DictionaryAttr>("fusion.metadata");
  auto args =
      metadata ? metadata.getAs<mlir::ArrayAttr>("args") : mlir::ArrayAttr{};
  // The args array mirrors the original entry argument order; aux promotion
  // arguments appended after it have no boundary record and stay ineligible.
  if (!args || slot < 0 || slot >= static_cast<int64_t>(args.size()))
    return false;
  auto arg = llvm::dyn_cast<mlir::DictionaryAttr>(args[slot]);
  auto role = arg ? arg.getAs<mlir::StringAttr>("role") : mlir::StringAttr{};
  return role && (role.getValue() == "fixed" || role.getValue() == "constant");
}

llvm::StringRef getGemmTargetName(GemmTarget target) {
  switch (target) {
    case GemmTarget::kHip09:
      return "hip09";
    case GemmTarget::kHip12:
      return "hip12";
    case GemmTarget::kKp920b:
      return "kp920b";
    case GemmTarget::kKp950:
      return "kp950";
  }
  return "unknown";
}

llvm::StringRef getGemmIsaName(GemmIsa isa) {
  return isa == GemmIsa::kSve ? "sve" : "neon";
}

llvm::StringRef getGemmExecutionKindName(GemmExecutionKind kind) {
  switch (kind) {
    case GemmExecutionKind::kMatrixVector:
      return "matrix-vector";
    case GemmExecutionKind::kVectorMatrix:
      return "vector-matrix";
    case GemmExecutionKind::kGemm:
      return "gemm";
  }
  llvm_unreachable("unknown GEMM execution kind");
}

GemmPathSelection selectGemmPath(GemmIsa isa, int64_t m, int64_t n,
                                 int64_t k, bool enablePrepack,
                                 bool hasPrepackableRhs,
                                 bool allowSpecializedExecution) {
  if (m * n * k <= kRowMajorOperationLimit)
    return {GemmExecutionKind::kGemm, RhsPacking::kDirect};
  if (allowSpecializedExecution && isa == GemmIsa::kNeon && n == 1)
    return {GemmExecutionKind::kMatrixVector, RhsPacking::kDirect};
  if (allowSpecializedExecution && isa == GemmIsa::kNeon && m == 1)
    return {GemmExecutionKind::kVectorMatrix, RhsPacking::kDirect};
  if (enablePrepack && hasPrepackableRhs &&
      k * n >= kMinPrepackedRhsElements)
    return {GemmExecutionKind::kGemm, RhsPacking::kPrepacked};
  return {GemmExecutionKind::kGemm, RhsPacking::kPacked};
}

GemmLeafKind getGemmLeafKind(GemmExecutionKind executionKind,
                             RhsPacking rhsPacking) {
  if (executionKind == GemmExecutionKind::kMatrixVector)
    return GemmLeafKind::kMatvec;
  if (executionKind == GemmExecutionKind::kVectorMatrix)
    return GemmLeafKind::kVecmat;
  return rhsPacking == RhsPacking::kDirect ? GemmLeafKind::kRowMajor
                                           : GemmLeafKind::kPacked;
}

bool usesLdbAbi(GemmLeafKind kind) {
  return kind == GemmLeafKind::kRowMajor || kind == GemmLeafKind::kVecmat;
}

llvm::StringRef getGemmPackedBSchema(GemmIsa isa) {
  return isa == GemmIsa::kSve ? "annc-sve-packed-b-v2"
                              : "annc-neon-packed-b-v1";
}

std::vector<CacheBlock2D> partitionCache2D(int64_t k, int64_t n, int64_t kc,
                                           int64_t nc) {
  std::vector<CacheBlock2D> blocks;
  if (k <= 0 || n <= 0) return blocks;
  kc = std::max<int64_t>(kc, 1);
  nc = std::max<int64_t>(nc, 1);
  const int64_t fullN = n / nc * nc;
  const auto appendJc = [&](int64_t kStart, int64_t kSize) {
    for (int64_t nStart = 0; nStart < fullN; nStart += nc)
      blocks.push_back({kStart, kSize, nStart, nc});
    if (fullN < n) blocks.push_back({kStart, kSize, fullN, n - fullN});
  };
  const int64_t firstK = std::min(k, kc);
  appendJc(0, firstK);
  const int64_t remaining = k - firstK;
  for (int64_t i = 0; i < remaining / kc; ++i)
    appendJc(firstK + i * kc, kc);
  if (remaining % kc != 0)
    appendJc(firstK + remaining / kc * kc, remaining % kc);
  return blocks;
}

llvm::StringRef getPackBAsmSymbol(GemmTarget target, GemmIsa isa) {
  (void)target;
  return isa == GemmIsa::kSve ? kSvePackBAsmSymbol : kNeonPackBAsmSymbol;
}

llvm::StringRef getKcModeName(KcMode mode) {
  return mode == KcMode::kOverwrite ? "overwrite" : "accumulate";
}

mlir::LogicalResult requireStage(mlir::Operation *op,
                                 llvm::StringRef expected) {
  auto stage = op->getAttrOfType<mlir::StringAttr>(kStageAttrName);
  if (!stage || stage.getValue() != expected) {
    return op->emitOpError()
           << "requires " << kStageAttrName << " = \"" << expected << "\"";
  }
  return mlir::success();
}

bool hasStage(mlir::Operation *op, llvm::StringRef expected) {
  auto stage = op->getAttrOfType<mlir::StringAttr>(kStageAttrName);
  return stage && stage.getValue() == expected;
}

void setStage(mlir::Operation *op, mlir::Builder &builder,
              llvm::StringRef stage) {
  op->setDiscardableAttr(kStageAttrName, builder.getStringAttr(stage));
}

}  // namespace annc::aarch64::gemm
