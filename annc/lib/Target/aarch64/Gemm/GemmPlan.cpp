#include "GemmPlan.h"

#include <algorithm>

#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Diagnostics.h"

namespace annc::aarch64::gemm {

const GemmKernelABI &getGemmKernelABI(GemmTarget target, GemmIsa isa,
                                      GemmDataType dataType,
                                      GemmExecutionKind executionKind) {
  (void)target;
  (void)dataType;
  static const GemmKernelABI neon{"annc-neon-f32-v1", 16, 6, 4, 1};
  static const GemmKernelABI sve{"annc-sve-f32-v1", 32, 6, 4, 0};
  static const GemmKernelABI matrixVector{"annc-neon-matvec-f32-v1", 16, 4,
                                           1, 4};
  static const GemmKernelABI vectorMatrix{"annc-neon-vecmat-f32-v1", 16, 1,
                                          4, 2};
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

llvm::FailureOr<int64_t> getGemmNr(const GemmKernelTile &kernelTile,
                                   int64_t vectorLengthBytes,
                                   GemmDataType dataType,
                                   GemmExecutionKind executionKind) {
  const int64_t elementBytes = dataType == GemmDataType::kF32 ? 4 : 0;
  if (elementBytes == 0 || vectorLengthBytes <= 0 ||
      vectorLengthBytes % elementBytes != 0 || kernelTile.panelLanes <= 0)
    return mlir::failure();
  if (executionKind == GemmExecutionKind::kMatrixVector) {
    if (kernelTile.panelLanes != 1) return mlir::failure();
    return int64_t{1};
  }
  return kernelTile.panelLanes * (vectorLengthBytes / elementBytes);
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
                                 bool hasPrepackableRhs) {
  if (m * n * k <= kRowMajorOperationLimit)
    return {GemmExecutionKind::kGemm, RhsPacking::kDirect};
  if (isa == GemmIsa::kNeon && n == 1)
    return {GemmExecutionKind::kMatrixVector, RhsPacking::kDirect};
  if (isa == GemmIsa::kNeon && m == 1)
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
