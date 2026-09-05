#ifndef ANNC_LIB_TARGET_AARCH64_GEMM_GEMMPLAN_H
#define ANNC_LIB_TARGET_AARCH64_GEMM_GEMMPLAN_H

#include <cstdint>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

namespace annc::aarch64::gemm {

inline constexpr llvm::StringLiteral kEpilogueAttrName = "annc.gemm.epilogue";
inline constexpr llvm::StringLiteral kGemmAttrName = "annc.gemm";
inline constexpr llvm::StringLiteral kProblemAttrName =
    "annc.aarch64.gemm_problem";
inline constexpr llvm::StringLiteral kCandidateAttrName =
    "annc.aarch64.gemm_candidate";
inline constexpr llvm::StringLiteral kPlanAttrName = "annc.aarch64.gemm_plan";
inline constexpr llvm::StringLiteral kStageAttrName = "annc.aarch64.gemm_stage";
inline constexpr llvm::StringLiteral kKcModeAttrName = "annc.aarch64.kc_mode";
inline constexpr llvm::StringLiteral kAsmSymbolAttrName =
    "annc.aarch64.asm_symbol";
inline constexpr llvm::StringLiteral kMicrokernelMAttrName =
    "annc.aarch64.microkernel_m";
inline constexpr llvm::StringLiteral kMicrokernelNAttrName =
    "annc.aarch64.microkernel_n";
inline constexpr llvm::StringLiteral kMicrokernelKAttrName =
    "annc.aarch64.microkernel_k";
inline constexpr llvm::StringLiteral kIntraThreadCountAttrName =
    "annc.intra_thread_count";
inline constexpr llvm::StringLiteral kExecutionKindAttrName =
    "execution_kind";
inline constexpr llvm::StringLiteral kRhsPackingAttrName = "rhs_packing";
inline constexpr llvm::StringLiteral kRhsPackSourceAttrName =
    "rhs_pack_source";
inline constexpr llvm::StringLiteral kPrepackedRhsAttrName =
    "annc.aarch64.prepacked_rhs";
inline constexpr llvm::StringLiteral kRhsNameAttrName =
    "annc.aarch64.rhs_name";

inline constexpr llvm::StringLiteral kCacheBlockedStage = "cache_blocked";
inline constexpr llvm::StringLiteral kThreadTiledStage = "thread_tiled";
inline constexpr llvm::StringLiteral kKernelTiledStage = "kernel_tiled";
inline constexpr llvm::StringLiteral kPackedStage = "packed";
inline constexpr llvm::StringLiteral kMicrokernelLoweredStage =
    "microkernel_lowered";

inline constexpr llvm::StringLiteral kPackBLeafName =
    "__annc_aarch64_gemm_pack_b_leaf";
inline constexpr llvm::StringLiteral kMicrokernelLeafName =
    "__annc_aarch64_gemm_microkernel_leaf";
// Row-major RHS leaf used by the small-shape no-pack path.  Its ABI carries
// the RHS leading dimension in addition to the packed-family arguments.
inline constexpr llvm::StringLiteral kMicrokernelRmLeafName =
    "__annc_aarch64_gemm_microkernel_rm_leaf";
inline constexpr llvm::StringLiteral kNeonPackBAsmSymbol =
    "annc_aarch64_neon_packb_f32";
inline constexpr llvm::StringLiteral kSvePackBAsmSymbol =
    "annc_aarch64_sve_packb_f32";
inline constexpr llvm::StringLiteral kSvePackedBElementsAsmSymbol =
    "annc_aarch64_sve_packed_b_elements_f32";
inline constexpr llvm::StringLiteral kSvePackedBOffsetAsmSymbol =
    "annc_aarch64_sve_packed_b_offset_f32";

inline constexpr int64_t kPlanVersion = 1;
inline constexpr int64_t kPrepackedRhsVersion = 1;
// Constant RHS matrices below this K*N threshold stay on the direct path.
inline constexpr int64_t kMinPrepackedRhsElements = 300;
// Problems up to this many FMAs keep the RHS in its original row-major layout
// instead of paying the runtime pack_b call.
inline constexpr int64_t kRowMajorOperationLimit = 4500;

enum class GemmTarget { kHip12, kHip09 };
enum class GemmIsa { kNeon, kSve };
enum class GemmDataType { kF32 };
enum class GemmExecutionKind { kGemm, kMatrixVector, kVectorMatrix };

struct GemmCacheTile {
  int64_t mc;
  int64_t nc;
  int64_t kc;
};

struct GemmKernelTile {
  int64_t mr;
  int64_t panelLanes;
};

// One (KC, NC) cache block of the pc-jc ordered RHS partition.
struct CacheBlock2D {
  int64_t kStart;
  int64_t kSize;
  int64_t nStart;
  int64_t nSize;
};

std::vector<CacheBlock2D> partitionCache2D(int64_t k, int64_t n, int64_t kc,
                                           int64_t nc);

struct GemmTuningConfig {
  GemmTarget target;
  GemmIsa isa;
  GemmDataType dataType;
  GemmCacheTile cacheTile;
  GemmKernelTile kernelTile;
};

struct GemmKernelABI {
  llvm::StringRef family;
  int64_t vectorLengthBytes;
  int64_t maxMr;
  int64_t maxPanelLanes;
  // Number of vector registers consumed by one fixed-count K iteration.
  int64_t kVectorUnroll;
};

const GemmKernelABI &getGemmKernelABI(GemmTarget target, GemmIsa isa,
                                      GemmDataType dataType,
                                      GemmExecutionKind executionKind);
llvm::FailureOr<int64_t> getGemmKScalarUnroll(const GemmKernelABI &abi,
                                              GemmDataType dataType);
llvm::Expected<GemmTuningConfig> loadGemmTuningConfig(llvm::StringRef path);
llvm::FailureOr<int64_t> getGemmNr(const GemmKernelTile &kernelTile,
                                   int64_t vectorLengthBytes,
                                   GemmDataType dataType,
                                   GemmExecutionKind executionKind);
llvm::StringRef getGemmDataTypeName(GemmDataType dataType);

struct GemmProblem {
  int64_t version;
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t lda;
  int64_t ldb;
  int64_t ldc;
};

// Whether the RHS is read in its original layout or materialized by pack_b.
enum class RhsPacking { kDirect, kPacked, kPrepacked };
enum class KcMode { kOverwrite, kAccumulate };
enum class RhsPackSource { kNone, kGenerated, kPrepacked };

struct GemmCandidate {
  int64_t version;
  GemmTarget target;
  GemmIsa isa;
  GemmDataType dataType;
  GemmCacheTile cacheTile;
  GemmKernelTile kernelTile;
  int64_t threadCount;
  GemmExecutionKind executionKind;
  RhsPacking rhsPacking;
};

struct GemmTilingPlan {
  int64_t version;
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  GemmCacheTile cacheTile;
  GemmKernelTile kernelTile;
  GemmDataType dataType;
  int64_t vectorLengthBytes;
  int64_t threadCount;
  KcMode firstKcMode;
  KcMode nextKcMode;
  GemmExecutionKind executionKind;
  RhsPacking rhsPacking;
  RhsPackSource rhsPackSource;
};

struct GemmPlan : GemmTilingPlan {
  GemmTarget target;
  GemmIsa isa;
  std::string prepackedRhsSymbol;
  int64_t prepackedRhsElements = 0;
};

bool isGemmAnchor(mlir::Operation *op);
mlir::Value getGemmInput(mlir::Operation *op, unsigned index);
mlir::Value getGemmOutput(mlir::Operation *op, unsigned index);
mlir::FailureOr<GemmProblem> readProblem(mlir::Operation *op);
mlir::FailureOr<GemmCandidate> readCandidate(mlir::Operation *op);
mlir::FailureOr<GemmTilingPlan> readTilingPlan(mlir::Operation *op);
mlir::FailureOr<GemmPlan> readPlan(mlir::Operation *op);
llvm::StringRef getGemmTargetName(GemmTarget target);
llvm::StringRef getGemmIsaName(GemmIsa isa);
llvm::StringRef getGemmExecutionKindName(GemmExecutionKind kind);

struct GemmPathSelection {
  GemmExecutionKind executionKind;
  RhsPacking rhsPacking;
};

GemmPathSelection selectGemmPath(GemmIsa isa, int64_t m, int64_t n,
                                 int64_t k, bool enablePrepack,
                                 bool hasPrepackableRhs);

// The microkernel leaf family selected for a plan, corresponding one-to-one
// with selectGemmPath: the matrix-vector (N==1) and vector-matrix (M==1)
// leaves, the row-major small-shape GEMM leaf, and the packed GEMM leaf.
// The argument signature follows: kMatvec and kPacked use seven index
// operands (no RHS leading dimension); kVecmat and kRowMajor add ldb as an
// eighth index operand.
enum class GemmLeafKind {
  kMatvec,
  kVecmat,
  kRowMajor,
  kPacked,
};

GemmLeafKind getGemmLeafKind(GemmExecutionKind executionKind,
                             RhsPacking rhsPacking);

// Whether the leaf signature carries the RHS leading dimension (kVecmat and
// kRowMajor use the row-major ldb ABI with eight index operands).
bool usesLdbAbi(GemmLeafKind kind);
llvm::StringRef getGemmPackedBSchema(GemmIsa isa);
llvm::StringRef getPackBAsmSymbol(GemmTarget target, GemmIsa isa);
llvm::StringRef getKcModeName(KcMode mode);
mlir::LogicalResult requireStage(mlir::Operation *op, llvm::StringRef expected);
bool hasStage(mlir::Operation *op, llvm::StringRef expected);
void setStage(mlir::Operation *op, mlir::Builder &builder,
              llvm::StringRef stage);

}  // namespace annc::aarch64::gemm

#endif  // ANNC_LIB_TARGET_AARCH64_GEMM_GEMMPLAN_H
