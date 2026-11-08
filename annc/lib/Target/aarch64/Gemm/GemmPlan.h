#ifndef ANNC_LIB_TARGET_AARCH64_GEMM_GEMMPLAN_H
#define ANNC_LIB_TARGET_AARCH64_GEMM_GEMMPLAN_H

#include <cstdint>
#include <optional>
#include <string>

#include "GemmStrategy.h"
#include "GemmThreadPlanner.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

namespace annc::aarch64::gemm {

inline constexpr llvm::StringLiteral kGemmAttrName = "annc.gemm";
inline constexpr llvm::StringLiteral kProblemAttrName =
    "annc.aarch64.gemm_problem";
inline constexpr llvm::StringLiteral kCandidateAttrName =
    "annc.aarch64.gemm_candidate";
inline constexpr llvm::StringLiteral kStrategyAttrName =
    "annc.aarch64.gemm_strategy";
inline constexpr llvm::StringLiteral kStrategiesAttrName =
    "annc.aarch64.gemm_strategies";
inline constexpr llvm::StringLiteral kCostModelsAttrName =
    "annc.aarch64.gemm_cost_models";
inline constexpr llvm::StringLiteral kStageAttrName = "annc.aarch64.gemm_stage";
inline constexpr llvm::StringLiteral kKcModeAttrName = "annc.aarch64.kc_mode";
inline constexpr llvm::StringLiteral kAsmSymbolAttrName =
    "annc.aarch64.asm_symbol";
inline constexpr llvm::StringLiteral kGeneratedMicrokernelAttrName =
    "annc.aarch64.generated_microkernel";
inline constexpr llvm::StringLiteral kEpilogueCarrierAttrName =
    "annc.aarch64.epilogue_carrier";
inline constexpr llvm::StringLiteral kEpilogueInsertionPointAttrName =
    "annc.aarch64.epilogue_insertion_point";
inline constexpr llvm::StringLiteral kKcRoleAttrName = "annc.aarch64.kc_role";
inline constexpr llvm::StringLiteral kMicrokernelMAttrName =
    "annc.aarch64.microkernel_m";
inline constexpr llvm::StringLiteral kMicrokernelNAttrName =
    "annc.aarch64.microkernel_n";
inline constexpr llvm::StringLiteral kMicrokernelKAttrName =
    "annc.aarch64.microkernel_k";
inline constexpr llvm::StringLiteral kExecutionKindAttrName =
    "execution_kind";
inline constexpr llvm::StringLiteral kRhsPackingAttrName = "rhs_packing";
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
inline constexpr llvm::StringLiteral kMatrixVectorMicrokernelLeafName =
    "__annc_aarch64_gemm_matrix_vector_microkernel_leaf";
inline constexpr llvm::StringLiteral kVectorMatrixMicrokernelLeafName =
    "__annc_aarch64_gemm_vector_matrix_microkernel_leaf";
inline constexpr llvm::StringLiteral kFusedMicrokernelLeafName =
    "__annc_aarch64_gemm_fused_microkernel_leaf";
// Row-major RHS leaf used by the small-shape no-pack path.  Its ABI carries
// the RHS leading dimension in addition to the packed-family arguments.
inline constexpr llvm::StringLiteral kMicrokernelRmLeafName =
    "__annc_aarch64_gemm_microkernel_rm_leaf";
inline constexpr llvm::StringLiteral kFusedRowMajorMicrokernelLeafName =
    "__annc_aarch64_gemm_fused_row_major_microkernel_leaf";

inline bool isFusedMicrokernelLeaf(llvm::StringRef callee) {
  return callee == kFusedMicrokernelLeafName ||
         callee == kFusedRowMajorMicrokernelLeafName;
}

inline bool isRowMajorMicrokernelLeaf(llvm::StringRef callee) {
  return callee == kMicrokernelRmLeafName ||
         callee == kFusedRowMajorMicrokernelLeafName;
}

inline bool isMicrokernelLeaf(llvm::StringRef callee) {
  return callee == kMicrokernelLeafName ||
         callee == kMatrixVectorMicrokernelLeafName ||
         callee == kVectorMatrixMicrokernelLeafName ||
         callee == kMicrokernelRmLeafName || isFusedMicrokernelLeaf(callee);
}

inline constexpr int64_t kEpilogueArgsPointerSlots = 8;
inline constexpr llvm::StringLiteral kNeonPackBAsmSymbol =
    "annc_aarch64_neon_packb_f32";
inline constexpr llvm::StringLiteral kSvePackBAsmSymbol =
    "annc_aarch64_sve_packb_f32";
inline constexpr llvm::StringLiteral kSvePackedBElementsAsmSymbol =
    "annc_aarch64_sve_packed_b_elements_f32";
inline constexpr llvm::StringLiteral kSvePackedBOffsetAsmSymbol =
    "annc_aarch64_sve_packed_b_offset_f32";

// The external tuning JSON schema version is independent from the in-IR
// candidate and problem metadata versions below.
inline constexpr int64_t kTuningConfigVersion = 3;
inline constexpr int64_t kProblemVersion = 1;
inline constexpr int64_t kCandidateVersion = 4;
inline constexpr int64_t kPrepackedRhsVersion = 2;
// Constant RHS matrices below this K*N threshold stay on the direct path.
inline constexpr int64_t kMinPrepackedRhsElements = 300;
// Problems up to this many FMAs keep the RHS in its original row-major layout
// instead of paying the runtime pack_b call.
inline constexpr int64_t kRowMajorOperationLimit = 4500;

enum class GemmTarget { kHip12, kHip09, kKp920b, kKp950 };
enum class GemmIsa { kNeon, kSve };
enum class GemmDataType { kF32 };
enum class GemmExecutionKind { kGemm, kMatrixVector, kVectorMatrix };

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
  GemmCacheStrategy cache;
  GemmKernelStrategy kernel;
  // Execution tiles remain the user configuration. The worker planner consumes
  // these exact values; it does not derive an independent cache-blocking plan.
  GemmThreadPlannerCostModel costModel;
};

struct GemmKernelABI {
  llvm::StringRef family;
  int64_t vectorLengthBytes;
  int64_t maxMr;
  int64_t maxPanelLanes;
  // Number of vector registers consumed by one fixed-count K iteration.
  int64_t kVectorUnroll;
  // Number of vector-length-wide streams stored in one packed-B panel.
  int64_t packedBStreams;
};

struct GemmDataTypeInfo {
  int64_t lhsBytes;
  int64_t rhsBytes;
  int64_t outputBytes;
};

const GemmKernelABI &getGemmKernelABI(GemmTarget target, GemmIsa isa,
                                      GemmDataType dataType,
                                      GemmExecutionKind executionKind);
llvm::FailureOr<int64_t> getGemmKScalarUnroll(const GemmKernelABI &abi,
                                              GemmDataType dataType);
const GemmDataTypeInfo &getGemmDataTypeInfo(GemmDataType dataType);
// Shared spelling of the tuning-config enum names. Both the external JSON
// loader and the in-IR metadata codec use these.
std::optional<GemmTarget> parseGemmTargetName(llvm::StringRef name);
std::optional<GemmIsa> parseGemmIsaName(llvm::StringRef name);
std::optional<GemmDataType> parseGemmDataTypeName(llvm::StringRef name);
llvm::Expected<GemmTuningConfig> loadGemmTuningConfig(llvm::StringRef path);
llvm::FailureOr<int64_t> getGemmNr(int64_t panelLanes,
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

// Reusable tuning choices. The problem, thread budget, and cost model remain
// separate inputs to planning.
struct GemmTuningStrategy {
  GemmTarget target;
  GemmIsa isa;
  GemmDataType dataType;
  GemmExecutionKind executionKind;
  GemmCacheStrategy cache;
  GemmKernelStrategy kernel;
  RhsPacking rhsPacking;
};

inline const GemmKernelABI &getGemmKernelABI(
    const GemmTuningStrategy &strategy) {
  return getGemmKernelABI(strategy.target, strategy.isa, strategy.dataType,
                          strategy.executionKind);
}

// Inputs to thread planning: selected tuning, cost model, and thread budget.
struct GemmThreadPlanningInput {
  GemmTuningStrategy strategy;
  GemmThreadPlannerCostModel costModel;
  int64_t maxThreadCount;
};

// A tuning strategy plus the planned static task topology, after thread
// planning.
struct GemmCandidate {
  GemmTuningStrategy strategy;
  int64_t threadCount;
  int64_t tasksM;
  int64_t tasksN;
  bool shardByColumns;
};

struct GemmExecutionContext {
  GemmProblem problem;
  GemmTuningStrategy strategy;
  std::string prepackedRhsSymbol;
  int64_t prepackedRhsElements = 0;
  bool prepackedRhsFromInput = false;
};

bool isGemmAnchor(mlir::Operation *op);
mlir::Value getGemmInput(mlir::Operation *op, unsigned index);
mlir::Value getGemmOutput(mlir::Operation *op, unsigned index);
mlir::FailureOr<GemmProblem> readProblem(mlir::Operation *op);
mlir::FailureOr<GemmTuningStrategy> readStrategy(mlir::Operation *op);
mlir::FailureOr<GemmTuningStrategy> readTuningStrategy(
    mlir::Operation *op, mlir::DictionaryAttr candidate);
mlir::FailureOr<GemmThreadPlanningInput> readThreadPlanningInput(
    mlir::Operation *op);
mlir::FailureOr<GemmCandidate> readCandidate(mlir::Operation *op);
mlir::FailureOr<GemmExecutionContext> readExecutionContext(
    mlir::Operation *op);
void writeProblem(mlir::Operation *op, const GemmProblem &problem);
void writeThreadPlanningInput(mlir::Operation *op,
                              const GemmThreadPlanningInput &input);
void writeCandidate(mlir::Operation *op,
                    const GemmThreadPlanningResult &result);
void setCandidateRhsPacking(mlir::Operation *op, RhsPacking packing);
void copyStrategyReference(mlir::Operation *source, mlir::Operation *dest);
void copyGemmMetadata(mlir::Operation *source, mlir::Operation *destination);
mlir::DictionaryAttr encodePrepackedRhsBinding(mlir::Operation *op, int64_t k,
                                               int64_t n,
                                               llvm::StringRef symbol,
                                               int64_t elements,
                                               bool fromInput = false);
void pruneGemmMetadata(mlir::Operation *owner);
void pruneGemmMetadata(mlir::ModuleOp module);
llvm::StringRef getGemmTargetName(GemmTarget target);
llvm::StringRef getGemmIsaName(GemmIsa isa);
llvm::StringRef getGemmExecutionKindName(GemmExecutionKind kind);

struct GemmPathSelection {
  GemmExecutionKind executionKind;
  RhsPacking rhsPacking;
};

GemmPathSelection selectGemmPath(GemmIsa isa, int64_t m, int64_t n,
                                 int64_t k, bool enablePrepack,
                                 bool hasPrepackableRhs,
                                 bool allowSpecializedExecution);

// The microkernel leaf family selected for a strategy, corresponding one-to-one
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
