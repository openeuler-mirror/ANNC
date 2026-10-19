#ifndef ANNC_HELPER_H
#define ANNC_HELPER_H
#include <cstdlib>
#include <numeric>

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace annc {
float apFloatToFloat(const llvm::APFloat &v);

double apFloatToDouble(const llvm::APFloat &v);

// Reads ANNC_GEMM, the MatMul fusion level switch. Consumers that must stay
// in sync on this contract:
//   - OpFusion (FuseMatMulAsFuncCallPattern): fusion gate + epilogue
//     candidate consumption;
//   - annc-tf-pipeline: runs the epilogue discovery pass at level 2 (the
//     candidate attr also reaches Target/aarch64 GEMM strategy selection,
//     so discovery must not run unconditionally).
// Levels: unset/"0" = off (default), "1" = plain MatMul only,
// "2" = MatMul + epilogue tail fusion (needs ANNC_GEMM_ASM_SOURCE_DIR).
// Surrounding whitespace is tolerated; anything unrecognized is 0.
// Read once on first use and cached for the process lifetime (function-local
// static: thread-safe init, consistent view — same pattern as Support/Log.h).
inline int anncGemmFusionLevel() {
  static const int level = [] {
    const char *value = std::getenv("ANNC_GEMM");
    if (!value) return 0;
    StringRef trimmed = StringRef(value).trim();
    if (trimmed == "1") return 1;
    if (trimmed == "2") return 2;
    return 0;
  }();
  return level;
}

#define CHECK_LOGICAL_SUCCESS(f) \
  if (!(f)) return failure();

void outputCode(const ModuleOp &module, const std::string &filenameWithExt,
                int64_t elide = 1);
void outputBinary(const ModuleOp &module, const std::string &output);

llvm::ArrayRef<int64_t> getTensorShape(const Value &val);

NameLoc getLoc(MLIRContext *context, const std::string &name);
std::string getLocName(Operation *op);
std::string getValLocName(Value val);

template <typename T>
T getIntValSafely(const Attribute &attr) {
  auto attr_ = dyn_cast<mlir::IntegerAttr>(attr);
  if (!attr_) return -1;
  if (attr_.getType().isSignedInteger()) {
    return static_cast<T>(attr_.getSInt());
  } else {
    return static_cast<T>(attr_.getInt());
  }
}
template <typename T>
llvm::ArrayRef<T> getIntArrSafely(const Attribute &arrAttr) {
  auto arrAttr_ = dyn_cast<ArrayAttr>(arrAttr);
  if (!arrAttr_) return {};
  llvm::SmallVector<T> arr;
  for (const auto &attr : arrAttr_) {
    arr.push_back(getIntValSafely<T>(attr));
  }
  return arr;
}
template <typename T>
llvm::SmallVector<T> ArrayToSmallVector(const ArrayAttr &arrAttr) {
  llvm::SmallVector<T> arr = {};
  if (arrAttr == nullptr) return arr;
  for (const auto &attr : arrAttr) {
    arr.push_back(getIntValSafely<T>(attr));
  }
  return arr;
}

template <typename T>
std::vector<T> mergeCommon(const std::vector<T> &vec1,
                           const std::vector<T> &vec2) {
  std::vector<T> common;
  std::for_each(vec1.begin(), vec1.end(), [&](T t) {
    if (std::find(vec2.begin(), vec2.end(), t) != vec2.end())
      common.push_back(t);
  });
  return common;
}

std::string ArrayToStr(llvm::ArrayRef<int64_t> &vec);

std::string getStrValSafely(const Attribute &attr);
std::vector<std::string> StrArrayToSmallVector(const ArrayAttr &attr);

std::vector<std::string> splitString(const std::string &str,
                                     char delimiter = ',');

int64_t getNumElements(Value val);
int64_t getElementSize(Value val);
int64_t getSize(Value val);

std::string getBaseName(std::string path);
void updateFuncOp(func::FuncOp funcOp);
std::string getOpTyName(Operation *op);

func::ReturnOp getReturnOpFromFunc(func::FuncOp funcOp);

inline func::FuncOp getFuncOpFromModuleOp(ModuleOp mOp, StringRef funcName) {
  return mOp.lookupSymbol<func::FuncOp>(funcName);
}
}  // namespace annc
#endif  // ANNC_HELPER_H