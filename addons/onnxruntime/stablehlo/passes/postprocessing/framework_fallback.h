#ifndef ANNC_ONNX_STABLEHLO_FRAMEWORK_FALLBACK_H
#define ANNC_ONNX_STABLEHLO_FRAMEWORK_FALLBACK_H

#include "llvm/ADT/StringRef.h"

namespace annc {

// Explicit allowlist shared by conversion and output verification. An unknown
// private operation must not silently become a framework fallback.
inline bool isSupportedOnnxFallback(llvm::StringRef domain,
                                    llvm::StringRef name, unsigned inputs,
                                    unsigned outputs) {
  return domain == "com.microsoft" && name == "FusedTensordotMatMul" &&
         inputs == 2 && outputs == 1;
}

}  // namespace annc

#endif
