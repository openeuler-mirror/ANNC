#ifndef ANNC_LIB_TARGET_AARCH64_GEMM_GEMMPACKING_H
#define ANNC_LIB_TARGET_AARCH64_GEMM_GEMMPACKING_H

#include "GemmPlan.h"

namespace annc::aarch64::gemm {

int64_t getGemmPackedBAlignment(const GemmKernelABI &abi);
int64_t getGemmPackedBElements(const GemmKernelABI &abi, int64_t k, int64_t n);

}  // namespace annc::aarch64::gemm

#endif  // ANNC_LIB_TARGET_AARCH64_GEMM_GEMMPACKING_H
