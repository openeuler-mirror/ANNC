#include "../GemmPacking.h"

#include <algorithm>

namespace annc::aarch64::gemm {

int64_t getGemmPackedBAlignment(const GemmKernelABI &abi) {
  return abi.packedBStreams *
         (abi.vectorLengthBytes / sizeof(float));
}

int64_t getGemmPackedBElements(const GemmKernelABI &abi, int64_t k,
                               int64_t n) {
  const int64_t alignment = getGemmPackedBAlignment(abi);
  return k * (n / alignment + (n % alignment != 0)) * alignment;
}

int64_t updateGemmPackedBWorkspaceElements(int64_t currentMax,
                                           const GemmKernelABI &abi, int64_t k,
                                           int64_t n) {
  return std::max(currentMax, getGemmPackedBElements(abi, k, n));
}

}  // namespace annc::aarch64::gemm
