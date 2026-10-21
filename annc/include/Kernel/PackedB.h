#ifndef ANNC_PACKED_B_H
#define ANNC_PACKED_B_H

#include <cstddef>
#include <cstdint>

// A read-only description of MLAS-packed B slices.  The owner lives in the EP
// compute state; kernels only borrow this descriptor for the duration of one
// call.  `slices` is indexed by the non-matrix batch dimensions in row-major
// order and points at MLAS-packed bytes represented as float pointers solely
// to keep the descriptor C-compatible with the existing memref code.
struct AnncPackedB {
  int32_t rank = 0;
  int64_t shape[4] = {0, 0, 0, 0};
  size_t slice_count = 0;
  const float* const* slices = nullptr;
};

namespace annc {
namespace kernels {

inline bool packedBMatches(const AnncPackedB* packed, const int64_t* shape,
                           int32_t rank) {
  if (packed == nullptr || shape == nullptr || rank != packed->rank ||
      rank < 2 || rank > 4 || packed->slices == nullptr) {
    return false;
  }
  for (int32_t i = 0; i < rank; ++i) {
    if (shape[i] != packed->shape[i]) return false;
  }
  size_t expected_slices = 1;
  for (int32_t i = 0; i < rank - 2; ++i) {
    if (packed->shape[i] <= 0 ||
        static_cast<size_t>(packed->shape[i]) >
            static_cast<size_t>(-1) / expected_slices) {
      return false;
    }
    expected_slices *= static_cast<size_t>(packed->shape[i]);
  }
  if (packed->slice_count != expected_slices) return false;
  for (size_t i = 0; i < packed->slice_count; ++i) {
    if (packed->slices[i] == nullptr) return false;
  }
  return true;
}

inline const float* packedBSlice(const AnncPackedB* packed, int64_t first,
                                 int64_t second) {
  if (packed == nullptr || packed->slices == nullptr || first < 0 ||
      second < 0) {
    return nullptr;
  }
  size_t index = 0;
  if (packed->rank == 2) {
    if (first != 0 || second != 0) return nullptr;
  } else if (packed->rank == 3) {
    if (second != 0 || static_cast<size_t>(first) >= packed->slice_count) {
      return nullptr;
    }
    index = static_cast<size_t>(first);
  } else if (packed->rank == 4) {
    const size_t width = static_cast<size_t>(packed->shape[1]);
    if (width == 0 || static_cast<size_t>(first) >=
                          static_cast<size_t>(packed->shape[0]) ||
        static_cast<size_t>(second) >= width) {
      return nullptr;
    }
    index = static_cast<size_t>(first) * width + static_cast<size_t>(second);
  } else {
    return nullptr;
  }
  if (index >= packed->slice_count) return nullptr;
  return packed->slices[index];
}

}  // namespace kernels
}  // namespace annc

#endif
