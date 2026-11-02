#ifndef ANNC_KERNEL_BUILTIN_KDNN_ADAPTOR_TENSOR_INFO_H
#define ANNC_KERNEL_BUILTIN_KDNN_ADAPTOR_TENSOR_INFO_H

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "Context/KernelContextUtils.h"
#include "Kernel/MemRefTypes.h"
#include "kdnn.hpp"

namespace annc::kernels::kdnn_adaptor {

template <std::size_t Rank>
inline KDNN::Shape makeShape(const int64_t (&values)[Rank]) {
  KDNN::Shape::SizeType dims[Rank];
  for (std::size_t i = 0; i < Rank; ++i) {
    dims[i] = static_cast<KDNN::Shape::SizeType>(values[i]);
  }
  return KDNN::Shape(dims, Rank);
}

inline KDNN::Shape makeMatrixShape(const int64_t* values) {
  const int64_t dims[] = {values[0], values[1]};
  return makeShape(dims);
}

template <typename PtrT>
struct DenseTensorRef final {
  KDNN::TensorInfo info;
  PtrT data;
};

// Requires a validated rank-2 F32 strided view with sizes and strides.
// Kernel entries must reject invalid views in release builds; these assertions
// only diagnose internal caller errors when NDEBUG is not defined.
inline DenseTensorRef<const float*> makeDenseTensorRef(
    const AnncTensorView& memref, KDNN::Layout layout) {
  assert(memref.rank == 2 && "KDNN requires a rank-2 tensor");
  assert(memref.element_type == ANNC_ELEMENT_TYPE_F32 &&
         "KDNN requires an F32 tensor");
  assert(memref.sizes && "KDNN requires tensor sizes");
  assert(memref.strides && "KDNN requires tensor strides");
  return DenseTensorRef<const float*>{
      KDNN::TensorInfo(makeMatrixShape(memref.sizes), KDNN::Element::TypeT::F32,
                       layout, makeMatrixShape(memref.strides)),
      contextData<float>(memref)};
}

// Same validated-view requirements as the const AnncTensorView overload.
inline DenseTensorRef<float*> makeMutableDenseTensorRef(AnncTensorView& memref,
                                                        KDNN::Layout layout) {
  assert(memref.rank == 2 && "KDNN requires a rank-2 tensor");
  assert(memref.element_type == ANNC_ELEMENT_TYPE_F32 &&
         "KDNN requires an F32 tensor");
  assert(memref.sizes && "KDNN requires tensor sizes");
  assert(memref.strides && "KDNN requires tensor strides");
  return DenseTensorRef<float*>{
      KDNN::TensorInfo(makeMatrixShape(memref.sizes), KDNN::Element::TypeT::F32,
                       layout, makeMatrixShape(memref.strides)),
      contextData<float>(memref)};
}

inline DenseTensorRef<const float*> makeDenseTensorRef(const AnncMemRef2DF32& memref,
                                                       KDNN::Layout layout) {
    return DenseTensorRef<const float*>{
        KDNN::TensorInfo(
            makeShape(memref.sizes),
            KDNN::Element::TypeT::F32,
            layout,
            makeShape(memref.strides)),
        ANNC_MEMREF_DATA(memref)};
}

inline DenseTensorRef<float*> makeMutableDenseTensorRef(AnncMemRef2DF32& memref,
                                                        KDNN::Layout layout) {
    return DenseTensorRef<float*>{
        KDNN::TensorInfo(
            makeShape(memref.sizes),
            KDNN::Element::TypeT::F32,
            layout,
            makeShape(memref.strides)),
        ANNC_MEMREF_DATA(memref)};
}

}  // namespace annc::kernels::kdnn_adaptor

#endif  // ANNC_KERNEL_BUILTIN_KDNN_ADAPTOR_TENSOR_INFO_H
