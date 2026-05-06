#ifndef ANNC_KERNEL_BUILTIN_KDNN_ADAPTOR_TENSOR_INFO_H
#define ANNC_KERNEL_BUILTIN_KDNN_ADAPTOR_TENSOR_INFO_H

#include <cstddef>
#include <cstdint>

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

template <typename PtrT>
struct DenseTensorRef final {
    KDNN::TensorInfo info;
    PtrT data;
};

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

} // namespace annc::kernels::kdnn_adaptor

#endif // ANNC_KERNEL_BUILTIN_KDNN_ADAPTOR_TENSOR_INFO_H
