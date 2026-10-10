#ifndef ANNC_ONNX2ATIR_ONNX_TENSOR_DECODER_H
#define ANNC_ONNX2ATIR_ONNX_TENSOR_DECODER_H

#include <cstdint>
#include <string>
#include <vector>

#include "onnx/onnx.pb.h"

// ONNX stores raw payloads little-endian, and this decoder rebuilds the typed
// fields by memcpy-ing host bytes rather than converting, so on a big-endian
// host every constant would be silently byte-swapped. Refuse to build there.
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "the ONNX frontend assumes a little-endian host"
#endif

namespace annc::onnx2atir {

// Maps an ONNX TensorProto.DataType to the ATIR/`annc::DType` canonical name
// ("float32", "int64", ...). Returns false for types the ATIR tensor type
// cannot carry (float8/int4 variants, UNDEFINED).
bool onnxDtypeName(int data_type, std::string& name);

// Decodes a TensorProto into little-endian host-layout bytes, the layout
// annc::MLIRBuilder::buildConstantNode expects in NodeInfo::raw_data.
// Handles raw_data plus every typed data field; rejects external data and
// string payloads (strings come back through decodeStrings instead).
bool decodeTensorBytes(const onnx::TensorProto& tensor,
                       const std::string& dtype_name,
                       std::vector<uint8_t>& bytes, std::string& error);

// Decodes a TensorProto of element type STRING into its values.
bool decodeTensorStrings(const onnx::TensorProto& tensor,
                         std::vector<std::string>& values, std::string& error);

// Number of elements the tensor's `dims` describe.
int64_t tensorElementCount(const onnx::TensorProto& tensor);

}  // namespace annc::onnx2atir

#endif  // ANNC_ONNX2ATIR_ONNX_TENSOR_DECODER_H
