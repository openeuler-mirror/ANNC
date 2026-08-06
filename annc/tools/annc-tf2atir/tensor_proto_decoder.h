#ifndef ANNC_TF2ATIR_TENSOR_PROTO_DECODER_H
#define ANNC_TF2ATIR_TENSOR_PROTO_DECODER_H

#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor.pb.h"

namespace annc::tf2atir {

class TensorProtoDecoder {
 public:
  static bool decode(const tensorflow::TensorProto& tensor,
                     const std::string& dtype, std::vector<uint8_t>& bytes,
                     std::string& error);
};

std::string base64Encode(const std::vector<uint8_t>& data);

}  // namespace annc::tf2atir

#endif  // ANNC_TF2ATIR_TENSOR_PROTO_DECODER_H
