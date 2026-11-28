#include "onnx_tensor_decoder.h"

#include <cstring>

namespace annc::onnx2atir {
namespace {

template <typename T>
void appendLittleEndian(std::vector<uint8_t>& bytes, T value) {
  static_assert(std::is_trivially_copyable<T>::value, "trivially copyable");
  const size_t offset = bytes.size();
  bytes.resize(offset + sizeof(T));
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

// ONNX stores every sub-32-bit integer type, plus BOOL/FLOAT16/BFLOAT16, in
// `int32_data` one value per int32. Narrowing back is what rebuilds the exact
// byte layout a same-typed `raw_data` tensor would have had.
bool appendNarrowed(std::vector<uint8_t>& bytes,
                    const google::protobuf::RepeatedField<int32_t>& data,
                    int data_type, std::string& error) {
  for (int32_t raw : data) {
    switch (data_type) {
      case onnx::TensorProto::BOOL:
        appendLittleEndian<uint8_t>(bytes, raw != 0 ? 1u : 0u);
        break;
      case onnx::TensorProto::INT8:
      case onnx::TensorProto::UINT8:
        appendLittleEndian<uint8_t>(bytes, static_cast<uint8_t>(raw));
        break;
      case onnx::TensorProto::INT16:
      case onnx::TensorProto::UINT16:
      case onnx::TensorProto::FLOAT16:
      case onnx::TensorProto::BFLOAT16:
        appendLittleEndian<uint16_t>(bytes, static_cast<uint16_t>(raw));
        break;
      case onnx::TensorProto::INT32:
        appendLittleEndian<int32_t>(bytes, raw);
        break;
      default:
        error = "dtype " + std::to_string(data_type) +
                " cannot be carried in int32_data";
        return false;
    }
  }
  return true;
}

size_t elementByteSize(const std::string& dtype_name) {
  if (dtype_name == "float64" || dtype_name == "int64" ||
      dtype_name == "uint64" || dtype_name == "complex64")
    return 8;
  if (dtype_name == "float32" || dtype_name == "int32" ||
      dtype_name == "uint32")
    return 4;
  if (dtype_name == "float16" || dtype_name == "bfloat16" ||
      dtype_name == "int16" || dtype_name == "uint16")
    return 2;
  if (dtype_name == "int8" || dtype_name == "uint8" || dtype_name == "bool")
    return 1;
  if (dtype_name == "complex128") return 16;
  return 0;
}

}  // namespace

bool onnxDtypeName(int data_type, std::string& name) {
  switch (data_type) {
    case onnx::TensorProto::FLOAT:      name = "float32";   return true;
    case onnx::TensorProto::UINT8:      name = "uint8";     return true;
    case onnx::TensorProto::INT8:       name = "int8";      return true;
    case onnx::TensorProto::UINT16:     name = "uint16";    return true;
    case onnx::TensorProto::INT16:      name = "int16";     return true;
    case onnx::TensorProto::INT32:      name = "int32";     return true;
    case onnx::TensorProto::INT64:      name = "int64";     return true;
    case onnx::TensorProto::STRING:     name = "string";    return true;
    case onnx::TensorProto::BOOL:       name = "bool";      return true;
    case onnx::TensorProto::FLOAT16:    name = "float16";   return true;
    case onnx::TensorProto::DOUBLE:     name = "float64";   return true;
    case onnx::TensorProto::UINT32:     name = "uint32";    return true;
    case onnx::TensorProto::UINT64:     name = "uint64";    return true;
    case onnx::TensorProto::COMPLEX64:  name = "complex64"; return true;
    case onnx::TensorProto::COMPLEX128: name = "complex128";return true;
    case onnx::TensorProto::BFLOAT16:   name = "bfloat16";  return true;
    default:
      name.clear();
      return false;
  }
}

int64_t tensorElementCount(const onnx::TensorProto& tensor) {
  int64_t count = 1;
  for (int64_t dim : tensor.dims()) count *= dim;
  return count;
}

bool decodeTensorBytes(const onnx::TensorProto& tensor,
                       const std::string& dtype_name,
                       std::vector<uint8_t>& bytes, std::string& error) {
  error.clear();
  bytes.clear();

  if (tensor.data_location() == onnx::TensorProto::EXTERNAL) {
    error = "tensor data is stored in an external file (data_location=EXTERNAL)";
    return false;
  }
  if (dtype_name == "string") {
    error = "string tensors are decoded through decodeTensorStrings";
    return false;
  }

  const int64_t count = tensorElementCount(tensor);
  const size_t elem_size = elementByteSize(dtype_name);
  if (elem_size == 0) {
    error = "unsupported dtype '" + dtype_name + "' for constant decoding";
    return false;
  }

  if (!tensor.raw_data().empty()) {
    // raw_data is the path real exporters take; a payload whose size
    // disagrees with the declared dims/dtype is truncated or padded. A
    // single element's worth of bytes would otherwise be splatted silently
    // across the whole shape, and any other mismatch aborts inside
    // DenseElementsAttr.
    if (tensor.raw_data().size() != static_cast<size_t>(count) * elem_size) {
      error = "tensor data size mismatch: expected " +
              std::to_string(count * static_cast<int64_t>(elem_size)) +
              " bytes for dtype '" + dtype_name + "', got " +
              std::to_string(tensor.raw_data().size());
      return false;
    }
    bytes.assign(tensor.raw_data().begin(), tensor.raw_data().end());
    return true;
  }

  switch (tensor.data_type()) {
    case onnx::TensorProto::FLOAT:
      for (float v : tensor.float_data()) appendLittleEndian<float>(bytes, v);
      break;
    case onnx::TensorProto::DOUBLE:
      for (double v : tensor.double_data()) appendLittleEndian<double>(bytes, v);
      break;
    case onnx::TensorProto::INT64:
      for (int64_t v : tensor.int64_data()) appendLittleEndian<int64_t>(bytes, v);
      break;
    case onnx::TensorProto::UINT64:
      for (uint64_t v : tensor.uint64_data()) appendLittleEndian<uint64_t>(bytes, v);
      break;
    case onnx::TensorProto::UINT32:
      // UINT32 shares `uint64_data` with UINT64 in the ONNX schema.
      for (uint64_t v : tensor.uint64_data())
        appendLittleEndian<uint32_t>(bytes, static_cast<uint32_t>(v));
      break;
    case onnx::TensorProto::BOOL:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::UINT16:
    case onnx::TensorProto::FLOAT16:
    case onnx::TensorProto::BFLOAT16:
    case onnx::TensorProto::INT32:
      if (!appendNarrowed(bytes, tensor.int32_data(), tensor.data_type(), error))
        return false;
      break;
    default:
      error = "no typed data field for dtype '" + dtype_name + "'";
      return false;
  }

  // A zero-element tensor legitimately carries no data field at all; anything
  // else must have produced exactly the full payload.
  if (bytes.size() != static_cast<size_t>(count) * elem_size) {
    error = "tensor data size mismatch: expected " +
            std::to_string(count * static_cast<int64_t>(elem_size)) +
            " bytes for dtype '" + dtype_name + "', got " +
            std::to_string(bytes.size());
    return false;
  }
  return true;
}

bool decodeTensorStrings(const onnx::TensorProto& tensor,
                         std::vector<std::string>& values, std::string& error) {
  error.clear();
  values.clear();
  if (tensor.data_location() == onnx::TensorProto::EXTERNAL) {
    error = "string tensor data is stored in an external file";
    return false;
  }
  values.assign(tensor.string_data().begin(), tensor.string_data().end());
  const int64_t count = tensorElementCount(tensor);
  if (static_cast<int64_t>(values.size()) != count) {
    error = "string tensor value count mismatch: expected " +
            std::to_string(count) + ", got " + std::to_string(values.size());
    return false;
  }
  return true;
}

}  // namespace annc::onnx2atir
