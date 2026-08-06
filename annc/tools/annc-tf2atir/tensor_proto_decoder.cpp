#include "tensor_proto_decoder.h"

#include <cstddef>
#include <cstring>

namespace annc::tf2atir {
namespace {

template <typename T>
void appendValue(std::vector<uint8_t>& bytes, T value) {
  const uint8_t* raw = reinterpret_cast<const uint8_t*>(&value);
  bytes.insert(bytes.end(), raw, raw + sizeof(T));
}

std::size_t elementCount(const tensorflow::TensorProto& tensor) {
  std::size_t count = 1;
  for (const auto& dim : tensor.tensor_shape().dim()) {
    if (dim.size() < 0) return 0;
    count *= static_cast<std::size_t>(dim.size());
  }
  return count;
}

template <typename T>
void padOrRepeat(std::vector<uint8_t>& bytes, std::size_t count, T last) {
  const std::size_t existing = bytes.size() / sizeof(T);
  for (std::size_t i = existing; i < count; ++i) appendValue(bytes, last);
}

}  // namespace

bool TensorProtoDecoder::decode(const tensorflow::TensorProto& tensor,
                                const std::string& dtype,
                                std::vector<uint8_t>& bytes,
                                std::string& error) {
  bytes.clear();
  error.clear();
  if (!tensor.tensor_content().empty()) {
    const std::string& content = tensor.tensor_content();
    bytes.assign(content.begin(), content.end());
    return true;
  }

  const std::size_t count = elementCount(tensor);
  if (dtype == "float32") {
    float last = 0.0F;
    for (float value : tensor.float_val()) {
      last = value;
      appendValue(bytes, value);
    }
    padOrRepeat(bytes, count, last);
    return true;
  }
  if (dtype == "float64") {
    double last = 0.0;
    for (double value : tensor.double_val()) {
      last = value;
      appendValue(bytes, value);
    }
    padOrRepeat(bytes, count, last);
    return true;
  }
  if (dtype == "float16" || dtype == "bfloat16") {
    uint16_t last = 0;
    for (int value : tensor.half_val()) {
      last = static_cast<uint16_t>(value);
      appendValue(bytes, last);
    }
    padOrRepeat(bytes, count, last);
    return true;
  }
  if (dtype == "int8") {
    int8_t last = 0;
    for (int value : tensor.int_val()) {
      last = static_cast<int8_t>(value);
      appendValue(bytes, last);
    }
    padOrRepeat(bytes, count, last);
    return true;
  }
  if (dtype == "uint8") {
    uint8_t last = 0;
    for (int value : tensor.int_val()) {
      last = static_cast<uint8_t>(value);
      appendValue(bytes, last);
    }
    padOrRepeat(bytes, count, last);
    return true;
  }
  if (dtype == "int16") {
    int16_t last = 0;
    for (int value : tensor.int_val()) {
      last = static_cast<int16_t>(value);
      appendValue(bytes, last);
    }
    padOrRepeat(bytes, count, last);
    return true;
  }
  if (dtype == "uint16") {
    uint16_t last = 0;
    for (int value : tensor.int_val()) {
      last = static_cast<uint16_t>(value);
      appendValue(bytes, last);
    }
    padOrRepeat(bytes, count, last);
    return true;
  }
  if (dtype == "int32") {
    int32_t last = 0;
    for (int value : tensor.int_val()) {
      last = value;
      appendValue(bytes, last);
    }
    padOrRepeat(bytes, count, last);
    return true;
  }
  if (dtype == "int64") {
    int64_t last = 0;
    for (int64_t value : tensor.int64_val()) {
      last = value;
      appendValue(bytes, last);
    }
    padOrRepeat(bytes, count, last);
    return true;
  }
  if (dtype == "bool") {
    uint8_t last = 0;
    for (bool value : tensor.bool_val()) {
      last = value ? 1 : 0;
      appendValue(bytes, last);
    }
    padOrRepeat(bytes, count, last);
    return true;
  }
  if (dtype == "string") {
    // The current ATIR builder treats strings as opaque values and does not
    // consume Const payload bytes. Preserve the legacy NodeInfo contract
    // until the dialect gains a representable string constant encoding.
    return true;
  }
  error = "TensorProto dtype is not supported by constant decoder: " + dtype;
  return false;
}

bool TensorProtoDecoder::decodeStrings(const tensorflow::TensorProto& tensor,
                                       std::vector<std::string>& values,
                                       std::string& error) {
  values.clear();
  error.clear();
  if (tensor.dtype() != tensorflow::DT_STRING) {
    error = "TensorProto is not DT_STRING";
    return false;
  }
  if (!tensor.tensor_content().empty()) {
    error = "DT_STRING TensorProto must use string_val, not tensor_content";
    return false;
  }

  const std::size_t count = elementCount(tensor);
  if (static_cast<std::size_t>(tensor.string_val_size()) > count) {
    error = "DT_STRING TensorProto has more string_val entries than elements";
    return false;
  }
  if (count == 0) {
    if (tensor.string_val_size() != 0) {
      error = "empty DT_STRING TensorProto has string_val entries";
      return false;
    }
    return true;
  }
  if (tensor.string_val_size() == 0) {
    error = "DT_STRING TensorProto has no string_val entries";
    return false;
  }

  values.reserve(count);
  for (const std::string& value : tensor.string_val()) values.push_back(value);
  values.resize(count, values.back());
  return true;
}

std::string base64Encode(const std::vector<uint8_t>& data) {
  static const char* characters =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  result.reserve((data.size() + 2) / 3 * 4);
  std::size_t i = 0;
  for (; i + 2 < data.size(); i += 3) {
    const uint32_t value = (static_cast<uint32_t>(data[i]) << 16) |
                           (static_cast<uint32_t>(data[i + 1]) << 8) |
                           static_cast<uint32_t>(data[i + 2]);
    result.push_back(characters[(value >> 18) & 0x3f]);
    result.push_back(characters[(value >> 12) & 0x3f]);
    result.push_back(characters[(value >> 6) & 0x3f]);
    result.push_back(characters[value & 0x3f]);
  }
  if (i < data.size()) {
    uint32_t value = static_cast<uint32_t>(data[i]) << 16;
    if (i + 1 < data.size()) value |= static_cast<uint32_t>(data[i + 1]) << 8;
    result.push_back(characters[(value >> 18) & 0x3f]);
    result.push_back(characters[(value >> 12) & 0x3f]);
    result.push_back(i + 1 < data.size() ? characters[(value >> 6) & 0x3f]
                                         : '=');
    result.push_back('=');
  }
  return result;
}

}  // namespace annc::tf2atir
