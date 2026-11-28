#include "common/pb2_common.h"

#include "tensorflow/compiler/mlir/tensorflow/utils/convert_attr.h"

#include <absl/strings/escaping.h>

#include <cmath>
#include <iomanip>
#include <limits>

namespace {
// Lossless tensor encoding for the JSON channel: base64 of the TensorProto.
void AppendTensorEncoded(const tensorflow::TensorProto& tp,
                         std::ostringstream& o) {
  o << "{\"__tensor__\":\"" << absl::Base64Escape(tp.SerializeAsString())
    << "\"}";
}

// Shape JSON encoding: unknown_rank must be explicit (otherwise it is
// indistinguishable from a scalar rank-0 {"dim":[]}).
void AppendShapeJson(const tensorflow::TensorShapeProto& sh,
                     std::ostringstream& o) {
  if (sh.unknown_rank()) {
    o << "{\"unknown_rank\":true}";
    return;
  }
  o << "{\"dim\":[";
  for (int i = 0; i < sh.dim_size(); ++i)
    o << (i ? "," : "") << sh.dim(i).size();
  o << "]}";
}
}  // namespace

mlir::Attribute AttrToMlir(const AttrValue& v, mlir::Builder& b) {
  auto r = tensorflow::ConvertNonFuncAttributeValue(v, &b);
  if (!r.ok()) {
    std::cerr << "[warn] attr conversion failed: " << r.status().ToString()
              << "\n";
    return {};
  }
  return *r;
}

// JSON string escape. Bytes >= 0x80 pass through unchanged: attr strings
// are expected to be UTF-8 (vocabularies included); a non-UTF-8 byte string
// would produce invalid JSON and the consumer's whole custom_attrs parse
// would fail loudly there. Frozen round-trip protocol; do not silently
// re-encode.
std::string JsonEscape(const std::string& s) {
  std::string out;
  char buf[8];
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if ((unsigned char)c < 0x20) {
          snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

// int64 output as a JSON number (host-side C++ parsing; not JS).
void AppendI64Json(int64_t v, std::ostringstream& o) { o << v; }
void AppendF32Json(float v, std::ostringstream& o) {
  o << std::setprecision(std::numeric_limits<float>::max_digits10) << v;
  if (!std::isfinite(v)) {
    std::cerr << "[warn] non-finite float attr in custom_attrs (round-trip "
                 "degrades to null); value = " << v << "\n";
    o.str("");
    o << "null";
  }
}
void AppendBoolJson(bool v, std::ostringstream& o) {
  o << (v ? "true" : "false");
}
void AppendStringJson(const std::string& s, std::ostringstream& o) {
  o << '"' << JsonEscape(s) << '"';
}
void AppendTypeJson(DataType t, std::ostringstream& o) {
  const std::string name = DataType_Name(t);
  if (!name.empty()) {
    o << '"' << name << '"';
    return;
  }
  o << static_cast<int64_t>(t);
}

template <typename Container, typename AppendElem>
void AppendJsonArray(const Container& c, AppendElem appendElem,
                     std::ostringstream& o) {
  o << "[";
  bool first = true;
  for (const auto& x : c) {
    if (!first) o << ",";
    first = false;
    appendElem(x, o);
  }
  o << "]";
}

std::string AttrValueToJson(const AttrValue& v) {
  std::ostringstream o;
  switch (v.value_case()) {
    case AttrValue::kType:
      AppendTypeJson(v.type(), o);
      break;
    case AttrValue::kI:
      AppendI64Json(v.i(), o);
      break;
    case AttrValue::kF:
      AppendF32Json(v.f(), o);
      break;
    case AttrValue::kB:
      AppendBoolJson(v.b(), o);
      break;
    case AttrValue::kS:
      AppendStringJson(v.s(), o);
      break;
    case AttrValue::kShape:
      AppendShapeJson(v.shape(), o);
      break;
    case AttrValue::kTensor:
      AppendTensorEncoded(v.tensor(), o);
      break;
    case AttrValue::kList: {
      const auto& l = v.list();
      if (l.tensor_size() > 0) {
        AppendJsonArray(
            l.tensor(),
            [](const tensorflow::TensorProto& tp, std::ostringstream& o) {
              AppendTensorEncoded(tp, o);
            },
            o);
        break;
      }
      // Element order contract: protobuf sublists are concatenated per
      // type (i, then f, s, type, b, shape), never interleaved as in the
      // source; consumers must read it as a per-type concatenation. The
      // per-segment format itself is owned by the per-type appenders.
      llvm::SmallVector<std::string, 8> parts;
      auto collect = [&parts](std::ostringstream& o) {
        parts.push_back(o.str());
      };
      std::ostringstream elem;
      for (int64_t x : l.i()) { elem.str(""); AppendI64Json(x, elem); collect(elem); }
      for (float f : l.f()) { elem.str(""); AppendF32Json(f, elem); collect(elem); }
      for (const auto& st : l.s()) { elem.str(""); AppendStringJson(st, elem); collect(elem); }
      for (int t : l.type()) {
        elem.str("");
        AppendTypeJson(static_cast<DataType>(t), elem);
        collect(elem);
      }
      for (bool bb : l.b()) { elem.str(""); AppendBoolJson(bb, elem); collect(elem); }
      for (const auto& sh : l.shape()) { elem.str(""); AppendShapeJson(sh, elem); collect(elem); }
      o << "[" << absl::StrJoin(parts, ",") << "]";
      break;
    }
    default:
      // func/placeholder attrs cannot ride this channel (emit null).
      // Contract: control-flow ops with func attrs go through
      // control_flow_carrier -- this branch is leak detection only. null
      // makes the host report a missing attribute (detectable) instead of
      // a silently wrong value.
      std::cerr << "[warn] unrepresentable attr (func/placeholder) in "
                   "custom_attrs degrades to null\n";
      o << "null";
      break;
  }
  return o.str();
}
