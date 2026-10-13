// annc_ep_graph_export.cc — OrtGraph -> ONNX model export and fingerprinting
// for the EP's runtime-compile mode (see annc_ep_compile.h).
//
// The serialization itself is the ORT-provided helper (vendored verbatim in
// ort_graph_to_proto.h, two includes retargeted); everything else here is the
// naming discipline that makes the exchange with the ANNC toolchain exact, and
// the hashing the disk cache keys on.
#define ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL
#include "ort_graph_to_proto.h"

#include "annc_ep_compile.h"

#include <cstring>
#include <fstream>
#include <sstream>

namespace annc_ep {
namespace {

// --- SHA-256 -----------------------------------------------------------------
// Self-contained, so the EP plugin keeps its current dependency set (ORT
// headers + libdl) and does not drag an OpenSSL link into the host process.
struct Sha256 {
  uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  uint64_t bit_count = 0;
  uint8_t block[64] = {};
  size_t block_len = 0;

  static uint32_t Rotate(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32 - bits));
  }

  void Compress(const uint8_t* chunk) {
    static const uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
        0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
        0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
        0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
        0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
        0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<uint32_t>(chunk[i * 4]) << 24) |
             (static_cast<uint32_t>(chunk[i * 4 + 1]) << 16) |
             (static_cast<uint32_t>(chunk[i * 4 + 2]) << 8) |
             static_cast<uint32_t>(chunk[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = Rotate(w[i - 15], 7) ^ Rotate(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = Rotate(w[i - 2], 17) ^ Rotate(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = Rotate(e, 6) ^ Rotate(e, 11) ^ Rotate(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = h + s1 + ch + k[i] + w[i];
      const uint32_t s0 = Rotate(a, 2) ^ Rotate(a, 13) ^ Rotate(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
  }

  void Update(const void* data, size_t len) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    bit_count += static_cast<uint64_t>(len) * 8;
    while (len > 0) {
      const size_t take = (len < 64 - block_len) ? len : 64 - block_len;
      std::memcpy(block + block_len, bytes, take);
      block_len += take;
      bytes += take;
      len -= take;
      if (block_len == 64) {
        Compress(block);
        block_len = 0;
      }
    }
  }

  std::string FinalHex() {
    const uint64_t bits = bit_count;
    const uint8_t padding = 0x80;
    Update(&padding, 1);
    const uint8_t zero = 0;
    while (block_len != 56) Update(&zero, 1);
    uint8_t length_be[8];
    for (int i = 0; i < 8; ++i) {
      length_be[i] = static_cast<uint8_t>(bits >> (56 - i * 8));
    }
    // Update() would count these bytes into bit_count; write them directly.
    std::memcpy(block + block_len, length_be, 8);
    Compress(block);
    block_len = 0;

    static const char* digits = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint32_t word : state) {
      for (int shift = 28; shift >= 0; shift -= 4) {
        out.push_back(digits[(word >> shift) & 0xf]);
      }
    }
    return out;
  }
};

}  // namespace

std::string Sha256String(const std::string& data) {
  Sha256 hash;
  hash.Update(data.data(), data.size());
  return hash.FinalHex();
}

std::string Sha256File(const std::string& path, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    if (error) *error = "cannot open '" + path + "'";
    return {};
  }
  Sha256 hash;
  std::vector<char> buffer(1 << 20);
  while (in.good()) {
    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize got = in.gcount();
    if (got > 0) hash.Update(buffer.data(), static_cast<size_t>(got));
  }
  return hash.FinalHex();
}

bool ExportGraphToModel(const OrtGraph* graph, const std::string& modelPath,
                        NameMaps* maps, std::string* error) {
  if (graph == nullptr) {
    *error = "null graph";
    return false;
  }
  onnx::ModelProto model;
  Ort::Status status = OrtEpUtils::OrtGraphToProto(*graph, model);
  if (!status.IsOK()) {
    *error = std::string("OrtGraphToProto failed: ") + status.GetErrorMessage();
    return false;
  }
  onnx::GraphProto* proto = model.mutable_graph();
  if (proto == nullptr) {
    *error = "exported model has no graph";
    return false;
  }

  // Pass 1: tensor renames. Node i's p-th result becomes "t<i>_<p>"; anything
  // the graph takes from outside becomes in_/w_/v_ by position. One map serves
  // every reference site, so the graph stays consistent by construction.
  std::unordered_map<std::string, std::string> rename;
  for (int i = 0; i < proto->node_size(); ++i) {
    const onnx::NodeProto& node = proto->node(i);
    for (int p = 0; p < node.output_size(); ++p) {
      const std::string& name = node.output(p);
      if (!name.empty()) rename[name] = "t" + std::to_string(i) + "_" + std::to_string(p);
    }
  }
  for (int k = 0; k < proto->input_size(); ++k) {
    const std::string& name = proto->input(k).name();
    if (!name.empty() && rename.find(name) == rename.end()) {
      rename[name] = "in_" + std::to_string(k);
    }
  }
  for (int k = 0; k < proto->initializer_size(); ++k) {
    const std::string& name = proto->initializer(k).name();
    if (!name.empty() && rename.find(name) == rename.end()) {
      rename[name] = "w_" + std::to_string(k);
    }
  }
  for (int k = 0; k < proto->value_info_size(); ++k) {
    const std::string& name = proto->value_info(k).name();
    if (!name.empty() && rename.find(name) == rename.end()) {
      rename[name] = "v_" + std::to_string(k);
    }
  }

  auto renamed = [&rename](const std::string& name) -> std::string {
    if (name.empty()) return name;
    auto it = rename.find(name);
    return it == rename.end() ? name : it->second;
  };

  // Pass 2: node and value names.
  maps->node_names.clear();
  maps->node_names.reserve(static_cast<size_t>(proto->node_size()));
  for (int i = 0; i < proto->node_size(); ++i) {
    onnx::NodeProto* node = proto->mutable_node(i);
    const std::string node_name = "n" + std::to_string(i);
    node->set_name(node_name);
    maps->node_names.push_back(node_name);
    maps->node_index_by_name[node_name] = i;
    for (int j = 0; j < node->input_size(); ++j) {
      node->set_input(j, renamed(node->input(j)));
    }
    for (int j = 0; j < node->output_size(); ++j) {
      node->set_output(j, renamed(node->output(j)));
    }
  }
  for (int k = 0; k < proto->input_size(); ++k) {
    proto->mutable_input(k)->set_name(renamed(proto->input(k).name()));
  }
  for (int k = 0; k < proto->output_size(); ++k) {
    proto->mutable_output(k)->set_name(renamed(proto->output(k).name()));
  }
  for (int k = 0; k < proto->value_info_size(); ++k) {
    proto->mutable_value_info(k)->set_name(renamed(proto->value_info(k).name()));
  }
  for (int k = 0; k < proto->initializer_size(); ++k) {
    proto->mutable_initializer(k)->set_name(renamed(proto->initializer(k).name()));
  }

  maps->tensor_to_renamed.clear();
  maps->renamed_to_tensor.clear();
  for (const auto& entry : rename) {
    maps->tensor_to_renamed[entry.first] = entry.second;
    maps->renamed_to_tensor[entry.second] = entry.first;
  }

  std::ofstream out(modelPath, std::ios::binary);
  if (!out.is_open() || !model.SerializeToOstream(&out)) {
    *error = "cannot write '" + modelPath + "'";
    return false;
  }
  return true;
}

std::string ModelSignature(const std::string& modelPath,
                           const std::string& toolchainFingerprint,
                           std::string* error) {
  onnx::ModelProto model;
  {
    std::ifstream in(modelPath, std::ios::binary);
    if (!in.is_open() || !model.ParseFromIstream(&in)) {
      if (error) *error = "cannot parse '" + modelPath + "'";
      return {};
    }
  }

  // Drop everything that does not decide what ANNC emits: producer metadata
  // varies per export, and initializer payloads are runtime arguments of the
  // kernels this pipeline generates (the FTDMM matcher only checks that the
  // weight is a rank-2 f32 constant with a static shape).
  model.clear_producer_name();
  model.clear_producer_version();
  model.clear_doc_string();
  model.clear_domain();
  model.clear_metadata_props();
  if (model.has_graph()) {
    model.mutable_graph()->clear_name();
    model.mutable_graph()->clear_doc_string();
    for (int i = 0; i < model.graph().initializer_size(); ++i) {
      onnx::TensorProto* initializer = model.mutable_graph()->mutable_initializer(i);
      initializer->clear_raw_data();
      initializer->clear_double_data();
      initializer->clear_float_data();
      initializer->clear_int32_data();
      initializer->clear_int64_data();
      initializer->clear_string_data();
      initializer->clear_external_data();
      initializer->set_data_location(onnx::TensorProto::DEFAULT);
    }
  }

  std::string serialized;
  if (!model.SerializeToString(&serialized)) {
    if (error) *error = "cannot serialize the model for signing";
    return {};
  }
  Sha256 hash;
  const std::string prefix = "annc-ep-cache-key-v1\n";
  hash.Update(prefix.data(), prefix.size());
  hash.Update(toolchainFingerprint.data(), toolchainFingerprint.size());
  const char separator = '\n';
  hash.Update(&separator, 1);
  hash.Update(serialized.data(), serialized.size());
  return hash.FinalHex();
}

}  // namespace annc_ep
