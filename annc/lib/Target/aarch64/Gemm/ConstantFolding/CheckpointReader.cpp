#include "CheckpointReader.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include "llvm/Support/raw_ostream.h"
namespace fs = std::filesystem;
namespace annc {
namespace constant_folding {
namespace {
constexpr uint64_t kTableMagic = 0xdb4775248b80fb57ULL;
constexpr size_t kFooterSize = 48;
constexpr size_t kBlockTrailerSize = 5;
bool decodeVarint(const uint8_t *buf, size_t len, size_t &pos, uint64_t &out) {
  out = 0;
  for (int shift = 0; pos < len && shift < 64; shift += 7) {
    const uint8_t byte = buf[pos++];
    out |= static_cast<uint64_t>(byte & 0x7f) << shift;
    if (!(byte & 0x80)) return true;
  }
  return false;
}

bool skipField(const uint8_t *buf, size_t len, size_t &pos, int wireType) {
  uint64_t scratch = 0;
  switch (wireType) {
  case 0:
    return decodeVarint(buf, len, pos, scratch);
  case 1:
  case 5: {
    const size_t width = wireType == 1 ? 8 : 4;
    if (len - pos < width) return false;
    pos += width;
    return true;
  }
  case 2:
    if (!decodeVarint(buf, len, pos, scratch) || scratch > len - pos)
      return false;
    pos += static_cast<size_t>(scratch);
    return true;
  default:
    return false;
  }
}

struct BundleEntry {
  int32_t dtype = 0;
  int32_t shard_id = 0;
  int64_t offset = 0;
  int64_t size = 0;
};
bool parseBundleEntry(const uint8_t *buf, size_t len, BundleEntry &entry) {
  size_t pos = 0;
  while (pos < len) {
    uint64_t tag = 0;
    if (!decodeVarint(buf, len, pos, tag)) return false;
    const uint64_t field = tag >> 3;
    const int wireType = static_cast<int>(tag & 7);
    uint64_t value = 0;
    if (field >= 1 && field <= 5 && field != 2) {
      if (wireType != 0 || !decodeVarint(buf, len, pos, value)) return false;
      if (field == 1) entry.dtype = static_cast<int32_t>(value);
      if (field == 3) entry.shard_id = static_cast<int32_t>(value);
      if (field == 4) entry.offset = static_cast<int64_t>(value);
      if (field == 5) entry.size = static_cast<int64_t>(value);
    } else if (!skipField(buf, len, pos, wireType)) {
      return false;
    }
  }
  return true;
}

bool nextBlockEntry(const uint8_t *block, size_t end, size_t &pos,
                    std::string &lastKey, std::string &key,
                    const uint8_t *&value, uint64_t &valueLen) {
  uint64_t shared = 0, unshared = 0, len = 0;
  if (!decodeVarint(block, end, pos, shared) ||
      !decodeVarint(block, end, pos, unshared) ||
      !decodeVarint(block, end, pos, len) || shared > lastKey.size() ||
      unshared > end - pos || len > end - pos - unshared)
    return false;
  key = lastKey.substr(0, static_cast<size_t>(shared));
  key.append(reinterpret_cast<const char *>(&block[pos]),
             static_cast<size_t>(unshared));
  pos += static_cast<size_t>(unshared);
  value = &block[pos];
  valueLen = len;
  pos += static_cast<size_t>(len);
  lastKey = key;
  return true;
}
bool blockRangeOk(const std::vector<uint8_t> &data, uint64_t offset,
                  uint64_t size) {
  return offset <= data.size() && size <= data.size() - offset &&
         kBlockTrailerSize <= data.size() - offset - size &&
         data[static_cast<size_t>(offset + size)] == 0;
}

}  // namespace

std::string CheckpointReader::dataFilePath(int32_t shard_id) const {
  char suffix[48];
  std::snprintf(suffix, sizeof(suffix), ".data-%05d-of-%05d", shard_id,
                num_shards_);
  return data_prefix_ + suffix;
}

std::optional<CheckpointReader> CheckpointReader::open(
    const std::string &dir) {
  std::string index_path;
  for (const char *suffix :
       {"variables/variables.index", "1/variables/variables.index",
        "variables.index"}) {
    const std::string candidate = (fs::path(dir) / suffix).string();
    if (fs::exists(candidate)) {
      index_path = candidate;
      break;
    }
  }
  if (index_path.empty()) {
    llvm::errs() << "[CheckpointReader] variables.index not found under " << dir
                 << "\n";
    return std::nullopt;
  }
  CheckpointReader reader;
  reader.data_prefix_ =
      (fs::path(index_path).parent_path() / "variables").string();
  if (!reader.parseIndexFile(index_path)) {
    llvm::errs() << "[CheckpointReader] failed to parse " << index_path
                 << "\n";
    return std::nullopt;
  }
  return reader;
}

bool CheckpointReader::parseIndexFile(const std::string &index_path) {
  std::ifstream file(index_path, std::ios::binary);
  if (!file) return false;
  const std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
  if (data.size() < kFooterSize) return false;

  uint64_t magic = 0;
  std::memcpy(&magic, &data[data.size() - 8], 8);
  if (magic != kTableMagic) return false;

  const uint8_t *footer = &data[data.size() - kFooterSize];
  size_t pos = 0;
  uint64_t handles[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i)
    if (!decodeVarint(footer, kFooterSize, pos, handles[i]))
      return false;
  const uint64_t idxOffset = handles[2];
  const uint64_t idxSize = handles[3];
  if (!blockRangeOk(data, idxOffset, idxSize)) return false;

  const uint8_t *index = &data[static_cast<size_t>(idxOffset)];
  const size_t indexLen = static_cast<size_t>(idxSize);
  size_t ipos = 0;
  std::string lastKey;
  while (ipos < indexLen) {
    std::string key;
    const uint8_t *handle = nullptr;
    uint64_t handleLen = 0;
    if (!nextBlockEntry(index, indexLen, ipos, lastKey, key, handle, handleLen))
      break;
    size_t hpos = 0;
    uint64_t blockOffset = 0, blockSize = 0;
    if (decodeVarint(handle, handleLen, hpos, blockOffset) &&
        decodeVarint(handle, handleLen, hpos, blockSize) &&
        blockRangeOk(data, blockOffset, blockSize))
      parseDataBlock(data, blockOffset, blockSize);
  }
  auto header = entries_.find("");
  if (header != entries_.end()) {
    num_shards_ = header->second.dtype;
    entries_.erase(header);
  }
  if (num_shards_ <= 0) num_shards_ = 1;
  return !entries_.empty();
}

void CheckpointReader::parseDataBlock(const std::vector<uint8_t> &file_data,
                                      uint64_t offset, uint64_t size) {
  if (offset > file_data.size() || size > file_data.size() - offset) return;
  const uint8_t *block = &file_data[static_cast<size_t>(offset)];
  const size_t blockLen = static_cast<size_t>(size);
  if (blockLen < 4) return;

  uint32_t numRestarts = 0;
  std::memcpy(&numRestarts, &block[blockLen - 4], 4);
  if (numRestarts > (blockLen - 4) / sizeof(uint32_t)) return;
  const size_t end =
      blockLen - 4 - static_cast<size_t>(numRestarts) * sizeof(uint32_t);

  size_t pos = 0;
  std::string lastKey;
  while (pos < end) {
    std::string key;
    const uint8_t *value = nullptr;
    uint64_t valueLen = 0;
    if (!nextBlockEntry(block, end, pos, lastKey, key, value, valueLen)) break;

    if (key.empty()) {
      BundleEntry header;
      if (parseBundleEntry(value, valueLen, header))
        entries_[""] = Entry{header.dtype, 0, 0, 0};
      continue;
    }
    BundleEntry proto;
    if (!parseBundleEntry(value, valueLen, proto)) continue;
    if (proto.shard_id < 0 || proto.offset < 0 || proto.size < 0) continue;
    entries_[key] = Entry{proto.dtype, proto.shard_id, proto.offset,
                          proto.size};
  }
}

bool CheckpointReader::readTensor(const std::string &var_name,
                                  std::vector<uint8_t> &out_data,
                                  int32_t &out_dtype) const {
  auto it = entries_.find(var_name);
  if (it == entries_.end()) return false;
  const Entry &entry = it->second;
  if (entry.shard_id < 0 || entry.shard_id >= num_shards_) return false;
  std::ifstream file(dataFilePath(entry.shard_id), std::ios::binary);
  if (!file) return false;
  file.seekg(static_cast<std::streamoff>(entry.offset));
  out_data.resize(static_cast<size_t>(entry.size));
  file.read(reinterpret_cast<char *>(out_data.data()),
            static_cast<std::streamsize>(entry.size));
  if (file.gcount() != entry.size) {
    llvm::errs() << "[CheckpointReader] short read for variable " << var_name
                 << "\n";
    return false;
  }
  out_dtype = entry.dtype;
  return true;
}

}  // namespace constant_folding
}  // namespace annc
