#ifndef ANNC_CONSTANT_FOLDING_CHECKPOINT_READER_H
#define ANNC_CONSTANT_FOLDING_CHECKPOINT_READER_H

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace annc {
namespace constant_folding {

class CheckpointReader {
public:
  // Opens <dir>/variables/variables.index, <dir>/1/variables/variables.index
  // or <dir>/variables.index.
  static std::optional<CheckpointReader> open(const std::string &dir);

  bool readTensor(const std::string &var_name, std::vector<uint8_t> &out_data,
                  int32_t &out_dtype) const;
private:
  struct Entry {
    int32_t dtype = 0;
    int32_t shard_id = 0;
    int64_t offset = 0;
    int64_t size = 0;
  };

  std::unordered_map<std::string, Entry> entries_;
  std::string data_prefix_;
  int32_t num_shards_ = 0;

  bool parseIndexFile(const std::string &index_path);
  void parseDataBlock(const std::vector<uint8_t> &file_data, uint64_t offset,
                      uint64_t size);
  std::string dataFilePath(int32_t shard_id) const;
};

} // namespace constant_folding
} // namespace annc

#endif // ANNC_CONSTANT_FOLDING_CHECKPOINT_READER_H
