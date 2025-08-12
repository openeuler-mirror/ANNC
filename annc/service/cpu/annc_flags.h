#ifndef ANNC_FLAGS_H_
#define ANNC_FLAGS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace annc {

enum OptType {
  GRAPH_OPT,
  GEMM_OPT,
  TF_OPT,
  NONE
};

struct Flag {
  std::string name;
  std::string description;
  bool default_value;
  OptType type;
};

class ANNCFlags {
 public:
  ANNCFlags();
  virtual ~ANNCFlags() = default;

  void parse_from_env();

  bool is_enabled(const std::string& flag_name) const;

 private:
  void parse_flags_from_string(const std::string& str);
  std::unordered_map<std::string, bool> flag_values_;
};

ANNCFlags& get_annc_flags();

}  // namespace annc
#endif
