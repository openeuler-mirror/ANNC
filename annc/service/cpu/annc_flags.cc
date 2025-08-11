#include "annc_flags.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace annc {
static const char kWS[] = " \t\r\n";  // whitespace

static const std::vector<Flag> g_default_flags = {
    {"matmul", "Register matmul operator.", false, GEMM_OPT},
    {"batch_matmul", "Register batch_matmul operator", false, GEMM_OPT},
    {"matmul_add", "Register matmul_add operator", false, GEMM_OPT},
    {"matmul_add_relu", "Register matmul_add_relu operator", false, GEMM_OPT},

    {"sps_emd_2", "Enable sparse_embedding2 fusion", false, GRAPH_OPT},
    {"pooling", "Enable pooling fustion", false, GRAPH_OPT},

    {"disable_tf_matmul_fusion", "Disable matmul fusion in TF", false, TF_OPT},
    {"enable_annc_pass", "Enable annc optimizer pass", true, TF_OPT},
};

static const Flag kGraphOptFlag = {
    "enable_graph_opt", "Enable all graph optimizations", false, GRAPH_OPT};
static const Flag kGemmOptFlag = {
    "enable_gemm_opt", "Enable all gemm optimizations", false, GEMM_OPT};

static size_t skip_whitespace(const std::string& s, size_t pos) {
  while (pos < s.size() && s[pos] == ' ') {
    ++pos;
  }
  return pos;
}

ANNCFlags::ANNCFlags() {
  for (const auto& flag : g_default_flags) {
    flag_values_[flag.name] = flag.default_value;
  }
  parse_from_env();
}

void ANNCFlags::parse_from_env() {
  const char* env = getenv("ANNC_FLAGS");
  if (env == nullptr || env[0] == '\0') {
    return;
  }

  std::string opts(env);
  parse_flags_from_string(opts);
}

void ANNCFlags::parse_flags_from_string(const std::string& str) {
  size_t pos = 0;
  const size_t n = str.size();

  while (pos < n) {
    pos = skip_whitespace(str, pos);
    if (pos >= n) break;

    if (str.substr(pos, 2) != "--") {
      std::cerr << "[ANNC ERROR] Invalid flag syntax: expected '--' at position"
                << pos << " in: " << str << std::endl;
      break;
    }
    pos += 2;

    size_t key_start = pos;
    size_t eq_pos = std::string::npos;

    while (pos < n && str[pos] != '=' && str[pos] != ' ') ++pos;

    if (pos < n && str[pos] == '=') eq_pos = pos;
    std::string key = str.substr(
        key_start, (eq_pos != std::string::npos ? eq_pos : pos) - key_start);

    std::string value = "true";
    if (eq_pos != std::string::npos) {
      size_t val_start = eq_pos + 1;
      size_t val_end = val_start;
      while (val_end < n && str[val_end] != ' ') ++val_end;
      value = str.substr(val_start, val_end - val_start);
      pos = val_end;
    } else {
      pos = skip_whitespace(str, pos);
    }

    if (key == kGraphOptFlag.name) {
      for (const auto& flag : g_default_flags) {
        if (flag.type == kGraphOptFlag.type) {
          flag_values_[flag.name] = true;
        }
      }
      continue;
    }

    if (key == kGemmOptFlag.name) {
      for (const auto& flag : g_default_flags) {
        if (flag.type == kGemmOptFlag.type) {
          flag_values_[flag.name] = true;
        }
      }
      continue;
    }

    if (flag_values_.find(key) == flag_values_.end()) {
      std::cerr << "[ANNC ERROR] Unknown flag: '" << key
                << "'. Valid flags are:";
      for (const auto& f : g_default_flags) {
        std::cerr << f.name << std::endl;
      }
    }

    bool enable = (value == "true" || value == "1" || value == "");
    flag_values_[key] = enable;
    if (enable) {
      std::cout << "enabled flag : --" << key << std::endl;
    }
  }
}

bool ANNCFlags::is_enabled(const std::string& name) const {
  auto it = flag_values_.find(name);
  return it != flag_values_.end() && it->second;
}

ANNCFlags& get_annc_flags() {
  static ANNCFlags flags;
  return flags;
}
}  // namespace annc
