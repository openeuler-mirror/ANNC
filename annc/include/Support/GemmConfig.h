#ifndef ANNC_SUPPORT_GEMM_CONFIG_H
#define ANNC_SUPPORT_GEMM_CONFIG_H

#include <cstdlib>
#include <iostream>
#include <string_view>

#include "Support/Log.h"

namespace annc {

// Stable deployment values, not cumulative optimization levels. All combines
// epilogue fusion and RHS prepacking; Basic-only KDNN AOT outlining remains
// exclusive to Basic. New modes must not renumber existing ones.
enum class GemmMode { Off = 0, Basic = 1, Epilogue = 2, Prepack = 3, All = 99 };

struct GemmConfig {
  GemmMode mode = GemmMode::Off;
  bool epilogue = false;
  bool prepack = false;
};

// Read once, before compilation or Op execution. The TF addon snapshots
// prepack at construction and never consults this helper on same-shape hits.
inline const GemmConfig& gemmConfig() {
  static const GemmConfig config = [] {
    GemmConfig result;
    const char* value = std::getenv("ANNC_GEMM");
    std::string_view trimmed = value ? value : "";
    const auto first = trimmed.find_first_not_of(" \t\n\r\f\v");
    if (first == std::string_view::npos) return result;
    trimmed = trimmed.substr(
        first, trimmed.find_last_not_of(" \t\n\r\f\v") - first + 1);
    const char* name = "off";
    if (trimmed == "0") {
      result.mode = GemmMode::Off;
    } else if (trimmed == "1") {
      result.mode = GemmMode::Basic;
      name = "basic";
    } else if (trimmed == "2") {
      result.mode = GemmMode::Epilogue;
      result.epilogue = true;
      name = "epilogue";
    } else if (trimmed == "3") {
      result.mode = GemmMode::Prepack;
      result.prepack = true;
      name = "prepack";
    } else if (trimmed == "99") {
      result.mode = GemmMode::All;
      result.epilogue = true;
      result.prepack = true;
      name = "all";
    } else {
      // Retain the existing off fallback, but make configuration errors visible
      // even when logging is disabled. Never interpret arbitrary integers.
      std::cerr << "[annc:gemm] invalid ANNC_GEMM '" << value
                << "'; expected 0=off, 1=basic, 2=epilogue, 3=prepack, 99=all; "
                   "falling back to 0=off\n";
      return result;
    }
    ANNC_LOG_INFO("gemm") << "ANNC_GEMM=" << trimmed << " (" << name
                          << "), epilogue=" << result.epilogue
                          << ", prepack=" << result.prepack;
    return result;
  }();
  return config;
}

}  // namespace annc

#endif  // ANNC_SUPPORT_GEMM_CONFIG_H
