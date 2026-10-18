// FusionSwitch.h — runtime switches for fused-pattern enablement.
//
// The same variables gate both layers:
//   * the compiler's OpFusion pass skips the disabled matcher, so the fused
//     operator is not generated at all;
//   * the ORT EP's site loop keeps a claim-time filter as a safety net.
#ifndef ANNC_SUPPORT_FUSIONSWITCH_H
#define ANNC_SUPPORT_FUSIONSWITCH_H

#include <cstdlib>
#include <cstring>
#include <string_view>

namespace annc {

// Environment variable controlling one fused pattern, or nullptr for patterns
// without a switch (including "matmul").
inline const char* fusionSwitchName(std::string_view pattern) {
  if (pattern == "tensordot_matmul") return "ANNC_EP_FUSION_TENSORDOT_MATMUL";
  if (pattern == "layer_norm") return "ANNC_EP_FUSION_LAYER_NORM";
  if (pattern == "masked_mul") return "ANNC_EP_FUSION_MASKED_MUL";
  if (pattern == "batched_matmul_add")
    return "ANNC_EP_FUSION_BATCHED_MATMUL_ADD";
  return nullptr;
}

// "0" disables the pattern; unset or any other value keeps it enabled.
inline bool fusionPatternEnabled(std::string_view pattern) {
  const char* name = fusionSwitchName(pattern);
  if (name == nullptr) return true;
  const char* value = std::getenv(name);
  return value == nullptr || std::strcmp(value, "0") != 0;
}

}  // namespace annc

#endif  // ANNC_SUPPORT_FUSIONSWITCH_H
