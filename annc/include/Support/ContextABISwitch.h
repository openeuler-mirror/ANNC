#ifndef ANNC_SUPPORT_CONTEXT_ABI_SWITCH_H
#define ANNC_SUPPORT_CONTEXT_ABI_SWITCH_H

#include <cstdlib>
#include <cstring>

namespace annc {

// Temporary opt-in switch for the new context ABI lowering route.
inline bool loweringContextABIEnabled() {
  const char *value = std::getenv("ANNC_ENABLE_LOWERING_CONTEXT_ABI");
  return value && (std::strcmp(value, "1") == 0 ||
                   std::strcmp(value, "true") == 0 ||
                   std::strcmp(value, "on") == 0 ||
                   std::strcmp(value, "yes") == 0);
}

}  // namespace annc

#endif  // ANNC_SUPPORT_CONTEXT_ABI_SWITCH_H
