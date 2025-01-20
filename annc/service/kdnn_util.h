#ifndef AI_COMPILER_KDNN_UTIL_H_
#define AI_COMPILER_KDNN_UTIL_H_

#include <string>
#include <vector>

#if defined(ANNC_ENABLED_KDNN)
#include "kblas.h"
#endif

namespace xla {

inline void set_tensor_shape(std::vector<int64_t>& shape) {}

}  // namespace xla

#endif  // AI_COMPILER_KDNN_UTIL_H_