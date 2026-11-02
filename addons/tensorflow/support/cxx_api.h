#ifndef ANNC_TENSORFLOW_SUPPORT_CXX_API_H
#define ANNC_TENSORFLOW_SUPPORT_CXX_API_H

#include <memory>

#include "c_api.h"

namespace annc {
namespace tensorflow {

// These deleters compile in the caller, while the C functions destroy objects
// in support's compilation environment. No C++ allocator crosses the boundary.
struct JitKernelDeleter {
  void operator()(AnncJitKernel* kernel) const { anncReleaseJitKernel(kernel); }
};
using JitKernelPtr = std::unique_ptr<AnncJitKernel, JitKernelDeleter>;

struct ThreadPoolAdapterDeleter {
  void operator()(AnncThreadPoolAdapter* adapter) const {
    anncDestroyThreadPoolAdapter(adapter);
  }
};
using ThreadPoolAdapterPtr =
    std::unique_ptr<AnncThreadPoolAdapter, ThreadPoolAdapterDeleter>;

}  // namespace tensorflow
}  // namespace annc

#endif  // ANNC_TENSORFLOW_SUPPORT_CXX_API_H
