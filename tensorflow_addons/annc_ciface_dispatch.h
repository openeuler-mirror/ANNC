// mlir_ciface argument dispatch, kept independent of TensorFlow so it can be
// unit-tested without loading the addon.
#ifndef ANNC_TENSORFLOW_ADDONS_ANNC_CIFACE_DISPATCH_H
#define ANNC_TENSORFLOW_ADDONS_ANNC_CIFACE_DISPATCH_H

#include <cstddef>
#include <utility>
#include <vector>

namespace annc {
namespace ciface {

// ANNCFused kernels are ordinary mlir_ciface functions whose signature is a
// sequence of memref descriptors (all passed as pointers).  The runtime knows
// only the argument count, so it casts the dlsym() handle to the matching
// function-pointer type and invokes it.  The old implementation stopped at 8
// arguments; bst_mmoe (10) and din_mmoe (14) need more.  Keep a generous cap so
// a malformed model fails with a clear error instead of an unbounded template
// instantiation.
constexpr std::size_t kMaxArgs = 64;

template <std::size_t... I>
void Invoke(void* func, const std::vector<void*>& args,
            std::index_sequence<I...>) {
  using CifaceFn = void (*)(decltype(I, static_cast<void*>(nullptr))...);
  reinterpret_cast<CifaceFn>(func)(args[I]...);
}

// Returns true when the argument count matches N and the function was called.
template <std::size_t N>
bool Dispatch(void* func, const std::vector<void*>& args) {
  if (args.size() == N) {
    Invoke(func, args, std::make_index_sequence<N>{});
    return true;
  }
  if constexpr (N > 0) {
    return Dispatch<N - 1>(func, args);
  }
  return false;
}

// Returns false when args.size() exceeds kMaxArgs; the caller maps that to an
// Unimplemented status.
inline bool Call(void* func, const std::vector<void*>& args) {
  if (args.size() > kMaxArgs) return false;
  return Dispatch<kMaxArgs>(func, args);
}

}  // namespace ciface
}  // namespace annc

#endif  // ANNC_TENSORFLOW_ADDONS_ANNC_CIFACE_DISPATCH_H
