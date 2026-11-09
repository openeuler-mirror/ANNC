#include "ciface_dispatch.h"

#include <array>
#include <iostream>
#include <utility>
#include <vector>

namespace {

int calls;
std::vector<void*> received;

template <size_t>
using Pointer = void*;

template <size_t... I>
void Capture(Pointer<I>... args) {
  ++calls;
  received = {args...};
}

template <size_t... I>
bool Check(std::index_sequence<I...>, bool accepted) {
  constexpr size_t count = sizeof...(I);
  std::array<int, count> values{};
  std::vector<void*> args;
  for (size_t i = 0; i < count; ++i) args.push_back(&values[i]);
  calls = 0;
  received.clear();
  bool ok = annc::ciface::Call(reinterpret_cast<void*>(&Capture<I...>), args);
  if (ok == accepted && calls == (accepted ? 1 : 0) &&
      (!accepted || args == received))
    return true;
  std::cerr << "Incorrect dispatch for " << count << " arguments\n";
  return false;
}

}  // namespace

int main() {
  return Check(std::make_index_sequence<0>{}, true) &&
                 Check(std::make_index_sequence<10>{}, true) &&
                 Check(std::make_index_sequence<64>{}, true) &&
                 Check(std::make_index_sequence<65>{}, false)
             ? 0
             : 1;
}
