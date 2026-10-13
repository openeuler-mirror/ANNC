#include "annc_ep_compile.h"

#include <unordered_set>

namespace annc_ep {

bool ValidateSite(const Site& site, std::string* error) {
  auto fail = [error](const char* message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (site.kernelName.empty()) return fail("site has no kernel name");
  if (site.pattern != "matmul") {
    return fail("site pattern is not supported by the current EP ABI");
  }
  if (site.executionMode != "aot") {
    return fail("site execution mode is not supported by the EP");
  }
  if (site.abi != "mlir_ciface") {
    return fail("site ABI is not supported by the EP");
  }
  if (site.memberNodes.empty()) return fail("site has no member nodes");

  std::unordered_set<std::string> members;
  for (const std::string& member : site.memberNodes) {
    if (member.empty()) return fail("site has an empty member node");
    if (!members.insert(member).second) {
      return fail("site member nodes contain a duplicate");
    }
  }
  if (site.flattenXIndex < -1 || site.weightIndex < -1) {
    return fail("site operand index is negative");
  }
  if (site.flattenXIndex >= 0 && site.weightIndex >= 0 &&
      site.flattenXIndex == site.weightIndex) {
    return fail("site operand indices must identify distinct operands");
  }
  return true;
}

}  // namespace annc_ep
