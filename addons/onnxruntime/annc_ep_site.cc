#include "annc_ep_compile.h"

#include <unordered_set>
#include <vector>

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
  if (site.args.size() != 2 || site.outputs.size() != 1) {
    return fail("matmul site must have two arguments and one output");
  }
  auto validateArg = [error](const Site::Arg& arg, const char* expected_role) {
    if (arg.name.empty() || arg.role != expected_role || arg.rank < 0 ||
        static_cast<size_t>(arg.rank) != arg.shape.size()) {
      if (error != nullptr) *error = "site argument has invalid name, role, rank, or shape";
      return false;
    }
    return true;
  };
  if (!validateArg(site.args[0], "dynamic") ||
      !validateArg(site.args[1], "fixed") ||
      !validateArg(site.outputs[0], "output")) {
    return false;
  }
  if (site.args[0].rank != 3 || site.args[1].rank != 2 ||
      site.outputs[0].rank != 3) {
    return fail("matmul site has unsupported argument ranks");
  }
  if (site.kernelArgOrder.size() != 3) {
    return fail("site kernel argument order has the wrong size");
  }
  std::unordered_set<int64_t> order;
  for (int64_t index : site.kernelArgOrder) {
    if (index < 0 || index >= 3 || !order.insert(index).second) {
      return fail("site kernel argument order is not a permutation");
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
