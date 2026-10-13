#include "annc_ep_compile.h"

#include <unordered_set>
#include <vector>

namespace annc_ep {

namespace {

bool Fail(std::string* error, const char* message) {
  if (error != nullptr) *error = message;
  return false;
}

bool ValidateArg(const Site::Arg& arg, const char* expected_role,
                 std::string* error) {
  if (arg.name.empty() || arg.role != expected_role || arg.rank < 0 ||
      static_cast<size_t>(arg.rank) != arg.shape.size()) {
    return Fail(error,
                "site argument has invalid name, role, rank, or shape");
  }
  return true;
}

// The section of the report the EP consumes for the legacy preallocated-output
// ABI: two arguments (one rank-3 dynamic input, one rank-2 fixed weight) and
// one rank-3 output, called in the order the report's kernel_arg_order names.
bool ValidateLegacySite(const Site& site, bool matmul, std::string* error) {
  if (site.args.size() != 2 || site.outputs.size() != 1) {
    return Fail(error, "matmul site must have two arguments and one output");
  }
  const Site::Arg& dynamic = site.args[matmul ? 0 : 1];
  const Site::Arg& fixed = site.args[matmul ? 1 : 0];
  if (!ValidateArg(dynamic, "dynamic", error) ||
      !ValidateArg(fixed, "fixed", error) ||
      !ValidateArg(site.outputs[0], "output", error)) {
    return false;
  }
  if (dynamic.rank != 3 || fixed.rank != 2 || site.outputs[0].rank != 3) {
    return Fail(error, "matmul site has unsupported argument ranks");
  }
  if (site.kernelArgOrder.size() != 3) {
    return Fail(error, "site kernel argument order has the wrong size");
  }
  std::unordered_set<int64_t> order;
  for (int64_t index : site.kernelArgOrder) {
    if (index < 0 || index >= 3 || !order.insert(index).second) {
      return Fail(error, "site kernel argument order is not a permutation");
    }
  }
  const std::vector<int64_t> expected_order =
      matmul ? std::vector<int64_t>{0, 1, 2} : std::vector<int64_t>{1, 0, 2};
  if (site.kernelArgOrder != expected_order) {
    return Fail(error,
                "site kernel argument order does not match the EP call ABI");
  }
  if (site.flattenXIndex < -1 || site.weightIndex < -1) {
    return Fail(error, "site operand index is negative");
  }
  if (site.flattenXIndex >= 0 && site.weightIndex >= 0 &&
      site.flattenXIndex == site.weightIndex) {
    return Fail(error, "site operand indices must identify distinct operands");
  }
  return true;
}

// The annc_execution_v2 contract: the kernel receives the execution context
// plus its argument memrefs and allocates the outputs itself through the
// context callbacks, so the call has no output memrefs and no argument order.
// The EP resolves each argument to an ORT input by rank, which is why ranks
// must be distinct; the outputs are a slot contract (rank/type) the callbacks
// check when the kernel allocates.
bool ValidateContextSite(const Site& site, std::string* error) {
  if (!site.kernelArgOrder.empty()) {
    return Fail(error,
                "context-ABI site must not carry a kernel argument order");
  }
  if (site.args.empty() || site.args.size() > 7) {
    return Fail(error,
                "context-ABI site must have between one and seven arguments");
  }
  if (site.outputs.empty()) {
    return Fail(error, "context-ABI site must declare at least one output");
  }
  std::unordered_set<int64_t> arg_ranks;
  for (const Site::Arg& arg : site.args) {
    if (arg.name.empty() || arg.role.empty() || arg.rank < 0 || arg.rank > 4 ||
        static_cast<size_t>(arg.rank) != arg.shape.size()) {
      return Fail(
          error,
          "context-ABI site argument has invalid name, role, rank, or shape");
    }
    if (arg.dtype != "f32") {
      return Fail(error,
                  "context-ABI site argument element type is not supported");
    }
    if (!arg_ranks.insert(arg.rank).second) {
      return Fail(error, "context-ABI site arguments must have distinct ranks");
    }
  }
  for (const Site::Arg& output : site.outputs) {
    if (output.role != "output" || output.rank < 0 || output.rank > 4 ||
        static_cast<size_t>(output.rank) != output.shape.size() ||
        output.dtype != "f32") {
      return Fail(error,
                  "context-ABI site output has invalid role, rank, or type");
    }
  }
  return true;
}

}  // namespace

bool ValidateSite(const Site& site, std::string* error) {
  if (site.kernelName.empty()) return Fail(error, "site has no kernel name");
  const bool matmul = site.pattern == "matmul";
  const bool tensordot = site.pattern == "tensordot_matmul";
  if (!matmul && !tensordot) {
    return Fail(error,
                "site pattern is not supported by the current EP ABI");
  }
  if (site.executionMode != "aot") {
    return Fail(error, "site execution mode is not supported by the EP");
  }
  if (site.memberNodes.empty()) return Fail(error, "site has no member nodes");

  std::unordered_set<std::string> members;
  for (const std::string& member : site.memberNodes) {
    if (member.empty()) return Fail(error, "site has an empty member node");
    if (!members.insert(member).second) {
      return Fail(error, "site member nodes contain a duplicate");
    }
  }

  if (site.abi == "annc_execution_v2") return ValidateContextSite(site, error);
  if (site.abi != "mlir_ciface") {
    return Fail(error, "site ABI is not supported by the EP");
  }
  return ValidateLegacySite(site, matmul, error);
}

}  // namespace annc_ep
