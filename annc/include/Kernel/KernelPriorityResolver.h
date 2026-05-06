#ifndef ANNC_KERNEL_PRIORITY_RESOLVER_H
#define ANNC_KERNEL_PRIORITY_RESOLVER_H

#include <optional>
#include <string>
#include <vector>

#include "Kernel/KernelRegistry.h"
#include "llvm/ADT/StringRef.h"

namespace annc {
namespace kernels {

/// Request structure for kernel resolution, supporting optional type
/// constraints for specialization.
struct KernelResolveRequest {
    std::string op_type;
    std::vector<TypeConstraintInfo> type_constraints;

    bool requiresSpecialization() const { return !type_constraints.empty(); }
};

/// Resolve a kernel symbol for the given op_type, backend, and optional
/// type_constraints. This is the low-level lookup that does not consider
/// priority chains — it queries a single backend directly.
std::optional<std::string> resolveKernelSymbol(
    llvm::StringRef opType, llvm::StringRef backend,
    const std::vector<TypeConstraintInfo>& typeConstraints = {});

/// Check whether any kernel is available for the given request under the
/// specified kdnn configuration.
/// This is the single source of truth for "can this op be lowered to a
/// builtin kernel?" — used by CustomFusionPatternBase to decide whether
/// a pattern should fire.
bool hasAnyAvailableKernel(const KernelResolveRequest& request,
                           bool enableKdnn);

/// Resolve the best available kernel symbol for the given request under the
/// specified kdnn configuration.
/// Priority: kdnn (if enabled and kernel exists) → aarch64 (fallback).
/// Returns std::nullopt if no kernel is available for any backend.
std::optional<std::string> resolveBestKernel(const KernelResolveRequest& request,
                                              bool enableKdnn);

} // namespace kernels
} // namespace annc

#endif // ANNC_KERNEL_PRIORITY_RESOLVER_H
