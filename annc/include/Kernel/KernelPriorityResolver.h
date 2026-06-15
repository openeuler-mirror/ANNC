#ifndef ANNC_KERNEL_PRIORITY_RESOLVER_H
#define ANNC_KERNEL_PRIORITY_RESOLVER_H

#include <optional>
#include <string>
#include <vector>

#include "Kernel/KernelRegistry.h"

namespace annc {
namespace kernels {

/// Request structure for kernel resolution, supporting optional type
/// constraints for specialization.
struct KernelResolveRequest {
    std::string op_type;
    std::vector<TypeConstraintInfo> type_constraints;

    bool requiresSpecialization() const { return !type_constraints.empty(); }
};

/// Check whether any kernel is available for the given request under the
/// specified kdnn configuration.
/// This is the single source of truth for "can this op be lowered to a
/// builtin kernel?" — used by CustomFusionPatternBase to decide whether
/// a pattern should fire.
bool hasAnyAvailableKernel(const KernelResolveRequest& request,
                           bool enableKdnn);

/// Resolve the best available kernel info for the given request under the
/// specified kdnn configuration.
/// Priority: kdnn (if enabled and kernel exists) → aarch64 (fallback).
/// Returns std::nullopt if no kernel is available for any backend.
std::optional<KernelInfo> resolveBestKernelInfo(const KernelResolveRequest& request,
                                                bool enableKdnn);

} // namespace kernels
} // namespace annc

#endif // ANNC_KERNEL_PRIORITY_RESOLVER_H
