#include "Kernel/KernelPriorityResolver.h"
#include "Kernel/KernelRegistry.h"

namespace annc {
namespace kernels {

std::optional<std::string> resolveKernelSymbol(
    llvm::StringRef opType, llvm::StringRef backend,
    const std::vector<TypeConstraintInfo>& typeConstraints) {
    if (!typeConstraints.empty()) {
        KernelQuery query;
        query.op_type = opType.str();
        query.backend = backend.str();
        query.type_constraints = typeConstraints;
        return KernelRegistry::instance().lookupKernelSymbol(query);
    }
    return KernelRegistry::instance().lookupKernelSymbol(opType.str(), backend.str());
}

bool hasAnyAvailableKernel(const KernelResolveRequest& request,
                           bool enableKdnn) {
    auto& registry = KernelRegistry::instance();

    if (enableKdnn) {
        if (!request.requiresSpecialization()) {
            if (registry.hasKernel(request.op_type, "kdnn")) return true;
        } else {
            KernelQuery q;
            q.op_type = request.op_type;
            q.backend = "kdnn";
            q.type_constraints = request.type_constraints;
            if (registry.hasKernel(q)) return true;
        }
    }

    if (!request.requiresSpecialization()) {
        if (registry.hasKernel(request.op_type, "aarch64")) return true;
    } else {
        KernelQuery q;
        q.op_type = request.op_type;
        q.backend = "aarch64";
        q.type_constraints = request.type_constraints;
        if (registry.hasKernel(q)) return true;
    }

    return false;
}

std::optional<std::string> resolveBestKernel(const KernelResolveRequest& request,
                                              bool enableKdnn) {
    // Priority: kdnn (if enabled and exists) → aarch64 (fallback)
    if (enableKdnn) {
        if (auto sym = resolveKernelSymbol(request.op_type, "kdnn",
                                           request.type_constraints)) {
            return sym;
        }
    }
    if (auto sym = resolveKernelSymbol(request.op_type, "aarch64",
                                       request.type_constraints)) {
        return sym;
    }
    return std::nullopt;
}

} // namespace kernels
} // namespace annc
