#include "Kernel/KernelPriorityResolver.h"
#include "Kernel/KernelRegistry.h"

namespace annc {
namespace kernels {
namespace {

std::optional<KernelInfo> lookupKernelForBackend(
    const std::string& opType, const std::string& backend,
    const std::vector<TypeConstraintInfo>& typeConstraints) {
    KernelQuery query;
    query.op_type = opType;
    query.backend = backend;
    query.type_constraints = typeConstraints;
    return KernelRegistry::instance().lookupKernel(query);
}

} // namespace

bool hasAnyAvailableKernel(const KernelResolveRequest& request,
                           bool enableKdnn) {
    auto& registry = KernelRegistry::instance();

    if (enableKdnn && request.hasPackedRhsFormat()) {
#ifdef ANNC_ENABLE_CONSTANT_FOLDING
        if (!request.requiresSpecialization()) {
            if (registry.hasKernel(request.op_type, "kdnn_packed")) return true;
        } else {
            KernelQuery q;
            q.op_type = request.op_type;
            q.backend = "kdnn_packed";
            q.type_constraints = request.type_constraints;
            if (registry.hasKernel(q)) return true;
        }
#endif
    }

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

std::optional<KernelInfo> resolveBestKernelInfo(
    const KernelResolveRequest& request, bool enableKdnn) {
    if (enableKdnn && request.hasPackedRhsFormat()) {
#ifdef ANNC_ENABLE_CONSTANT_FOLDING
        if (auto info = lookupKernelForBackend(request.op_type, "kdnn_packed",
                                               request.type_constraints)) {
            return info;
        }
#endif
    }

    if (enableKdnn) {
        if (auto info = lookupKernelForBackend(request.op_type, "kdnn",
                                               request.type_constraints)) {
            return info;
        }
    }
    if (auto info = lookupKernelForBackend(request.op_type, "aarch64",
                                           request.type_constraints)) {
        return info;
    }
    return std::nullopt;
}

} // namespace kernels
} // namespace annc
