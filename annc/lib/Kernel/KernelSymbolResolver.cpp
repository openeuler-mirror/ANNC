#include "Kernel/KernelSymbolResolver.h"

namespace annc {
namespace kernels {

std::optional<std::string> ResolveKernelSymbol(const KernelSymbolResolverRequest& request) {
    if (request.requiresSpecialization()) {
        KernelQuery query;
        query.op_type = request.op_type;
        query.backend = request.backend;
        query.type_constraints = request.type_constraints;

        return KernelRegistry::instance().lookupKernelSymbol(query);
    }

    return KernelRegistry::instance().lookupKernelSymbol(request.op_type, request.backend);
}

} // namespace kernels
} // namespace annc
