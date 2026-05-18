#ifndef ANNC_KERNEL_SYMBOL_RESOLVER_H
#define ANNC_KERNEL_SYMBOL_RESOLVER_H

#include <optional>
#include <string>
#include <vector>

#include "Kernel/KernelRegistry.h"

namespace annc {
namespace kernels {

struct KernelSymbolResolverRequest {
    std::string op_type;
    std::string backend = "aarch64";
    std::vector<TypeConstraintInfo> type_constraints;

    bool requiresSpecialization() const { return !type_constraints.empty(); }
};

std::optional<std::string> ResolveKernelSymbol(const KernelSymbolResolverRequest& request);

} // namespace kernels
} // namespace annc

#endif // ANNC_KERNEL_SYMBOL_RESOLVER_H
