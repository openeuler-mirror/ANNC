#include "Kernel/KernelRegistry.h"

#include <iostream>
#include <sstream>

namespace annc {
namespace kernels {

namespace detail {
void ensureBuiltinKernelRegistryLinked();
} // namespace detail

KernelRegistry& KernelRegistry::instance() {
    detail::ensureBuiltinKernelRegistryLinked();
    static KernelRegistry instance;
    return instance;
}

bool KernelRegistry::sameSignature(const KernelInfo& lhs, const KernelInfo& rhs) {
    return lhs.op_type == rhs.op_type && lhs.backend == rhs.backend &&
           lhs.type_constraints == rhs.type_constraints;
}

bool KernelRegistry::exactMatch(const KernelInfo& info, const KernelQuery& query) {
    return info.op_type == query.op_type && info.backend == query.backend &&
           info.type_constraints == query.type_constraints;
}

bool KernelRegistry::genericMatch(const KernelInfo& info, const KernelQuery& query) {
    if (info.op_type != query.op_type || info.backend != query.backend ||
        info.type_constraints.size() != query.type_constraints.size()) {
        return false;
    }

    for (const auto& requested : query.type_constraints) {
        bool matched = false;
        for (const auto& available : info.type_constraints) {
            if (available.name == requested.name && available.any_type) {
                matched = true;
                break;
            }
        }
        if (!matched) {
            return false;
        }
    }

    return true;
}

void KernelRegistry::registerKernel(KernelInfo info) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto& entries = registry_[info.op_type][info.backend];
    for (auto& entry : entries) {
        if (!sameSignature(entry.info, info)) {
            continue;
        }

        std::cerr << "[ANNC Kernel] Warning: Overwriting existing kernel '"
                  << info.op_type << "' for backend '" << info.backend << "'";
        if (info.isSpecialized()) {
            std::cerr << " with " << info.type_constraints.size() << " type constraint(s)";
        }
        std::cerr << "\n"
                  << "  Previous: " << entry.info.source_file << ":" << entry.info.line << "\n"
                  << "  New: " << info.source_file << ":" << info.line << "\n";
        entry.info = std::move(info);
        return;
    }

    std::cout << "[ANNC Kernel] Registered kernel: " << info.op_type
              << " (backend: " << info.backend << ", symbol: " << info.symbol_name;
    if (info.isSpecialized()) {
        std::cout << ", specializations: ";
        for (size_t i = 0; i < info.type_constraints.size(); ++i) {
            if (i != 0) {
                std::cout << ", ";
            }
            std::cout << info.type_constraints[i].name << "=" << info.type_constraints[i].cpp_type_name;
        }
    }
    std::cout << ")\n";

    entries.push_back(KernelEntry{std::move(info)});
}

std::optional<KernelInfo> KernelRegistry::lookupKernel(const KernelQuery& query) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto opIt = registry_.find(query.op_type);
    if (opIt == registry_.end()) {
        return std::nullopt;
    }

    auto backendIt = opIt->second.find(query.backend);
    if (backendIt == opIt->second.end()) {
        return std::nullopt;
    }

    const auto& entries = backendIt->second;

    if (!query.type_constraints.empty()) {
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
            if (exactMatch(it->info, query)) {
                return it->info;
            }
        }
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
            if (genericMatch(it->info, query)) {
                return it->info;
            }
        }
        return std::nullopt;
    }

    for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
        if (!it->info.isSpecialized()) {
            return it->info;
        }
    }

    return std::nullopt;
}

std::optional<std::string> KernelRegistry::lookupKernelSymbol(const std::string& opType,
                                                               const std::string& backend) const {
    auto info = lookupKernel(KernelQuery{opType, backend, {}});
    if (!info.has_value()) {
        return std::nullopt;
    }
    return info->symbol_name;
}

std::optional<std::string> KernelRegistry::lookupKernelSymbol(const KernelQuery& query) const {
    auto info = lookupKernel(query);
    if (!info.has_value()) {
        return std::nullopt;
    }
    return info->symbol_name;
}

bool KernelRegistry::hasKernel(const std::string& opType,
                               const std::string& backend) const {
    return lookupKernel(KernelQuery{opType, backend, {}}).has_value();
}

bool KernelRegistry::hasKernel(const KernelQuery& query) const {
    return lookupKernel(query).has_value();
}

std::vector<std::string> KernelRegistry::listKernels() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> result;
    for (const auto& opPair : registry_) {
        std::stringstream ss;
        ss << opPair.first << " [";
        bool firstBackend = true;
        for (const auto& backendPair : opPair.second) {
            if (!firstBackend) {
                ss << ", ";
            }
            size_t specializedCount = 0;
            size_t defaultCount = 0;
            for (const auto& entry : backendPair.second) {
                if (entry.info.isSpecialized()) {
                    ++specializedCount;
                } else {
                    ++defaultCount;
                }
            }
            ss << backendPair.first << ": default=" << defaultCount
               << ", specialized=" << specializedCount;
            firstBackend = false;
        }
        ss << "]";
        result.push_back(ss.str());
    }
    return result;
}

void KernelRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    registry_.clear();
}

} // namespace kernels
} // namespace annc
