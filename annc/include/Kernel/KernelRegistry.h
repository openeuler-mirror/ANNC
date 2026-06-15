#ifndef ANNC_KERNEL_REGISTRY_H
#define ANNC_KERNEL_REGISTRY_H

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace annc {
namespace kernels {

struct TypeConstraintInfo {
    std::string name;
    std::string cpp_type_name;
    bool any_type = false;

    bool operator==(const TypeConstraintInfo& other) const noexcept {
        return name == other.name && cpp_type_name == other.cpp_type_name &&
               any_type == other.any_type;
    }
};

struct KernelInfo {
    std::string op_type;
    std::string backend;
    std::string symbol_name;
    std::string source_file;
    int line = 0;
    std::vector<TypeConstraintInfo> type_constraints;

    bool isSpecialized() const noexcept { return !type_constraints.empty(); }
};

struct KernelQuery {
    std::string op_type;
    std::string backend = "aarch64";
    std::vector<TypeConstraintInfo> type_constraints;
};

namespace detail {

template <typename T>
inline std::string cppTypeName() {
    using Decayed = std::decay_t<T>;
    if constexpr (std::is_same_v<Decayed, bool>) {
        return "bool";
    } else if constexpr (std::is_same_v<Decayed, std::int8_t>) {
        return "int8_t";
    } else if constexpr (std::is_same_v<Decayed, std::int16_t>) {
        return "int16_t";
    } else if constexpr (std::is_same_v<Decayed, std::int32_t>) {
        return "int32_t";
    } else if constexpr (std::is_same_v<Decayed, std::int64_t>) {
        return "int64_t";
    } else if constexpr (std::is_same_v<Decayed, std::uint8_t>) {
        return "uint8_t";
    } else if constexpr (std::is_same_v<Decayed, std::uint16_t>) {
        return "uint16_t";
    } else if constexpr (std::is_same_v<Decayed, std::uint32_t>) {
        return "uint32_t";
    } else if constexpr (std::is_same_v<Decayed, std::uint64_t>) {
        return "uint64_t";
    } else if constexpr (std::is_same_v<Decayed, float>) {
        return "float";
    } else if constexpr (std::is_same_v<Decayed, double>) {
        return "double";
    } else {
        return typeid(Decayed).name();
    }
}

} // namespace detail

class KernelBuilder {
public:
    explicit KernelBuilder(std::string opType) : op_type_(std::move(opType)) {}

    KernelBuilder& Backend(std::string backend) {
        backend_ = std::move(backend);
        return *this;
    }

    template <typename T>
    KernelBuilder& TypeConstraint(const char* name) {
        type_constraints_.push_back(TypeConstraintInfo{name, detail::cppTypeName<T>()});
        return *this;
    }

    KernelBuilder& AnyType(const char* name) {
        type_constraints_.push_back(TypeConstraintInfo{name, "*", true});
        return *this;
    }

    KernelInfo Build(std::string symbolName, const char* sourceFile, int line) const {
        KernelInfo info;
        info.op_type = op_type_;
        info.backend = backend_;
        info.symbol_name = std::move(symbolName);
        info.source_file = sourceFile;
        info.line = line;
        info.type_constraints = type_constraints_;
        return info;
    }

private:
    std::string op_type_;
    std::string backend_ = "aarch64";
    std::vector<TypeConstraintInfo> type_constraints_;
};

inline KernelBuilder Name(const char* opType) {
    return KernelBuilder(opType);
}

class KernelRegistry {
public:
    static KernelRegistry& instance();

    void registerKernel(KernelInfo info);

    std::optional<KernelInfo> lookupKernel(const KernelQuery& query) const;

    std::optional<std::string> lookupKernelSymbol(const std::string& opType,
                                                  const std::string& backend = "aarch64") const;

    std::optional<std::string> lookupKernelSymbol(const KernelQuery& query) const;

    bool hasKernel(const std::string& opType,
                   const std::string& backend = "aarch64") const;

    bool hasKernel(const KernelQuery& query) const;

    std::vector<std::string> listKernels() const;

    void clear();

private:
    KernelRegistry() = default;
    KernelRegistry(const KernelRegistry&) = delete;
    KernelRegistry& operator=(const KernelRegistry&) = delete;

    struct KernelEntry {
        KernelInfo info;
    };

    static bool sameSignature(const KernelInfo& lhs, const KernelInfo& rhs);
    static bool exactMatch(const KernelInfo& info, const KernelQuery& query);
    static bool genericMatch(const KernelInfo& info, const KernelQuery& query);

    mutable std::mutex mutex_;
    std::map<std::string, std::map<std::string, std::vector<KernelEntry>>> registry_;
};

} // namespace kernels
} // namespace annc

#endif // ANNC_KERNEL_REGISTRY_H
