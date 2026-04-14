#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Kernel/KernelSymbolResolver.h"

using namespace annc::kernels;

namespace {

struct CustomCallSymbolRequest {
    std::string op_type;
    std::string backend = "aarch64";
    std::vector<TypeConstraintInfo> type_constraints;
};

std::string resolveForCustomCallOrThrow(const CustomCallSymbolRequest& request) {
    KernelSymbolResolverRequest resolverRequest;
    resolverRequest.op_type = request.op_type;
    resolverRequest.backend = request.backend;
    resolverRequest.type_constraints = request.type_constraints;

    auto symbol = ResolveKernelSymbol(resolverRequest);
    if (!symbol.has_value()) {
        throw std::runtime_error("failed to resolve symbol for op: " + request.op_type);
    }
    return *symbol;
}

void expectHasPrefix(const std::string& value, const std::string& prefix, const char* what) {
    if (value.rfind(prefix, 0) != 0) {
        throw std::runtime_error(std::string("symbol prefix mismatch for ") + what + ": " + value);
    }
}

void expectHasSuffix(const std::string& value, const std::string& suffix, const char* what) {
    if (value.size() < suffix.size() ||
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) != 0) {
        throw std::runtime_error(std::string("symbol suffix mismatch for ") + what + ": " + value);
    }
}

} // namespace

int main() {
    std::cout << "=== CustomCall Symbol Resolution Stub ===\n";

    CustomCallSymbolRequest matmul;
    matmul.op_type = "MatMul";
    matmul.backend = "aarch64";
    const std::string matmulSymbol = resolveForCustomCallOrThrow(matmul);
    expectHasPrefix(matmulSymbol, "ANNCKernel_matmul_", "MatMul default");
    expectHasSuffix(matmulSymbol, "_auto", "MatMul default");
    std::cout << "default lookup: MatMul -> " << matmulSymbol << "\n";

    try {
        CustomCallSymbolRequest missing;
        missing.op_type = "DefinitelyMissingOp";
        missing.backend = "aarch64";
        (void)resolveForCustomCallOrThrow(missing);
        throw std::runtime_error("missing op unexpectedly resolved");
    } catch (const std::runtime_error&) {
        std::cout << "missing lookup: DefinitelyMissingOp -> <error>\n";
    }

    constexpr int kIterations = 200000;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        const std::string symbol = resolveForCustomCallOrThrow(matmul);
        if (symbol.empty()) {
            throw std::runtime_error("matmul lookup failed during benchmark");
        }
    }
    auto end = std::chrono::steady_clock::now();
    const auto totalNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    const double avgNs = static_cast<double>(totalNs) / static_cast<double>(kIterations);
    std::cout << "resolveForCustomCallOrThrow(MatMul) average: " << avgNs
              << " ns over " << kIterations << " iterations\n";

    std::cout << "Stub result: PASS\n";
    return 0;
}
