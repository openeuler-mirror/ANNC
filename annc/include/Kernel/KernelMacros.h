#ifndef ANNC_KERNEL_MACROS_H
#define ANNC_KERNEL_MACROS_H

#include <utility>

#include "Kernel/KernelAPIMacros.h"
#include "Kernel/KernelRegistry.h"

#define ANNC_REGISTER_KERNEL_BUILDER_IMPL(builder_expr, kernel_symbol, unique_id) \
    namespace { \
        struct ANNC_CONCAT(KernelRegistrar_, unique_id) { \
            ANNC_CONCAT(KernelRegistrar_, unique_id)() { \
                auto info = (builder_expr).Build(kernel_symbol, __FILE__, __LINE__); \
                ::annc::kernels::KernelRegistry::instance().registerKernel(std::move(info)); \
            } \
        } ANNC_CONCAT(kernel_registrar_, unique_id); \
    }

#define ANNC_REGISTER_KERNEL_BUILDER(builder_expr, kernel_symbol) \
    ANNC_REGISTER_KERNEL_BUILDER_IMPL(builder_expr, kernel_symbol, __COUNTER__)

#define ANNC_REGISTER_AUTO_KERNEL_SPEC(spec_token, builder_expr) \
    ANNC_REGISTER_KERNEL_BUILDER(builder_expr, ANNC_AUTO_KERNEL_SYMBOL_STRING(spec_token))

#endif // ANNC_KERNEL_MACROS_H
