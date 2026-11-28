// Intentionally no include guard: this header is designed to be included
// multiple times with different ANNC_BUILTIN_KERNEL_SPECS_FILE values.

#include "Kernel/KernelRegistry.h"
#include "Kernel/KernelAPIMacros.h"
#include "Kernel/KernelStatus.h"
#include "Context/ThreadPool.h"

#ifndef ANNC_BUILTIN_KERNEL_SPECS_FILE
#error "ANNC_BUILTIN_KERNEL_SPECS_FILE must be defined before including BuiltinKernelSpecDeclare.h"
#endif

using ::annc::kernels::Name;

#include "Context/KernelContext.h"
#include "Kernel/ExecutionContext.h"

#define ANNC_KERNEL_SPEC(spec_token, builder_expr, args_decl, body) \
    EXTERN_C std::int32_t ANNC_AUTO_KERNEL_C_API(spec_token)(ANNC_UNPAREN args_decl);
#define ANNC_CONTEXT_KERNEL_SPEC(spec_token, builder_expr, body) \
    EXTERN_C AnncStatusCode ANNC_AUTO_KERNEL_C_API(spec_token)(AnncKernelContext* context);
#include ANNC_BUILTIN_KERNEL_SPECS_FILE
#undef ANNC_CONTEXT_KERNEL_SPEC
#undef ANNC_KERNEL_SPEC
#undef ANNC_BUILTIN_KERNEL_SPECS_FILE
