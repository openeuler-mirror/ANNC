// Intentionally no include guard: this header is designed to be included
// multiple times with different ANNC_BUILTIN_KERNEL_SPECS_FILE values.

#include "Kernel/KernelRegistry.h"
#include "Kernel/KernelAPIMacros.h"
#include "Kernel/threadpool/ThreadPoolContext.h"

#ifndef ANNC_BUILTIN_KERNEL_SPECS_FILE
#error "ANNC_BUILTIN_KERNEL_SPECS_FILE must be defined before including BuiltinKernelSpecDefine.h"
#endif

using ::annc::kernels::Name;

#define ANNC_KERNEL_SPEC(spec_token, builder_expr, args_decl, body) \
    DEFINE_KERNEL(void, ANNC_AUTO_KERNEL_SYMBOL(spec_token), ANNC_UNPAREN args_decl) body
#include ANNC_BUILTIN_KERNEL_SPECS_FILE
#undef ANNC_KERNEL_SPEC
#undef ANNC_BUILTIN_KERNEL_SPECS_FILE
