// Intentionally no include guard: this header is designed to be included
// multiple times with different ANNC_BUILTIN_KERNEL_SPECS_FILE values.

#include "Kernel/KernelRegistry.h"
#include "Kernel/KernelMacros.h"

#ifndef ANNC_BUILTIN_KERNEL_SPECS_FILE
#error "ANNC_BUILTIN_KERNEL_SPECS_FILE must be defined before including BuiltinKernelSpecRegister.h"
#endif

using ::annc::kernels::Name;

#define ANNC_KERNEL_SPEC(spec_token, builder_expr, args_decl, body) \
    ANNC_REGISTER_AUTO_KERNEL_SPEC(spec_token, builder_expr)
#include ANNC_BUILTIN_KERNEL_SPECS_FILE
#undef ANNC_KERNEL_SPEC
#undef ANNC_BUILTIN_KERNEL_SPECS_FILE
