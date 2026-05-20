#ifndef ANNC_KERNEL_API_MACROS_H
#define ANNC_KERNEL_API_MACROS_H

#define ANNC_CONCAT_IMPL(a, b) a##b
#define ANNC_CONCAT(a, b) ANNC_CONCAT_IMPL(a, b)
#define ANNC_CONCAT3_IMPL(a, b, c) a##b##c
#define ANNC_CONCAT3(a, b, c) ANNC_CONCAT3_IMPL(a, b, c)
#define ANNC_CONCAT4_IMPL(a, b, c, d) a##b##c##d
#define ANNC_CONCAT4(a, b, c, d) ANNC_CONCAT4_IMPL(a, b, c, d)
#define ANNC_STRINGIZE_IMPL(x) #x
#define ANNC_STRINGIZE(x) ANNC_STRINGIZE_IMPL(x)

#define ANNC_UNPAREN(...) __VA_ARGS__

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

#define API_PREFIX _mlir_ciface_
#define API_CONCAT(a, b) API_CONCAT_IMPL(a, b)
#define API_CONCAT_IMPL(a, b) a##b

#define DEFINE_KERNEL(ret_type, name, ...) \
EXTERN_C ret_type API_CONCAT(API_PREFIX, name)(__VA_ARGS__)

#ifndef ANNC_SPEC_TOKEN_PREFIX
#define ANNC_SPEC_TOKEN_PREFIX Spec
#endif

#define ANNC_INTERNAL_SPEC_TOKEN(line) ANNC_CONCAT3(ANNC_SPEC_TOKEN_PREFIX, _L, line)
#define ANNC_AUTO_KERNEL_SYMBOL(spec_token) ANNC_CONCAT3(ANNCKernel_, spec_token, _auto)
#define ANNC_AUTO_KERNEL_SYMBOL_STRING(spec_token) ANNC_STRINGIZE(ANNC_AUTO_KERNEL_SYMBOL(spec_token))
#define ANNC_AUTO_KERNEL_C_API(spec_token) API_CONCAT(API_PREFIX, ANNC_AUTO_KERNEL_SYMBOL(spec_token))

#define ANNC_KERNEL(builder_expr, ...) \
    ANNC_KERNEL_SPEC(ANNC_INTERNAL_SPEC_TOKEN(__LINE__), builder_expr, __VA_ARGS__)

#endif // ANNC_KERNEL_API_MACROS_H
