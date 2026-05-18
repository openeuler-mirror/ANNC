#include <cstdint>

#include "Kernel/MemRefTypes.h"
#include "tests/kernels/auto_kernel_support.h"

int g_auto_spec_f32_calls = 0;
int g_auto_spec_i32_calls = 0;

void auto_default_impl(AnncMemRef1DF32* output, AnncMemRef1DF32* input) {
    float* out = ANNC_MEMREF_DATA(*output);
    const float* in = ANNC_MEMREF_DATA(*input);
    const int64_t size = ANNC_MEMREF_SIZE_1D(*input);
    for (int64_t i = 0; i < size; ++i) {
        out[i] = in[i] * 3.0f;
    }
}

void auto_spec_f32_impl() {
    ++g_auto_spec_f32_calls;
}

void auto_spec_i32_impl() {
    ++g_auto_spec_i32_calls;
}
