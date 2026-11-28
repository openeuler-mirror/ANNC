#ifndef ANNC_TESTS_AUTO_KERNEL_SUPPORT_H
#define ANNC_TESTS_AUTO_KERNEL_SUPPORT_H

#include "Kernel/MemRefTypes.h"

extern int g_auto_spec_f32_calls;
extern int g_auto_spec_i32_calls;

void auto_default_impl(AnncMemRef1DF32* output, AnncMemRef1DF32* input);
void auto_spec_f32_impl();
void auto_spec_i32_impl();

#endif
