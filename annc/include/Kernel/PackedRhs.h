#ifndef ANNC_KERNEL_PACKED_RHS_H
#define ANNC_KERNEL_PACKED_RHS_H
#include <stdint.h>
// Private JIT ABI: original row-major F32 RHS, pc-jc order.
// ISA 0 = NEON packed-b-v1; ISA 1 = SVE packed-b-v2.
// argument is the function argument index before execution ABI lowering.
typedef struct {
  int64_t version, argument, k, n, kc, nc, isa, panel, elements, alignment;
} AnncPackedRhsSpec;
typedef const AnncPackedRhsSpec* (*AnncGetPackedRhsSpecs)(int64_t* count);
typedef void (*AnncPreparePackedRhs)(int64_t index, const float* source,
                                     float* destination);
#endif
