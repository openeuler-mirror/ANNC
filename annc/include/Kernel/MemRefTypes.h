#ifndef ANNC_MEMREF_TYPES_H
#define ANNC_MEMREF_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void*    allocated;
    void*    aligned;
    int64_t  offset;
    int64_t  sizes[8];
    int64_t  strides[8];
    int32_t  rank;
} AnncMemRef;

typedef struct {
    float*   allocated;
    float*   aligned;
    int64_t  offset;
    int64_t  sizes[1];
    int64_t  strides[1];
} AnncMemRef1DF32;

typedef struct {
    const char* data;
    int64_t     size;
} AnncStringRef;

typedef struct {
    AnncStringRef* allocated;
    AnncStringRef* aligned;
    int64_t        offset;
    int64_t        sizes[1];
    int64_t        strides[1];
} AnncMemRef1DString;

typedef struct {
    float*   allocated;
    float*   aligned;
    int64_t  offset;
    int64_t  sizes[2];
    int64_t  strides[2];
} AnncMemRef2DF32;

#define ANNC_MEMREF_DATA(memref) ((memref).aligned + (memref).offset)

#define ANNC_MEMREF_SIZE_1D(memref) ((memref).sizes[0])

#define ANNC_MEMREF_SIZE_2D(memref) ((memref).sizes[0] * (memref).sizes[1])

#ifdef __cplusplus
}
#endif

#endif
