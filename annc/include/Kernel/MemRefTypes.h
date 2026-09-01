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

typedef struct {
    int32_t* allocated;
    int32_t* aligned;
    int64_t  offset;
    int64_t  sizes[1];
    int64_t  strides[1];
} AnncMemRef1DI32;

typedef struct {
    int64_t* allocated;
    int64_t* aligned;
    int64_t  offset;
    int64_t  sizes[1];
    int64_t  strides[1];
} AnncMemRef1DI64;

typedef struct {
    int64_t* allocated;
    int64_t* aligned;
    int64_t  offset;
    int64_t  sizes[2];
    int64_t  strides[2];
} AnncMemRef2DI64;

// 以下类型为 812 融合移植所需(A 侧独有,0-D 布局与 MLIR C interface
// memref<...> 一致:{allocated, aligned, offset},无 sizes/strides 数组)。
typedef struct {
    float*   allocated;
    float*   aligned;
    int64_t  offset;
    int64_t  sizes[3];
    int64_t  strides[3];
} AnncMemRef3DF32;

typedef struct {
    int32_t* allocated;
    int32_t* aligned;
    int64_t  offset;
    int64_t  sizes[2];
    int64_t  strides[2];
} AnncMemRef2DI32;

typedef struct {
    int32_t* allocated;
    int32_t* aligned;
    int64_t  offset;
} AnncMemRef0DI32;

typedef struct {
    float*   allocated;
    float*   aligned;
    int64_t  offset;
} AnncMemRef0DF32;

typedef struct {
    int64_t* allocated;
    int64_t* aligned;
    int64_t  offset;
} AnncMemRef0DI64;

#define ANNC_MEMREF_DATA(memref) ((memref).aligned + (memref).offset)

#define ANNC_MEMREF_SIZE_1D(memref) ((memref).sizes[0])

#define ANNC_MEMREF_SIZE_2D(memref) ((memref).sizes[0] * (memref).sizes[1])

#ifdef __cplusplus
}
#endif

#endif
