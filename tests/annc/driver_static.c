// test.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* ================================
 * MLIR memref ABI definitions 123
 * ================================ */

/* memref<1536x1344xf32>, memref<1344x1152xf32>, memref<1536x1152xf32> */
typedef struct {
    float   *allocated;
    float   *aligned;
    int64_t  offset;
    int64_t  sizes[2];
    int64_t  strides[2];
} memref_2d_f32;

/* memref<1152xf32> */
typedef struct {
    float   *allocated;
    float   *aligned;
    int64_t  offset;
    int64_t  sizes[1];
    int64_t  strides[1];
} memref_1d_f32;

/* ================================
 * MLIR generated C interface
 * ================================ */

// 
typedef void (*fused_matmul_func_t) (
    memref_2d_f32 *ret,
    memref_2d_f32 *A,
    memref_2d_f32 *B,
    memref_2d_f32 *C,
    memref_1d_f32 *bias
);

// 
static const char* get_library_name() {
    const char* lib_name = getenv("ANNC_LIBRARY_NAME");
    return lib_name ? lib_name : "kpgemm";
}

// 
static fused_matmul_func_t get_fused_matmul_func() {
    printf(" _mlir_ciface_fused_matmul ...\n");
    
    //  lib_name  .so 
    char func_name[256];
    const char* lib_name = get_library_name();
    strncpy(func_name, lib_name, sizeof(func_name) - 1);
    func_name[sizeof(func_name) - 1] = '\0';
    
    //  ./ 
    if (strncmp(func_name, "./", 2) == 0) {
        memmove(func_name, func_name + 2, strlen(func_name + 2) + 1);
    }
    
    //  .so 
    char *dot_so = strstr(func_name, ".so");
    if (dot_so != NULL) {
        *dot_so = '\0';
    }
    
    // _mlir_ciface_<>
    char full_func_name[300];
    snprintf(full_func_name, sizeof(full_func_name), "_mlir_ciface_%s", func_name);
    
    // dlsym
    // 
    // 
    return (fused_matmul_func_t)0x1; // NULL
}

// 
extern void _mlir_ciface_fused_matmul_add_relu_A0732AD9DB33D09F(
    memref_2d_f32 *ret,
    memref_2d_f32 *A,
    memref_2d_f32 *B,
    memref_2d_f32 *C,
    memref_1d_f32 *bias
);

// 
static void fused_matmul_wrapper(
    memref_2d_f32 *ret,
    memref_2d_f32 *A,
    memref_2d_f32 *B,
    memref_2d_f32 *C,
    memref_1d_f32 *bias
) {
    _mlir_ciface_fused_matmul_add_relu_A0732AD9DB33D09F(ret, A, B, C, bias);
}

/* ================================
 * Static dimensions (must match MLIR)
 * ================================ */

//  -DM= -DK= -DN= 
#ifndef M
#define M 1536
#endif
#ifndef K
#define K 1344
#endif
#ifndef N
#define N 1152
#endif

/* ================================
 * Utility
 * ================================ */

static void *aligned_malloc(size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) {
        return NULL;
    }
    return ptr;
}

/* ================================
 * Main
 * ================================ */

int main(void) {
    printf("[test] allocating buffers...\n");

    float *A_buf    = aligned_malloc((size_t)M * K * sizeof(float));
    float *B_buf    = aligned_malloc((size_t)K * N * sizeof(float));
    float *C_buf    = aligned_malloc((size_t)M * N * sizeof(float));
    float *bias_buf = aligned_malloc((size_t)N * sizeof(float));

    if (!A_buf || !B_buf || !C_buf || !bias_buf) {
        fprintf(stderr, "[test] allocation failed\n");
        return 1;
    }

    /* Initialize inputs */
    for (int i = 0; i < M * K; ++i)
        A_buf[i] = 1.0f;

    for (int i = 0; i < K * N; ++i)
        B_buf[i] = 1.0f;

    for (int i = 0; i < M * N; ++i)
        C_buf[i] = 0.0f;

    for (int i = 0; i < N; ++i)
        bias_buf[i] = 0.1f;

    /* ================================
     * Construct memref descriptors
     * ================================ */

    memref_2d_f32 A = {
        .allocated = A_buf,
        .aligned   = A_buf,
        .offset    = 0,
        .sizes     = { M, K },
        .strides   = { K, 1 }
    };

    memref_2d_f32 B = {
        .allocated = B_buf,
        .aligned   = B_buf,
        .offset    = 0,
        .sizes     = { K, N },
        .strides   = { N, 1 }
    };

    memref_2d_f32 C = {
        .allocated = C_buf,
        .aligned   = C_buf,
        .offset    = 0,
        .sizes     = { M, N },
        .strides   = { N, 1 }
    };

    memref_1d_f32 bias = {
        .allocated = bias_buf,
        .aligned   = bias_buf,
        .offset    = 0,
        .sizes     = { N },
        .strides   = { 1 }
    };

    printf("[test] calling _mlir_ciface_fused_matmul...\n");

    memref_2d_f32 ret = C; //  alloc
    fused_matmul_wrapper(&ret, &A, &B, &C, &bias);

    /* ================================
     * Verify result
     *
     * C[i][j] = relu(K + bias[j])
     * ================================ */

    float expected = (float)K + 0.1f;
    int errors = 0;

    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, C_buf[i]);
        if (fabsf(C_buf[i] - expected) > 1e-3f)
            errors++;
    }

    if (errors == 0)
        printf(" Result looks correct!\n");
    else
        printf(" Found %d errors\n", errors);

    free(A_buf);
    free(B_buf);
    free(C_buf);
    free(bias_buf);

    return errors ? 1 : 0;
}
