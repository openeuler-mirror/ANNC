#ifndef ANNC_CBLAS_H
#define ANNC_CBLAS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */

#define OPENBLAS_CONST const

#define CBLAS_INDEX size_t

typedef enum CBLAS_ORDER {
  CblasRowMajor = 101,
  CblasColMajor = 102
} CBLAS_ORDER;
typedef CBLAS_ORDER CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113,
  CblasConjNoTrans = 114
} CBLAS_TRANSPOSE;

typedef int blasint;

void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order,
                 OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
                 OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha,
                 OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
                 OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb,
                 OPENBLAS_CONST float beta, float *C,
                 OPENBLAS_CONST blasint ldc);

float cblas_sasum(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x,
                  OPENBLAS_CONST blasint incx);

#ifdef __cplusplus
}
#endif  /* __cplusplus */

#endif  // ANNC_CBLAS_H