#ifndef ANNC_KPGEMM_H
#define ANNC_KPGEMM_H
#include <stdio.h>

#include <algorithm>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define OPENBLAS_CONST const

#define KP_INDEX size_t

typedef enum KP_ORDER { KpRowMajor = 101, KpColMajor = 102 } KP_ORDER;
typedef KP_ORDER KP_LAYOUT;
typedef enum KP_TRANSPOSE {
  KpNoTrans = 111,
  KpTrans = 112,
  KpConjTrans = 113,
  KpConjNoTrans = 114
} KP_TRANSPOSE;

typedef int blasint;

void kp_sgemm(OPENBLAS_CONST enum KP_ORDER Order,
              OPENBLAS_CONST enum KP_TRANSPOSE TransA,
              OPENBLAS_CONST enum KP_TRANSPOSE TransB, OPENBLAS_CONST blasint M,
              OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
              OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A,
              OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B,
              OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C,
              OPENBLAS_CONST blasint ldc);

float kp_sasum(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x,
               OPENBLAS_CONST blasint incx);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
