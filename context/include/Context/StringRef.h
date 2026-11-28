#ifndef ANNC_CONTEXT_STRING_REF_H
#define ANNC_CONTEXT_STRING_REF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed bytes. The caller owns the storage for the duration of the call.
typedef struct {
  const char* data;
  int64_t size;
} AnncStringRef;

#ifdef __cplusplus
}
#endif

#endif
