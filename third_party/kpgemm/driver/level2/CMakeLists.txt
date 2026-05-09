
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})

# special defines for complex
foreach (float_type ${FLOAT_TYPES})

  if (USE_THREAD)
    GenerateNamedObjects("gemv_thread.c" "" "gemv_thread_n" false "" "" false ${float_type})
    GenerateNamedObjects("gemv_thread.c" "TRANSA" "gemv_thread_t" false "" "" false ${float_type})
  endif ()

# special defines for complex
  if (${float_type} STREQUAL "COMPLEX" OR ${float_type} STREQUAL "ZCOMPLEX")
    if (USE_THREAD)
      GenerateNamedObjects("gemv_thread.c" "CONJ" "gemv_thread_r" false "" "" false ${float_type})
      GenerateNamedObjects("gemv_thread.c" "CONJ;TRANSA" "gemv_thread_c" false "" "" false ${float_type})
      GenerateNamedObjects("gemv_thread.c" "XCONJ" "gemv_thread_o" false "" "" false ${float_type})
      GenerateNamedObjects("gemv_thread.c" "XCONJ;TRANSA" "gemv_thread_u" false "" "" false ${float_type})
      GenerateNamedObjects("gemv_thread.c" "XCONJ;CONJ" "gemv_thread_s" false "" "" false ${float_type})
      GenerateNamedObjects("gemv_thread.c" "XCONJ;CONJ;TRANSA" "gemv_thread_d" false "" "" false ${float_type})
    endif ()
  endif ()
endforeach ()

if (BUILD_BFLOAT16)
  if (USE_THREAD)
    GenerateNamedObjects("sbgemv_thread.c" "" "gemv_thread_n" false "" "" false "BFLOAT16")
    GenerateNamedObjects("sbgemv_thread.c" "TRANSA" "gemv_thread_t" false "" "" false "BFLOAT16")
  endif ()
endif ()

if ( BUILD_COMPLEX AND NOT  BUILD_SINGLE)
  if (USE_THREAD)
	  GenerateNamedObjects("gemv_thread.c" "" "gemv_thread_n" false "" "" false "SINGLE")
	  GenerateNamedObjects("gemv_thread.c" "TRANSA" "gemv_thread_t" false "" "" false "SINGLE")
  endif ()
endif ()
if ( BUILD_COMPLEX16 AND NOT  BUILD_DOUBLE)
  if (USE_THREAD)
	  GenerateNamedObjects("gemv_thread.c" "" "gemv_thread_n" false "" "" false "DOUBLE")
	  GenerateNamedObjects("gemv_thread.c" "TRANSA" "gemv_thread_t" false "" "" false "DOUBLE")
  endif ()
endif ()

add_library(driver_level2 OBJECT ${OPENBLAS_SRC})

if (USE_OPENMP)
  target_link_libraries(driver_level2 OpenMP::OpenMP_C)
endif()
