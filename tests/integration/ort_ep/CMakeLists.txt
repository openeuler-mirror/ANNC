# L2 integration / e2e tests for the ONNX Runtime EP.
#
# These are Python drivers registered with CTest here; they exercise the built
# libannc_ep.so together with the ONNX pipeline and the builtin kernels.  The
# plugin itself is defined in addons/onnxruntime/, and tests/CMakeLists.txt
# enters this directory only while that target exists.

enable_testing()

foreach(mode IN ITEMS normal missing-symbol missing-library malformed-report concurrent
                 cache-config missing-pipeline pipeline-failure zero-site)
  add_test(NAME AnncEpAotFixture.${mode}
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/ep_aot_fixture.py
      --ep-library $<TARGET_FILE:annc_ep> --mode ${mode})
  set_tests_properties(AnncEpAotFixture.${mode} PROPERTIES TIMEOUT 60)
endforeach()

# annc_execution_v2 sites: the context callbacks, the pool the context
# carries, and the failure paths of both.
foreach(mode IN ITEMS normal threads nested concurrent same-session error bad-slot)
  add_test(NAME AnncEpContextFixture.${mode}
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/ep_context_fixture.py
      --ep-library $<TARGET_FILE:annc_ep> --mode ${mode})
  set_tests_properties(AnncEpContextFixture.${mode} PROPERTIES TIMEOUT 120)
endforeach()

# Library search directories shared by the AOT/context e2e drivers.  The
# drivers merge every *.a from --extra-lib-dir and the LLVM archive directory
# into one -L directory, which is what the annc driver expects.
set(AnncEpKernelLibArgs "")
if(TARGET LLVMSupport)
  list(APPEND AnncEpKernelLibArgs
       --llvm-lib-dir $<TARGET_FILE_DIR:LLVMSupport>)
endif()
foreach(lib_target IN ITEMS ANNCBuiltinKernels ANNCGemmSupport)
  if(TARGET ${lib_target})
    list(APPEND AnncEpKernelLibArgs
         --extra-lib-dir $<TARGET_FILE_DIR:${lib_target}>)
  endif()
endforeach()
if(TARGET ANNC::AArch64GemmMicrokernels)
  list(APPEND AnncEpKernelLibArgs
       --extra-lib-dir $<TARGET_FILE_DIR:ANNC::AArch64GemmMicrokernels>)
endif()
# MLAS-backed kernels resolve the ORT MLAS/KleidiAI/cpuinfo archives at link
# time.  They live in the EP build tree and its _deps subdirectories.
if(ANNC_KERNEL_MLAS AND ANNC_EP_ORT_BUILD_TREE)
  foreach(mlas_dir IN ITEMS
      "${ANNC_EP_ORT_BUILD_TREE}"
      "${ANNC_EP_ORT_BUILD_TREE}/_deps/kleidiai-build"
      "${ANNC_EP_ORT_BUILD_TREE}/_deps/pytorch_cpuinfo-build")
    if(EXISTS "${mlas_dir}")
      list(APPEND AnncEpKernelLibArgs
           --extra-lib-dir "${mlas_dir}")
    endif()
  endforeach()
endif()

# The legacy AOT e2e drives the pipeline from this tree; keep it runnable
# without relying on a developer's installed dev pipeline.
if(TARGET annc-onnx-pipeline)
  add_test(NAME AnncEpFtdmmAotE2e
    COMMAND ${CMAKE_COMMAND} -E env
      "ANNC_EP_PIPELINE=$<TARGET_FILE:annc-onnx-pipeline>"
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/ftdmm_aot_e2e.py
      --ep-library $<TARGET_FILE:annc_ep>
      ${AnncEpKernelLibArgs})
else()
  add_test(NAME AnncEpFtdmmAotE2e
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/ftdmm_aot_e2e.py
      --ep-library $<TARGET_FILE:annc_ep>
      ${AnncEpKernelLibArgs})
endif()
set_tests_properties(AnncEpFtdmmAotE2e PROPERTIES TIMEOUT 300)

# One fused pattern per switch; each is run enabled (claimed) and disabled
# (the EP drops the site and the CPU EP runs it).
add_test(NAME AnncEpFusionToggles
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/fusion_toggles_e2e.py
    --ep-library $<TARGET_FILE:annc_ep>)
set_tests_properties(AnncEpFusionToggles PROPERTIES TIMEOUT 300)

# The context-ABI FTDMM needs the pipeline and kernel library built from
# this tree (the legacy e2e can also run against an installed dev pipeline).
if(TARGET annc-onnx-pipeline)
  add_test(NAME AnncEpFtdmmContextE2e
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/ftdmm_context_e2e.py
      --ep-library $<TARGET_FILE:annc_ep>
      --pipeline $<TARGET_FILE:annc-onnx-pipeline>
      --kernel-lib-dir $<TARGET_FILE_DIR:ANNCBuiltinKernels>
      ${AnncEpKernelLibArgs})
  set_tests_properties(AnncEpFtdmmContextE2e PROPERTIES TIMEOUT 600)
  # masked_mul: a site whose arguments need name binding and whose two
  # outputs need the report->fused-node permutation.
  add_test(NAME AnncEpMaskedMulContextE2e
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/masked_mul_context_e2e.py
      --ep-library $<TARGET_FILE:annc_ep>
      --pipeline $<TARGET_FILE:annc-onnx-pipeline>
      --kernel-lib-dir $<TARGET_FILE_DIR:ANNCBuiltinKernels>
      ${AnncEpKernelLibArgs})
  set_tests_properties(AnncEpMaskedMulContextE2e PROPERTIES TIMEOUT 600)
  # layer_norm: rank-1 gamma/beta force name binding, eps is rank-0.
  add_test(NAME AnncEpLayerNormContextE2e
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm_context_e2e.py
      --ep-library $<TARGET_FILE:annc_ep>
      --pipeline $<TARGET_FILE:annc-onnx-pipeline>
      --kernel-lib-dir $<TARGET_FILE_DIR:ANNCBuiltinKernels>
      ${AnncEpKernelLibArgs})
  set_tests_properties(AnncEpLayerNormContextE2e PROPERTIES TIMEOUT 600)
  # batched_matmul_add: all three arguments are rank-3 f32 (name binding).
  add_test(NAME AnncEpBatchedMatMulAddContextE2e
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/batched_matmul_add_context_e2e.py
      --ep-library $<TARGET_FILE:annc_ep>
      --pipeline $<TARGET_FILE:annc-onnx-pipeline>
      --kernel-lib-dir $<TARGET_FILE_DIR:ANNCBuiltinKernels>
      ${AnncEpKernelLibArgs})
  set_tests_properties(AnncEpBatchedMatMulAddContextE2e PROPERTIES TIMEOUT 600)
endif()
