add_subdirectory(converter)
add_subdirectory(tf2atir)
add_subdirectory(tf-pipeline)
# TF stablehlo V2 route (opt-in, ANNC_ENABLE_TF2ATIR_V2): the offline
# pipeline orchestrator over the whole V2 chain (its sibling tools all
# live in the same build/bin; converter-v2 is registered separately).
if(ANNC_ENABLE_TF2ATIR_V2)
  add_subdirectory(tf-pipeline-v2)
endif()
