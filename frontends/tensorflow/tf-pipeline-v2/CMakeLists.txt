# annc-tf-pipeline-v2: offline orchestrator for the TF stablehlo V2 route
# (entered with ANNC_ENABLE_TF2ATIR_V2). Pure process orchestration - no
# MLIR library is linked, so the tool stays out of the LLVM/MLIR link
# closure (mirrors frontends/tensorflow/tf-pipeline).
# Headers-only SDK builds skip the offline converter executable; the
# orchestrator stays buildable (its Step 8 needs --skip-converter there).
if(NOT TARGET annc-converter-v2)
  message(STATUS
    "annc-tf-pipeline-v2: offline converter executable not built "
    "(headers-only TF SDK); use --skip-converter at run time")
endif()
add_executable(annc-tf-pipeline-v2
  annc-tf-pipeline-v2.cpp
)
install(TARGETS annc-tf-pipeline-v2 DESTINATION bin)
