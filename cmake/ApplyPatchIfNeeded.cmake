if(NOT DEFINED ANNC_PATCH_FILE OR NOT DEFINED ANNC_PATCH_WORKDIR)
    message(FATAL_ERROR "ANNC_PATCH_FILE and ANNC_PATCH_WORKDIR are required")
endif()

execute_process(
    COMMAND git apply --unidiff-zero --check "${ANNC_PATCH_FILE}"
    WORKING_DIRECTORY "${ANNC_PATCH_WORKDIR}"
    RESULT_VARIABLE patch_check_result
)

if(patch_check_result EQUAL 0)
    execute_process(
        COMMAND git apply --unidiff-zero "${ANNC_PATCH_FILE}"
        WORKING_DIRECTORY "${ANNC_PATCH_WORKDIR}"
        RESULT_VARIABLE patch_apply_result
    )
    if(NOT patch_apply_result EQUAL 0)
        message(FATAL_ERROR "Failed to apply patch: ${ANNC_PATCH_FILE}")
    endif()
else()
    execute_process(
        COMMAND git apply --unidiff-zero --reverse --check "${ANNC_PATCH_FILE}"
        WORKING_DIRECTORY "${ANNC_PATCH_WORKDIR}"
        RESULT_VARIABLE patch_reverse_check_result
    )
    if(NOT patch_reverse_check_result EQUAL 0)
        message(FATAL_ERROR
            "Patch is neither applicable nor already applied: ${ANNC_PATCH_FILE}")
    endif()
endif()
