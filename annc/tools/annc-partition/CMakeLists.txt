execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE ANNC_SOURCE_REVISION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT ANNC_SOURCE_REVISION)
  set(ANNC_SOURCE_REVISION "unknown")
endif()

add_executable(annc-partition
  annc-partition.cpp
  PartitionTask.cpp
)
target_link_libraries(annc-partition PRIVATE
  LLVMSupport
)
target_compile_definitions(annc-partition PRIVATE
  ANNC_PARTITION_VERSION="0.1.0"
  ANNC_SOURCE_REVISION="${ANNC_SOURCE_REVISION}"
)
llvm_update_compile_flags(annc-partition)
install(TARGETS annc-partition DESTINATION bin)
