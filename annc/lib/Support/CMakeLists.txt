# Shared thread-pool support is defined by the root context/ module.

add_library(ANNCCheckpoint STATIC CheckpointReader.cpp)
target_include_directories(ANNCCheckpoint PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/annc/include>
  $<INSTALL_INTERFACE:include>)
set_target_properties(ANNCCheckpoint PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS ANNCCheckpoint EXPORT MLIRTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
