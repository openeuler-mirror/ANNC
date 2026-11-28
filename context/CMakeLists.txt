add_library(ANNCContext INTERFACE)
target_include_directories(ANNCContext INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(TARGETS ANNCContext EXPORT MLIRTargets)
install(DIRECTORY include/Context DESTINATION include)

# Embed the C accessors in hosts even when their only callers are dlopen'ed
# generated kernels. The archive is also available for standalone consumers.
add_library(ANNCContextMemRefObjects OBJECT lib/MemRef.cpp)
target_link_libraries(ANNCContextMemRefObjects PUBLIC ANNCContext)
set_target_properties(ANNCContextMemRefObjects PROPERTIES
  POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
  CXX_VISIBILITY_PRESET default)
add_library(ANNCContextMemRef STATIC
  $<TARGET_OBJECTS:ANNCContextMemRefObjects>)
target_link_libraries(ANNCContextMemRef PUBLIC ANNCContext)
install(TARGETS ANNCContextMemRef EXPORT MLIRTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Framework adapters implement the pool; this library only bridges borrowed
# handles for generated-code callers and never creates worker threads.
add_library(ANNCThreadPool STATIC lib/ThreadPool.cpp lib/Parallel.cpp)
target_link_libraries(ANNCThreadPool PUBLIC ANNCContext)
install(TARGETS ANNCThreadPool EXPORT MLIRTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
