add_library(ANNCContext INTERFACE)
target_include_directories(ANNCContext INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(TARGETS ANNCContext EXPORT MLIRTargets)
install(DIRECTORY include/Context DESTINATION include)

# Framework adapters implement the pool; this library only bridges borrowed
# handles for generated-code callers and never creates worker threads.
add_library(ANNCThreadPool STATIC lib/ThreadPool.cpp lib/Parallel.cpp)
target_link_libraries(ANNCThreadPool PUBLIC ANNCContext)
install(TARGETS ANNCThreadPool EXPORT MLIRTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
