add_library(ANNCThreadPool STATIC
    ThreadPool.cpp
    Parallel.cpp
)

install(TARGETS ANNCThreadPool
        EXPORT MLIRTargets
        ARCHIVE DESTINATION lib)
