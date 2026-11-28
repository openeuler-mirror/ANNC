#ifndef ANNC_TENSORFLOW_PLUGIN_1_15_THREADPOOL_CALLBACKS_H
#define ANNC_TENSORFLOW_PLUGIN_1_15_THREADPOOL_CALLBACKS_H

#include "c_api.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace thread {
class ThreadPool;
}  // namespace thread
}  // namespace tensorflow

namespace annc {
namespace tensorflow {

// Borrows pool; it must outlive the ScopedThreadPool using these callbacks.
// Without a TF pool, pass null callbacks to ScopedThreadPool instead.
AnncThreadPoolCallbacks MakeThreadPoolCallbacks(
    ::tensorflow::thread::ThreadPool& pool);

}  // namespace tensorflow
}  // namespace annc

#endif  // ANNC_TENSORFLOW_PLUGIN_1_15_THREADPOOL_CALLBACKS_H
