#ifndef ANNC_TENSORFLOW_PLUGIN_COMPAT_H
#define ANNC_TENSORFLOW_PLUGIN_COMPAT_H

#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"

#if !(TF_MAJOR_VERSION >= 2 && TF_MINOR_VERSION >= 2)
namespace tensorflow {
using tstring = std::string;
}
#endif

#ifndef TF_GUARDED_BY
#define TF_GUARDED_BY(x) GUARDED_BY(x)
#endif

#if TF_MAJOR_VERSION < 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION < 10)
#define OkStatus() ::tensorflow::Status()
#endif

#endif  // ANNC_TENSORFLOW_PLUGIN_COMPAT_H
