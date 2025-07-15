package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "annc_model_opt",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]) + [
        "//tensorflow/core/public:session.h",
        "//tensorflow/core/public:session_options.h",
    ],
    linkstatic = True,
    alwayslink = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/util/tensor_bundle",
    ],
)

cc_binary(
    name = "libannc_graph_opt.so",
    srcs = glob(["*.cc", "*.h"]) + [
        "//tensorflow/core/public:session.h",
        "//tensorflow/core/public:session_options.h",
    ],
    linkshared = True,
    deps = [
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/util/tensor_bundle",
    ],
)
