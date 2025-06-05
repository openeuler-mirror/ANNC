package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "annc",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        "//annc/service:hlo_util",
        "//annc/service:optimizer_util",
        "//annc/service:kdnn_util",
        "//annc/service:blas_util",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_pass",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_target_registry",
        "@org_tensorflow//tensorflow/compiler/xla/service:pattern_matcher",
        "@openblas//:openblas"
    ],
)

genrule(
    name = "generate_empty_hpc",
    outs = ["dummy.c"],
    cmd = "touch $@",
)

cc_binary(
    name = "hpc.so",
    srcs = [":dummy.c"],
    linkshared = True,
)

cc_library(
    name = "cambridge",
    srcs = glob(["xla/*.cpp", "xla/*.cc", "llvm/*.cpp", "hpc.so"]),
    linkstatic = True,
    alwayslink = True,
    visibility = ["//visibility:public"],
    deps = [
        "@xla//xla/mlir_hlo:transforms_passes",
        "@xla//xla/mlir_hlo:thlo_passes",
        ":hpc.so"
    ],
)

cc_binary(
    name = "libannc.so",
    srcs = glob(["*.cc", "*.h"]),
    linkshared = True,
    deps = [
        "//annc/service:hlo_util",
        "//annc/service:optimizer_util",
        "//annc/service:kdnn_util",
        "//annc/service:blas_util",
        #":cambridge",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_pass",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_target_registry",
        "@org_tensorflow//tensorflow/compiler/xla/service:pattern_matcher",
        "@openblas//:openblas",
    ],
)
