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
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_pass",
        "@xla//xla/service:custom_call_target_registry",
        "@xla//xla/service:pattern_matcher",
        "@openblas//:openblas"
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
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_pass",
        "@xla//xla/service:custom_call_target_registry",
        "@xla//xla/service:pattern_matcher",
        "@xla//xla/mlir_hlo:mlir_hlo",
        "@xla//xla/mlir_hlo:thlo_passes",
        "@xla//xla/mlir_hlo:transforms_passes",
        "@openblas//:openblas",
    ],
)
