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

cc_library(
    name = "cambridge",
    srcs = ["xla/kernel_selector.cc",
            "xla/kernel_selector_ops_rewriter.cc",
            "xla/xnnpack_ops.cc",
            "xla/xnnpack_ops_rewriter.cc",
            ],
    hdrs = ["xla/kernel_selector.h",
            "xla/kernel_selector_ops_rewriter.h",
            "xla/xnnpack_ops.h",
            "xla/xnnpack_ops_rewriter.h",
            "xla/xnnpack_pattern_utils.h",
            ],
    linkstatic = True,
    alwayslink = True,
    visibility = ["//visibility:public"],
    deps = [
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_pass",
        "@xla//xla:literal_comparison",
        "@xla//xla:literal_util",
        "@xla//xla/service/cpu:cpu_runtime",
        "@xla//xla/service/cpu:simple_orc_jit",
        "@xla//xla/service/cpu:runtime_lightweight_check",
        "@xla//xla:executable_run_options",
        "@xla//xla/service/cpu:ir_emitter",
        "@xla//xla/service:pattern_matcher",
        ":libmlir",
        ":libXNNPACK",
    ],
)

cc_import(
    name = "libXNNPACK",
    visibility = ["//visibility:public"],
    shared_library = "xla/libs/libXNNPACK.so",
    system_provided = 0
)

cc_import(
  name = "libmlir",
  visibility = ["//visibility:public"],
  shared_library = "xla/libs/libblas_920b_single_mlir.so",
  system_provided = 0,
)

cc_binary(
    name = "libannc.so",
    srcs = glob(["*.cc", "*.h"]),
    copts = ["-fopenmp"],
    linkshared = True,
    deps = [
        "//annc/service:hlo_util",
        "//annc/service:optimizer_util",
        "//annc/service:kdnn_util",
        "//annc/service:blas_util",
        "//annc/service/bisheng-cpu:bisheng-cpu",
        ":cambridge",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_pass",
        "@xla//xla/service:custom_call_target_registry",
        "@xla//xla/service:pattern_matcher",
        "@openblas//:openblas",
    ],
)
