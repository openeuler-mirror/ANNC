package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_import(
    name = "aicompiler",
    hdrs = [
        "kdnn_rewriter.h",
        "auto_tiling.h",
        "mix_precsions.h",
        "quantize.h",
    ],
    shared_library = ":aicompiler.so",
    visibility = ["//visibility:public"]
)

cc_binary(
    name = "aicompiler.so",
    srcs = glob(["*.cc", "*.h"]),
    linkshared = True,
    deps = [
        "//aicompiler/service:optimizer_util",
        "//aicompiler/service:kdnn_util",
        "@xla//xla/hlo/ir:hlo",
    ],
)
