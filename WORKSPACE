workspace(name = "aicompiler")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//tools:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("6.1.0")

load("//third_party:repo.bzl", "load_openblas")

load_openblas()

http_archive(
    name = "xla",
    patch_args = ["-p1"],
    patches = [
        "//install:openxla.patch",
        "//install:llvm.patch",
        "//install/llvm:annc1.patch",
        "//install/llvm:annc2.patch",
        "//install/llvm:annc3.patch",
        "//install/llvm:annc4.patch",
        "//install/xla:concat.patch",
        "//install/xla:fix.patch",
        "//install/xla:variables.patch",
        "//install/xla:profiling.patch",
        "//install/xla:dffm.patch",
    ],
    sha256 = "90e72fa3558a69cf2562e4600e62c478d22c3986c642d7dcdc7ef0841ded52c5",
    strip_prefix = "xla-40008cb2c85749ae436be61c40d3279cb24705c7",
    urls = [
        "https://github.com/openxla/xla/archive/40008cb2c85749ae436be61c40d3279cb24705c7.tar.gz",
    ],
)

# Initialize OpenXLA's external dependencies.
load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()
