workspace(name = "aicompiler")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//tools:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("2.0.0")

load("//third_party:repo.bzl", "load_openblas")

load_openblas()

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

http_archive(
    name = "mkl_linux",
    sha256 = "a936d6b277a33d2a027a024ea8e65df62bd2e162c7ca52c48486ed9d5dc27160",
    strip_prefix = "mklml_lnx_2019.0.5.20190502",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/intel/mkl-dnn/releases/download/v0.21/mklml_lnx_2019.0.5.20190502.tgz",
        "https://github.com/intel/mkl-dnn/releases/download/v0.21/mklml_lnx_2019.0.5.20190502.tgz",
    ],
)

http_archive(
    name = "org_tensorflow",
    sha256 = "ed8b08aec9b1cebcb4af75c5a42b50e316afddf3ed4d3d219310f205230d4ff6",
    strip_prefix = "tensorflow-cfe0c80169ae984bcdc99ff6de7444164aaa8e07",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/cfe0c80169ae984bcdc99ff6de7444164aaa8e07.tar.gz",
    ],
    patch_args = ["-p1"],
    patches = ["//install:openxla.patch"],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace()
