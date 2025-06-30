load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def load_openblas():
    http_archive(
        name = "openblas",
        build_file = "//third_party/openblas:BUILD",
        strip_prefix = "OpenBLAS-0.3.29",
        urls = [
            "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29.tar.gz",
        ],
        sha256 = "38240eee1b29e2bde47ebb5d61160207dc68668a54cac62c076bb5032013b1eb",
    )
