genrule(
    name = "build_openblas",
    srcs = glob([
        "**/*"],
        exclude = ["**/*.a", "**/*.so"]),
    outs = ["libopenblas.a"],
    cmd = """
        cd $$(dirname $(location //:README.md)) && \
        make CFLAGS="-march=armv8-a" NO_SHARED=1 ONLY_CBLAS=1 && \
        cd - && \
        cp $$(dirname $(location //:README.md))/libopenblas_*.a $@
    """,
)

cc_import(
    name = "openblas",
    static_library = "libopenblas.a",
    visibility = ["//visibility:public"],
)
