genrule(
    name = "build_kpgemm",
    srcs = glob(["**"], exclude = ["**/*.a", "**/*.so"]),
    outs = ["libkpgemm.a"],
    cmd = """
        SRC_REL_PATH=$(execpath :CMakeLists.txt) && \
        SRC_DIR=$$(dirname $$SRC_REL_PATH) && \
        ABS_SRC_DIR=$$(pwd)/$$SRC_DIR && \
        BUILD_DIR=$$(pwd)/build_kpgemm && \
        mkdir -p "$$BUILD_DIR" && \
        pushd "$$BUILD_DIR" && \
        cmake "$$ABS_SRC_DIR" -DBUILD_SHARED_LIBS=OFF && \
        make -j16 && \
        popd >/dev/null && \
        cp $$BUILD_DIR/lib/libkpgemm.a $$BUILD_DIR/lib/libkpgemm_private.a && \
        nm -g --defined-only $$BUILD_DIR/lib/libkpgemm_private.a | \
          awk '$$2 ~ /^[A-Za-z]$$/ && $$3 !~ /^kp_/ { print $$3 " kpgemm_private_" $$3 }' | \
          sort -u > $$BUILD_DIR/kpgemm_rename_syms.txt && \
        objcopy --redefine-syms=$$BUILD_DIR/kpgemm_rename_syms.txt $$BUILD_DIR/lib/libkpgemm_private.a && \
        ranlib $$BUILD_DIR/lib/libkpgemm_private.a && \
        cp $$BUILD_DIR/lib/libkpgemm_private.a $(location libkpgemm.a)
    """,
)

cc_import(
    name = "kpgemm",
    static_library = "libkpgemm.a",
    visibility = ["//visibility:public"],
)
