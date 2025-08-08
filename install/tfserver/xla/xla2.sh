TF_PATCH_PATH="$ANNC_PATH/install"
patch -p1 < $TF_PATCH_PATH/tensorflow.patch
PATH_OF_PATCHES="$ANNC_PATH/install/xla"
patch -p1 < $PATH_OF_PATCHES/BUILD.patch
patch -p1 < $PATH_OF_PATCHES/cpu_compiler.patch
patch -p1 < $PATH_OF_PATCHES/debug_options_flags.cc.patch
patch -p1 < $PATH_OF_PATCHES/hlo_xla_runtime_pipeline.h.patch
patch -p1 < $PATH_OF_PATCHES/simple_orc_jit.cc.patch
patch -p1 < $PATH_OF_PATCHES/cpu_runtime.h.patch
patch -p1 < $PATH_OF_PATCHES/cpu_runtime.cc.patch
patch -p1 < $PATH_OF_PATCHES/xla.proto.patch
patch -p1 < $PATH_OF_PATCHES/ir_emitter.h.patch
patch -p1 < $PATH_OF_PATCHES/ir_emitter.cc.patch
patch -p1 < $PATH_OF_PATCHES/remap-subgraph.patch
