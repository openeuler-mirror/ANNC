TF_PATH=""
ORG_TF_PATH=""
XLA_PATH=""

TF_PATCH_PATH="$ANNC/install"
if [ ! -d "${ORG_TF_PATH}" ]; then
    echo "The ORG_TF_PATH should be set."
fi
cd $ORG_TF_PATH
patch -p1 < $TF_PATCH_PATH/tensorflow.patch

PATH_OF_PATCHES="$ANNC/install/xla"
if [ ! -d "${TF_PATH}" ]; then
    echo "The TF_PATH should be set."
fi
cd $TF_PATH
patch -p1 < $PATH_OF_PATCHES/cpu_compiler.patch
patch -p1 < $PATH_OF_PATCHES/simple_orc_jit.cc.patch
patch -p1 < $PATH_OF_PATCHES/cpu_runtime.h.patch
patch -p1 < $PATH_OF_PATCHES/cpu_runtime.cc.patch
patch -p1 < $PATH_OF_PATCHES/ir_emitter.h.patch
patch -p1 < $PATH_OF_PATCHES/ir_emitter.cc.patch
patch -p1 < $PATH_OF_PATCHES/cpu_instruction_fusion.h.patch

if [ ! -d "${XLA_PATH}" ]; then
    echo "The XLA_PATH should be set."
fi
cd $XLA_PATH
patch -p1 < $PATH_OF_PATCHES/BUILD.patch
patch -p1 < $PATH_OF_PATCHES/debug_options_flags.cc.patch
patch -p1 < $PATH_OF_PATCHES/hlo_xla_runtime_pipeline.h.patch
patch -p1 < $PATH_OF_PATCHES/xla.proto.patch