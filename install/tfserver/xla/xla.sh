PATH_OF_PATCHES="$ANNC_PATH/install/xla"
patch -p1 < $PATH_OF_PATCHES/cpu.patch
patch -p1 < $PATH_OF_PATCHES/mlir_hlo.patch
patch -p1 < $PATH_OF_PATCHES/option.patch
patch -p1 < $PATH_OF_PATCHES/debug.patch
patch -p1 < $PATH_OF_PATCHES/matmul_lower.patch
