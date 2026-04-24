#!/bin/bash
PASSES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

patternName="$1"
iter="$2"

# compile generated file
echo "Compiling ${patternName}.cc"

# bazel --output_user_root=./output build //tensorflow_serving/custom_ops/${patternName}:${patternName} > ${PASSES_DIR}/outputs/compile_outputs/${patternName}${iter}.txt 2>&1
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++14 -shared ${PASSES_DIR}/outputs/generated_files/${iter}${patternName}.cc -o ${PASSES_DIR}/outputs/so/${patternName}.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 > ${PASSES_DIR}/outputs/compile_outputs/${patternName}${iter}.txt 2>&1

echo "Compilation Complete: ${PASSES_DIR}/outputs/compile_outputs/${patternName}${iter}.txt"