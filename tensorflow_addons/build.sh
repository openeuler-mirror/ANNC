#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

MODE="${1:-serving}"

echo "=========================================="
echo "ANNC TensorFlow Addons Build"
echo "=========================================="
echo ""

if [ "$MODE" = "serving" ]; then
    echo "Mode: TF Serving (LD_PRELOAD, no TF linkage)"
    echo ""
    echo "Architecture:"
    echo "  libannc_optimizer.so  — LD_PRELOAD plugin (no TF linkage)"
    echo "  libannc_fused_op.so   — dlopen'd by optimizer (no TF linkage)"
    echo "  TF symbols resolved from tensorflow_model_server (-rdynamic)"
    CMAKE_EXTRA="-DANNC_FOR_TF_SERVING=ON"
elif [ "$MODE" = "pip" ]; then
    echo "Mode: pip TensorFlow (linked against tensorflow_cc)"
    echo ""
    echo "Architecture:"
    echo "  libannc_optimizer.so  — linked against tensorflow_cc"
    echo "  libannc_fused_op.so   — linked against tensorflow_cc"
    CMAKE_EXTRA="-DANNC_FOR_TF_SERVING=OFF"
else
    echo "Usage: $0 [serving|pip]"
    echo "  serving  — Build for TF Serving (LD_PRELOAD mode, default)"
    echo "  pip      — Build for pip-installed TensorFlow"
    exit 1
fi

echo ""
echo "[1/1] Building ANNC addons ..."

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA}

cmake --build . -j$(nproc 2>/dev/null || echo 4)

echo ""
echo "=========================================="
echo "Build complete!"
echo ""
echo "Output:"
ls -la libannc_optimizer.so libannc_fused_op.so 2>/dev/null

if [ "$MODE" = "serving" ]; then
    echo ""
    echo "── Usage ──"
    echo ""
    echo "  # 1. Create session config (see session_config.pbtxt):"
    echo "  #    graph_options.rewrite_options.custom_optimizers { name: \"ANNCOptimizer\" }"
    echo ""
    echo "  # 2. Start TF Serving with ANNC:"
    echo "  LD_PRELOAD=\"${BUILD_DIR}/libannc_optimizer.so\" \\"
    echo "  tensorflow_model_server \\"
    echo "    --port=8500 \\"
    echo "    --model_name=my_model \\"
    echo "    --model_base_path=/path/to/model \\"
    echo "    --tensorflow_session_config_file=session_config.pbtxt"
    echo ""
    echo "  # 3. Environment variables (all optional):"
    echo "  export ANNC_ENABLE=1"
    echo "  export ANNC_PIPELINE_PATH=/path/to/annc-tf-pipeline"
    echo "  export ANNC_WORK_DIR=/tmp/annc_work"
    echo "  export ANNC_VERBOSE=0"
    echo "  export ANNC_TIMEOUT=300"
    echo "  export ANNC_KEEP_TEMPS=0"
    echo "  export ANNC_FUSED_OP_PATH=/path/to/libannc_fused_op.so  # auto-detected"
fi
echo "=========================================="
